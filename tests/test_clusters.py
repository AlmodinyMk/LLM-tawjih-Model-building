"""Model servers on Kubernetes (kubectl) and through Ansible."""
import json
import os
import unittest
from pathlib import Path

from test_tawjih import Base
from tawjihlib import clusters as C

try:
    import yaml
except ImportError:          # the tests that read YAML back need PyYAML
    yaml = None

KUBECTL = """#!/bin/sh
# stand-in: notes how it was called and what it read
echo "$*" >> "%(log)s"
case "$*" in
  *apply*) cat > "%(log)s.stdin" ;;
esac
case "$*" in
  *get*) cat <<'X'
{"items": [
 {"kind": "Deployment", "metadata": {"name": "tawjih-big"},
  "spec": {"replicas": 2}, "status": {"readyReplicas": 1,
                                      "updatedReplicas": 2}},
 {"kind": "Service", "metadata": {"name": "tawjih-big"},
  "spec": {"clusterIP": "10.0.0.7", "ports": [{"port": 8000}]}},
 {"kind": "Pod", "metadata": {"name": "tawjih-big-1"},
  "status": {"phase": "Pending", "containerStatuses": [
   {"ready": false, "restartCount": 3,
    "state": {"waiting": {"reason": "ImagePullBackOff"}}}]}}]}
X
  ;;
esac
"""
PLAYBOOK = """#!/bin/sh
echo "$*" >> "%(log)s"
"""


class Base2(Base):
    def setUp(self):
        super().setUp()
        self.cli("server", "add", "big", "--engine", "vllm", "--model",
                 "Qwen/Qwen3-32B", "--tp", "2", "--gpus", "0,1", "--port",
                 "8000", "--api-key-env", "BIG_KEY")
        self.cli("server", "add", "own", "--engine", "custom", "--command",
                 "python3 serve.py --port {port}", "--port", "9000")

    def cfg(self, name):
        return json.loads((self.root / "servers.json").read_text(
            encoding="utf-8"))["servers"][name]


class Manifests(Base2):
    def test_deployment_and_service(self):
        docs = C.manifest("big", self.cfg("big"), replicas=2,
                          namespace="ml", claim="models")
        dep, svc = docs
        box = dep["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(box["image"], "vllm/vllm-openai:v0.32.0")
        self.assertEqual(box["command"][:3], ["vllm", "serve",
                                              "Qwen/Qwen3-32B"])
        self.assertIn("0.0.0.0", box["command"])
        self.assertEqual(box["resources"]["limits"]["nvidia.com/gpu"], 2)
        self.assertEqual(dep["spec"]["replicas"], 2)
        self.assertEqual(dep["metadata"]["namespace"], "ml")
        # the key comes from a Secret, never into the manifest
        keyed = [e for e in box["env"] if "valueFrom" in e]
        self.assertEqual(keyed[0]["valueFrom"]["secretKeyRef"],
                         {"name": "tawjih-big-key", "key": "BIG_KEY"})
        self.assertFalse(any(e["name"] == "CUDA_VISIBLE_DEVICES"
                             for e in box["env"]))
        self.assertEqual(box["readinessProbe"]["httpGet"]["path"], "/health")
        claim = dep["spec"]["template"]["spec"]["volumes"][1]
        self.assertEqual(claim["persistentVolumeClaim"]["claimName"],
                         "models")
        self.assertEqual(svc["spec"]["ports"][0]["port"], 8000)

    @unittest.skipIf(yaml is None, "PyYAML is not installed")
    def test_yaml_reads_back(self):
        text = C.manifest_text("big", self.cfg("big"))
        self.assertIn("kubectl create secret generic tawjih-big-key", text)
        docs = list(yaml.safe_load_all(text))
        self.assertEqual(docs, C.manifest("big", self.cfg("big")))
        book = C.playbook("big", self.cfg("big"), install=True,
                          listen="0.0.0.0")
        self.assertEqual(yaml.safe_load(C.to_yaml(book)), book)

    def test_cli(self):
        code, out, err = self.cli("k8s", "manifest", "big", "--replicas", "3")
        self.assertEqual(code, 0, err)
        self.assertIn('replicas: 3', out)
        code, _, err = self.cli("k8s", "manifest", "own")
        self.assertNotEqual(code, 0)
        self.assertIn("give --image", err)
        code, out, _ = self.cli("عنقود", "manifest", "own", "--image",
                                "me/serve:1")
        self.assertIn('"me/serve:1"', out)

    def test_playbook(self):
        book = C.playbook("big", self.cfg("big"), install=True,
                          listen="0.0.0.0")[0]
        names = [t["name"] for t in book["tasks"]]
        self.assertEqual(names[0], "install vllm")
        key = [t for t in book["tasks"] if t.get("no_log")][0]
        self.assertIn("lookup('env', 'BIG_KEY')",
                      key["ansible.builtin.copy"]["content"])
        unit = [t for t in book["tasks"] if t["name"] == "the unit"][0][
            "ansible.builtin.copy"]["content"]
        self.assertIn("ExecStart=/usr/bin/env vllm serve Qwen/Qwen3-32B "
                      "--host 0.0.0.0", unit)
        self.assertIn("EnvironmentFile=/etc/tawjih/big.env", unit)
        stopped = C.playbook("big", self.cfg("big"), state="stopped")[0]
        self.assertFalse(any(t["name"] == "wait until it answers"
                             for t in stopped["tasks"]))
        with self.assertRaises(tawjih_die()):
            C.playbook("own", self.cfg("own"), install=True)


def tawjih_die():
    from tawjihlib.core import Die
    return Die


@unittest.skipIf(os.name == "nt", "the stand-ins are shell scripts")
class Tools(Base2):
    def setUp(self):
        super().setUp()
        self.log = self.root / "calls.log"
        for env, name, body in (("KUBECTL", "kubectl", KUBECTL),
                                ("ANSIBLE_PLAYBOOK", "ansible-playbook",
                                 PLAYBOOK)):
            f = self.root / name
            f.write_text(body % {"log": self.log}, encoding="utf-8")
            f.chmod(0o755)
            os.environ[env] = str(f)
            self.addCleanup(os.environ.pop, env, None)

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def history(self):
        lines = []
        for f in (self.root / "history").glob("servers-*.jsonl"):
            lines += [json.loads(x) for x in f.read_text().splitlines()]
        return lines

    def test_status(self):
        code, out, err = self.cli("k8s", "status", "--namespace", "ml")
        self.assertEqual(code, 0, err)
        self.assertIn("--namespace ml get deployment,service,pod -l "
                      "app.kubernetes.io/managed-by=tawjih -o json",
                      self.calls()[0])
        self.assertIn("1/2 ready", out)
        self.assertIn("10.0.0.7:8000", out)
        self.assertIn("Pending, 3 restarts (ImagePullBackOff)", out)

    def test_apply_is_checked_first(self):
        code, out, _ = self.cli("k8s", "apply", "big")
        self.assertEqual(code, 0)
        self.assertIn("apply --dry-run=server -f -", self.calls()[0])
        self.assertIn("nothing changed", out)
        self.assertEqual(self.history(), [])
        sent = Path(str(self.log) + ".stdin").read_text()
        self.assertIn("tawjih-big", sent)
        code, out, _ = self.cli("k8s", "apply", "big", "--yes", "--context",
                                "prod")
        self.assertEqual(code, 0)
        self.assertEqual(self.calls()[1], "--context prod apply -f -")
        h = self.history()[-1]
        self.assertEqual((h["action"], h["server"], h["context"]),
                         ("k8s apply", "big", "prod"))

    def test_changes_need_yes(self):
        for args in (("scale", "big", "--replicas", "4"),
                     ("rollback", "big"), ("delete", "big")):
            code, out, _ = self.cli("k8s", *args)
            self.assertEqual(code, 1)
            self.assertIn("add --yes", out)
        self.assertEqual(self.calls(), [])
        self.cli("k8s", "scale", "big", "--replicas", "4", "--yes")
        self.cli("k8s", "rollback", "big", "--yes")
        self.cli("k8s", "delete", "big", "--yes")
        self.assertEqual(self.calls(), [
            "scale deployment/tawjih-big --replicas=4",
            "rollout undo deployment/tawjih-big",
            "delete deployment,service -l app.kubernetes.io/name=tawjih-big"])
        self.assertEqual([h["action"] for h in self.history()],
                         ["k8s scale", "k8s rollback", "k8s delete"])

    def test_policy(self):
        self.cli("policy", "set", "--allow-network", "local")
        code, _, err = self.cli("k8s", "status")
        self.assertNotEqual(code, 0)
        self.assertIn("does not allow the network", err)
        code, out, _ = self.cli("k8s", "manifest", "big")   # stays here
        self.assertEqual(code, 0)
        self.assertEqual(self.calls(), [])

    def test_ansible(self):
        code, out, err = self.cli("ansible", "run", "big", "--hosts",
                                  "gpu1,ops@gpu2")
        self.assertEqual(code, 0, err)
        inv = (self.root / "servers" / "ansible" / "hosts.ini").read_text()
        self.assertEqual(inv, "[tawjih]\ngpu1\ngpu2 ansible_user=ops\n")
        self.assertTrue(self.calls()[0].endswith("big.yml --check --diff"))
        self.assertIn("nothing changed", out)
        self.assertEqual(self.history(), [])
        self.cli("ansible", "run", "big", "--hosts", "gpu1", "--yes",
                 "--state", "restarted", "--limit", "gpu1")
        self.assertTrue(self.calls()[1].endswith("big.yml --limit gpu1"))
        self.assertEqual(self.history()[-1]["action"], "ansible restarted")
        code, _, err = self.cli("ansible", "run", "big", "--hosts", "a;b")
        self.assertNotEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
