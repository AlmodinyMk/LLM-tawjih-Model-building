"""Tests for plugins/runtimes.py and the bridge's runtime command: jobs on
this computer, and on a stand-in for a remote service driven by commands.
No training library is needed: a stand-in Python writes the adapter."""
import json
import os
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "plugins"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import neural_bridge as nb  # noqa: E402
from test_tawjih import Base  # noqa: E402

EX = "Q: 2x+4=10\nA: x = 3\nE: 3\n"


class Bridge(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.fill("1.1.1", instructions="Answer with x only.", examples=EX)
        self.ds = self.root / "datasets"

    def nb(self, *args):
        import contextlib
        import io
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = nb.main(["-p", self.dir] + list(args))
        return code, out.getvalue(), err.getvalue()

    def compile(self):
        code, out, err = self.nb("compile", "-o", str(self.ds))
        self.assertEqual(code, 0, err + out)

FAKE_PY = """#!/bin/sh
# a stand-in for the training Python: reads the config, writes an adapter
echo "training with $@"
python3 - "$@" <<'EOF'
import json, os, sys
cfg = json.load(open(sys.argv[2]))
assert os.path.isfile(cfg["dataset"]), cfg["dataset"]
os.makedirs(cfg["output_dir"], exist_ok=True)
open(os.path.join(cfg["output_dir"], "adapter_config.json"), "w").write(
    json.dumps({"r": cfg["lora_r"]}))
EOF
"""


@unittest.skipIf(os.name == "nt", "jobs run with sh")
class Runtimes(Bridge):
    def setUp(self):
        super().setUp()
        os.environ["TAWJIH_RUNTIMES"] = str(self.root / "rt.json")
        self.addCleanup(os.environ.pop, "TAWJIH_RUNTIMES", None)
        self.py = self.root / "fakepy"
        self.py.write_text(FAKE_PY)
        self.py.chmod(0o755)
        self.cloud = self.root / "cloud"
        self.compile()

    def add_cloud(self):
        c = str(self.cloud)
        code, out, err = self.cli(
            "runtime", "add", "cloud", "--kind", "command",
            "--python", str(self.py),
            "--submit", "mkdir -p %s/{id} && cp -r {job}/. %s/{id}/ && "
            "sh %s/{id}/run.sh" % (c, c, c),
            "--status", "cat %s/{id}/exit" % c,
            "--logs", "cat %s/{id}/log.txt" % c,
            "--collect", "cp -r %s/{id}/out/. {job}/out/" % c)
        self.assertEqual(code, 0, err)

    def test_list_add_rm_default(self):
        out = self.cli("runtime", "list")[1]
        self.assertIn("local", out)
        self.add_cloud()
        self.assertIn("cloud", self.cli("runtime", "list")[1])
        self.assertEqual(self.cli("runtime", "default", "cloud")[0], 0)
        self.assertIn("(default)", [l for l in self.cli(
            "runtime", "list")[1].splitlines() if l.startswith("cloud")][0])
        self.assertEqual(self.cli("runtime", "rm", "cloud")[0], 0)
        data = json.loads((self.root / "rt.json").read_text())
        self.assertEqual(data["default"], "local")
        code, _, err = self.cli("runtime", "add", "x", "--kind", "ssh")
        self.assertIn("--host", err)
        code, _, err = self.cli("runtime", "add", "x", "--kind", "command",
                               "--submit", "run --api_key=abcdefghijkl")
        self.assertIn("key or password", err)

    def test_default_alone_and_no_jobs(self):
        self.assertIn("default runtime: local",
                      self.cli("runtime", "default")[1])
        self.assertIn("no job has been started", self.cli(
            "runtime", "status")[2])

    def test_check_here(self):
        code, out, err = self.cli("runtime", "check")
        self.assertEqual(code, 0, err)
        self.assertIn("Python", out)
        self.assertIn("GB free", out)

    def test_train_as_a_local_job(self):
        code, out, err = self.nb("--python", str(self.py), "train", "a1",
                                 "--base-model", "tiny", "--dataset",
                                 str(self.ds / "sft_train.jsonl"),
                                 "--runtime", "local", "--run")
        self.assertEqual(code, 0, err + out)
        adir = self.root / "models" / "adapters" / "a1"
        self.assertTrue((adir / "adapter_config.json").is_file())
        reg = json.loads((self.root / "models" / "registry.json")
                         .read_text())
        self.assertTrue(reg["adapters"]["a1"]["trained"])
        self.assertIn("collected", self.cli("runtime", "jobs")[1])

    def test_local_folder_model_and_merge_needs(self):
        base = self.root / "mybase"
        base.mkdir()
        old = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, old)
        code, out, err = self.nb("--python", str(self.py), "train", "a4",
                                 "--base-model", "mybase", "--dataset",
                                 str(self.ds / "sft_train.jsonl"),
                                 "--runtime", "local", "--run")
        self.assertEqual(code, 0, err + out)
        job = next((self.root / "models" / "jobs").iterdir())
        cfg = json.loads((job / "train_config.json").read_text())
        self.assertEqual(cfg["base_model"], str(base.resolve()))
        entry = json.loads((self.root / "models" / "registry.json")
                           .read_text())["adapters"]["a4"]
        self.assertEqual(entry["path"], "models/adapters/a4")
        self.assertEqual(entry["base"], "mybase")
        self.assertEqual(entry["peft"], "lora")
        code, out, err = self.nb("merge", "a4", "--runtime", "local")
        self.assertEqual(code, 0, err + out)
        self.assertIn(str(base.resolve()), out)

    def test_train_on_a_remote_service(self):
        self.add_cloud()
        args = ["train", "a2", "--base-model", "tiny", "--dataset",
                str(self.ds / "sft_train.jsonl"), "--runtime", "cloud"]
        code, out, err = self.nb(*args)
        self.assertEqual(code, 0, err)
        self.assertIn("plan only", out)
        self.assertFalse(self.cloud.exists())
        code, out, err = self.nb(*args, "--run")
        self.assertNotEqual(code, 0)
        self.assertIn("--yes", err)
        code, out, err = self.nb(*args, "--run", "--yes")
        self.assertEqual(code, 0, err)
        self.assertIn("runtime collect", out)
        self.assertIn(": done", self.cli("runtime", "status", "last")[1])
        self.assertIn("training with", self.cli("runtime", "logs",
                                               "last")[1])
        adir = self.root / "models" / "adapters" / "a2"
        self.assertFalse((adir / "adapter_config.json").is_file())
        code, out, err = self.cli("runtime", "collect", "last")
        self.assertEqual(code, 0, err)
        self.assertTrue((adir / "adapter_config.json").is_file())
        # the dataset went with the job, and no absolute path of here
        job = next(self.cloud.iterdir())
        cfg = json.loads((job / "train_config.json").read_text())
        self.assertEqual(cfg["dataset"], "data/sft_train.jsonl")
        self.assertEqual(cfg["output_dir"], "out")

    def test_pipeline_as_one_job(self):
        self.add_cloud()
        args = ["pipeline", "pp", "--base-model", "Org/Tiny", "--data-dir",
                str(self.ds), "--runtime", "cloud"]
        code, out, err = self.nb(*args)
        self.assertEqual(code, 0, err)
        self.assertIn('"$PY" train.py train_config.json', out)
        code, out, err = self.nb(*args, "--run", "--yes")
        self.assertEqual(code, 0, err + out)
        self.assertIn(": done", self.cli("runtime", "status", "last")[1])
        self.assertEqual(self.cli("runtime", "collect", "last")[0], 0)
        entry = json.loads((self.root / "models" / "registry.json")
                           .read_text())["adapters"]["pp"]
        self.assertTrue(entry["trained"])
        self.assertEqual((entry["base"], entry["peft"]), ("Org/Tiny",
                                                          "lora"))
        self.assertTrue((self.root / "models" / "adapters" / "pp" /
                         "adapter_config.json").is_file())
        job = next(d for d in self.cloud.iterdir()
                   if (d / "pipeline.sh").is_file())
        self.assertIn("set -e", (job / "pipeline.sh").read_text())

    def test_policy_keeps_jobs_here(self):
        self.add_cloud()
        (self.root / "tawjih.policy.json").write_text(
            '{"allow_network": "local"}')
        code, _, err = self.nb("train", "a3", "--base-model", "tiny",
                               "--dataset", str(self.ds / "sft_train.jsonl"),
                               "--runtime", "cloud", "--run", "--yes")
        self.assertNotEqual(code, 0)
        self.assertIn("policy", err)
        shutil.rmtree(self.root / "models" / "jobs", ignore_errors=True)


FAKE_SSH = """#!/usr/bin/env python3
# a stand-in for ssh: records its arguments, then runs the command in a
# folder that plays the remote computer
import json, os, subprocess, sys
log = os.environ["FAKE_LOG"]
open(log, "a").write(json.dumps(["ssh"] + sys.argv[1:]) + "\\n")
sys.exit(subprocess.run(["sh", "-c", sys.argv[2]],
                        cwd=os.environ["FAKE_REMOTE"]).returncode)
"""
FAKE_SCP = """#!/usr/bin/env python3
# a stand-in for scp: host:path is a path in the remote folder
import json, os, shutil, sys
open(os.environ["FAKE_LOG"], "a").write(json.dumps(["scp"] + sys.argv[1:])
                                        + "\\n")
args = [a for a in sys.argv[1:] if a != "-r"]
def where(a):
    return os.path.join(os.environ["FAKE_REMOTE"], a.split(":", 1)[1]) \\
        if ":" in a else a
*srcs, dst = [where(a) for a in args]
for s in srcs:
    if s.endswith("/."):
        shutil.copytree(s[:-2], dst, dirs_exist_ok=True)
    elif os.path.isdir(s):
        shutil.copytree(s, os.path.join(dst, os.path.basename(s)),
                        dirs_exist_ok=True)
    else:
        shutil.copy(s, dst)
"""


@unittest.skipIf(os.name == "nt", "the stand-ins are scripts")
class Ssh(Bridge):
    """The ssh runtime with stand-in ssh and scp: every call is a list of
    arguments (no local shell), scp works with relative names inside the
    job folder, and the remote command keeps its && and $(...)."""

    def setUp(self):
        super().setUp()
        os.environ["TAWJIH_RUNTIMES"] = str(self.root / "rt.json")
        self.addCleanup(os.environ.pop, "TAWJIH_RUNTIMES", None)
        bindir = self.root / "bin"
        bindir.mkdir()
        for name, text in (("ssh", FAKE_SSH), ("scp", FAKE_SCP)):
            (bindir / name).write_text(text)
            (bindir / name).chmod(0o755)
        self.remote = self.root / "remote"
        self.remote.mkdir()
        self.log = self.root / "calls.jsonl"
        old = dict(os.environ)
        self.addCleanup(lambda: (os.environ.clear(), os.environ.update(old)))
        os.environ.update(PATH=str(bindir) + os.pathsep + os.environ["PATH"],
                          FAKE_REMOTE=str(self.remote),
                          FAKE_LOG=str(self.log))
        py = self.root / "fakepy"
        py.write_text(FAKE_PY)
        py.chmod(0o755)
        self.compile()
        self.assertEqual(self.cli("runtime", "add", "gpu", "--kind", "ssh",
                                  "--host", "gpu-box", "--python", str(py),
                                  "--workdir", "jobs")[0], 0)

    def calls(self):
        return [json.loads(x) for x in self.log.read_text().splitlines()]

    def test_send_follow_collect(self):
        code, out, err = self.nb("train", "a5", "--base-model", "tiny",
                                 "--dataset", str(self.ds /
                                                  "sft_train.jsonl"),
                                 "--runtime", "gpu", "--run", "--yes")
        self.assertEqual(code, 0, err + out)
        import time
        for _ in range(100):
            if "done" in self.cli("runtime", "status", "last")[1]:
                break
            time.sleep(0.1)
        self.assertIn(": done", self.cli("runtime", "status", "last")[1])
        self.assertIn("training with", self.cli("runtime", "logs",
                                               "last")[1])
        code, out, err = self.cli("runtime", "collect", "last")
        self.assertEqual(code, 0, err)
        adir = self.root / "models" / "adapters" / "a5"
        self.assertTrue((adir / "adapter_config.json").is_file())
        calls = self.calls()
        start = [c for c in calls if c[0] == "ssh" and "nohup" in c[2]][0]
        self.assertEqual(len(start), 3)          # ssh, host, one command
        self.assertIn("&& nohup sh run.sh", start[2])
        status = [c for c in calls if c[0] == "ssh" and "kill -0" in c[2]]
        self.assertIn("$(cat", status[0][2])
        for c in calls:
            if c[0] == "scp":
                local = [a for a in c[1:] if a != "-r" and ":" not in a]
                self.assertTrue(all(not os.path.isabs(a) for a in local), c)


if __name__ == "__main__":
    unittest.main()
