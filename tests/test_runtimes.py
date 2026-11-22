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


if __name__ == "__main__":
    unittest.main()
