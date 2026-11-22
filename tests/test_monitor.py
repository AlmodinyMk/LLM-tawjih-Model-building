"""Tests for tawjih top: the live file a run keeps while it goes, and the
board drawn from it, the history and the system."""
import json
import os
import re
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base  # noqa: E402
from tawjihlib.commands import display_width  # noqa: E402
from tawjihlib.home import Paint  # noqa: E402
from tawjihlib.monitor import (Project, System, alive, render,  # noqa: E402
                               summary)


class Top(Base):
    def setUp(self):
        super().setUp()
        import tawjih
        self.gate = threading.Event()
        self.seen = []

        def slow(prompt, model=""):
            if "WAIT" in prompt:
                live = list((self.root / "history" / "live").glob("*.json"))
                self.seen.append(json.loads(live[0].read_text(
                    encoding="utf-8")) if live else None)
            return "ok " + prompt.split("# Instructions\n")[1].split("\n")[0]
        tawjih.BACKENDS["slow"] = slow
        self.addCleanup(tawjih.BACKENDS.pop, "slow", None)
        self.cli("new", "الجبر")
        self.cli("new", "WAIT here", "--parent", "1")
        self.fill("1", instructions="first")
        self.fill("1.1", instructions="WAIT")
        self.cli("path", "p", "1 TO 1.1")

    def run_p(self):
        code, out, err = self.cli("run", "p", "--input", "x", "--backend",
                                  "slow", "--no-cache")
        self.assertEqual(code, 0, err)

    def test_live_file_while_running_and_gone_after(self):
        self.run_p()
        live = self.seen[0]
        self.assertEqual(live["status"], "running")
        self.assertEqual(live["pid"], os.getpid())
        self.assertNotIn("input", live)
        st = {p["step"]: p["status"] for p in live["steps"]}
        self.assertEqual(st, {1: "done", 2: "running"})
        self.assertFalse(list((self.root / "history" / "live").glob("*")))
        rec = json.loads(next((self.root / "history").glob(
            "production-p-*.json")).read_text(encoding="utf-8"))
        self.assertTrue(all("seconds" in p for p in rec["steps"]))
        self.assertIn("seconds", rec)

    def test_a_dead_run_is_stopped(self):
        rec = {"name": "p", "status": "running", "pid": 999999999,
               "t0": time.time() - 5, "steps": []}
        self.assertFalse(alive(999999999))
        self.assertEqual(summary(rec, live=True)["status"], "stopped")
        self.assertEqual(summary(dict(rec, pid=os.getpid()),
                                 live=True)["status"], "running")

    def test_board(self):
        self.run_p()
        self.run_p()
        proj = Project(self.root)
        snap = proj.snapshot()
        self.assertEqual(len(snap["recent"]), 2)
        nid = self.store().resolve("1.1")
        self.assertEqual(snap["stats"][nid]["runs"], 2)
        sysm = System(self.root)
        sysm.sample()
        s = sysm.sample()
        for w, h in ((80, 24), (132, 40)):
            for fancy in (True, False):
                lines, _ = render(s, snap, proj.tree(), w, h, 0, None, 2.0,
                                  Paint(None), fancy)
                self.assertEqual(len(lines), h)
                for line in lines:
                    self.assertEqual(display_width(line), w, line)
        text = "\n".join(lines)
        for word in ("cpu", "mem", "trees", "runs", "الجبر", "WAIT here",
                     "done"):
            self.assertIn(word, text)
        lines, _ = render(s, snap, proj.tree(), 100, 30, 0, None, 2.0,
                          Paint("true"), True)
        plain = [re.sub(r"\033\[[0-9;]*m", "", x) for x in lines]
        self.assertTrue(all(display_width(x) == 100 for x in plain))

    def test_commands(self):
        self.run_p()
        code, out, err = self.cli("top", "--once", "--width", "90",
                                  "--height", "26")
        self.assertEqual(code, 0, err)
        self.assertIn("runs", out)
        code, out, err = self.cli("مراقبة", "--json")
        self.assertEqual(code, 0, err)
        data = json.loads(out)
        self.assertEqual(data["recent"][0]["name"], "p")
        self.assertIn("mem", data["system"])

    def test_outside_a_project(self):
        empty = self.root / "nothing"
        empty.mkdir()
        proj = Project(empty)
        snap = proj.snapshot()
        lines, _ = render(System(empty).sample(), snap, proj.tree(), 80, 24)
        self.assertEqual(len(lines), 24)


if __name__ == "__main__":
    unittest.main()
