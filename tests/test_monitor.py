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


class Board(Base):
    """GPUs' free memory, the project's model servers on the board, the
    keys that act on a server, and the board in Arabic."""
    GPUS = [{"index": 0, "name": "A", "util": 90.0, "used": 78000 * 2 ** 20,
             "total": 81920 * 2 ** 20, "free": 3000 * 2 ** 20},
            {"index": 1, "name": "A", "util": 0.0, "used": 1000 * 2 ** 20,
             "total": 81920 * 2 ** 20, "free": 80000 * 2 ** 20}]

    def frame(self, snap, sel=0, **kw):
        sysm = System(self.root)
        s = sysm.sample()
        s["gpu"] = self.GPUS
        lines, _ = render(s, snap, Project(self.root).tree(), 130, 36, sel,
                          None, 2.0, Paint(None), True, "", **kw)
        for line in lines:
            self.assertEqual(display_width(line), 130, line)
        return "\n".join(lines)

    def test_gpu_rows_and_free_memory(self):
        from tawjihlib.monitor import gpu_row
        g = gpu_row("1, NVIDIA H100, 12, 1000, 81920, 80000")
        self.assertEqual((g["index"], g["util"], g["free"] / 2 ** 20),
                         (1, 12.0, 80000))
        text = self.frame(Project(self.root).snapshot())
        self.assertIn("GPU0", text)
        self.assertIn("GPU1", text)
        self.assertIn("free for auto: 1", text)
        text = self.frame(Project(self.root).snapshot(), min_free=90000)
        self.assertIn("free for auto: none", text)

    def stand_in(self):
        import socket
        import subprocess
        from test_engines import STAND_IN
        prog = self.root / "stand_in.py"
        prog.write_text(STAND_IN, encoding="utf-8")
        with socket.socket() as so:
            so.bind(("127.0.0.1", 0))
            port = so.getsockname()[1]
        proc = subprocess.Popen([sys.executable, str(prog), "--port",
                                 str(port), "--model", "m1"],
                                stdout=subprocess.DEVNULL)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        import urllib.request
        for _ in range(100):
            try:
                urllib.request.urlopen("http://127.0.0.1:%d/health" % port,
                                       timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        return proc, port

    def test_servers_on_the_board_and_their_keys(self):
        from tawjihlib.monitor import server_action
        proc, port = self.stand_in()
        self.cli("server", "add", "sg", "--engine", "sglang", "--model",
                 "m1", "--port", str(port))
        self.cli("server", "add", "idle", "--engine", "vllm", "--model",
                 "x", "--port", "1")
        state = self.root / "servers" / "sg"
        state.mkdir(parents=True)
        (state / "pid").write_text(str(proc.pid))
        (state / "log.txt").write_text("loading weights\nready\n")
        proj = Project(self.root)
        snap = proj.snapshot()
        rows = [r for r in snap["rows"] if r["kind"] == "server"]
        self.assertEqual([(r["name"], r["status"], r["health"])
                          for r in rows],
                         [("sg", "running", 200), ("idle", "stopped", None)])
        sel = snap["rows"].index(rows[0])
        text = self.frame(snap, sel)
        for word in ("server sg", "sglang", "Health: 200", "ready",
                     ":%d" % port, "s start  x stop  t tune  f flush"):
            self.assertIn(word, text)
        # its CPU and memory, from two samples of its processes
        from tawjihlib.monitor import watched
        self.assertIn(proc.pid, watched(snap))
        sysm = System(self.root)
        sysm.sample(watched(snap))
        time.sleep(0.3)
        si = sysm.sample(watched(snap))
        si["gpu"] = self.GPUS
        lines, _ = render(si, snap, proj.tree(), 130, 36, sel, None, 2.0,
                          Paint(None), True)
        text = "\n".join(lines)
        if os.path.isdir("/proc"):
            self.assertRegex(text, r"CPU [0-9.]+%   RAM [0-9.]+ MiB")
        data = json.loads(self.cli("top", "--json")[1])
        sg = [x for x in data["servers"] if x["name"] == "sg"][0]
        if os.path.isdir("/proc"):
            self.assertGreater(sg["rss"], 0)
        row = rows[0]
        self.assertEqual(server_action(self.root, row, "f"), "flush sg: 200")
        msg = server_action(self.root, row, "t",
                            lambda q: "mem=0.5 schedule-policy=lpm")
        self.assertIn("mem=0.5", msg)
        cfg = json.loads((self.root / "servers.json").read_text())[
            "servers"]["sg"]
        self.assertEqual((cfg["mem"], cfg["args"]),
                         (0.5, {"schedule-policy": "lpm"}))
        self.assertIn("error", server_action(self.root, row, "t",
                                             lambda q: "mem=7"))
        self.assertIn("no known way", server_action(
            self.root, rows[1], "f"))
        self.assertIn("providers", server_action(self.root, row, "d"))
        self.cli("server", "provider", "sg")
        self.assertIn("--backends sg", server_action(self.root, row, "d"))
        self.assertIn("already running", server_action(self.root, row, "s"))
        self.assertIn("stopped", server_action(self.root, row, "x"))
        proc.wait(timeout=20)
        hist = "".join(f.read_text(encoding="utf-8") for f in
                       (self.root / "history").glob("servers-*.jsonl"))
        for action in ('"api"', '"tune"', '"stop"'):
            self.assertIn(action, hist)
        data = json.loads(self.cli("top", "--json")[1])
        self.assertEqual([x["name"] for x in data["servers"]],
                         ["idle", "sg"])

    def test_arabic(self):
        text = self.frame(Project(self.root).snapshot(), lang="ar")
        for word in ("المعالج", "الذاكرة", "التشغيلات",
                     "حرة للاختيار التلقائي"):
            self.assertIn(word, text)
        code, out, err = self.cli("top", "--once", "--arabic", "--width",
                                  "100", "--height", "28")
        self.assertEqual(code, 0, err)
        self.assertIn("المعالج", out)


if __name__ == "__main__":
    unittest.main()
