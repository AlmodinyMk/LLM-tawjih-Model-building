"""Tests for tawjih dashboard: the board's data (the project's and the
system's), the keys that act on servers (none with --read-only), Arabic
joined and turned for Tk on X11, and the window itself where Tkinter and a
display are there."""
import json
import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base  # noqa: E402
import test_monitor  # noqa: E402
from tawjihlib.dashboard import (  # noqa: E402
    Board, detail_lines, row_cells, shape, visual,
)

try:
    import tkinter
    HAS_TK = True
except ImportError:
    HAS_TK = False


def can_show():
    if not HAS_TK or (os.name != "nt" and not os.environ.get("DISPLAY")):
        return False
    try:
        tkinter.Tk().destroy()
        return True
    except tkinter.TclError:
        return False


class Arabic(unittest.TestCase):
    def test_joined_forms(self):
        self.assertEqual(shape("شرح"), "ﺷﺮﺡ")
        self.assertEqual(shape("لا"), "ﻻ")                # lam-alef
        self.assertEqual(shape("سلا"), "ﺳﻼ")
        self.assertEqual(shape("و"), "ﻭ")
        self.assertEqual(shape("abc"), "abc")

    def test_visual_order(self):
        cases = {"سلام (1.1)": "(1.1) ﻡﻼﺳ",
                 "المزوّد: slow": "slow :" + visual("المزوّد"),
                 "step 3: شرح": "step 3: \ufea1\ufeae\ufeb7",
                 "1.1.1.3": "1.1.1.3", "v1.2-rc": "v1.2-rc"}
        for text, want in cases.items():
            self.assertEqual(visual(text), want, text)
        # a mark stays on its letter
        self.assertIn("ﻭّ", visual("المزوّد"))


class Data(Base):
    def test_snapshot(self):
        self.cli("new", "الجبر")
        self.cli("new", "فرع", "--parent", "1")
        self.fill("1", instructions="first")
        self.fill("1.1", instructions="second")
        self.cli("path", "p", "1 TO 1.1")
        self.assertEqual(self.cli("run", "p", "--input", "x", "--backend",
                                  "fake")[0], 0)
        self.cli("server", "add", "big", "--engine", "vllm", "--model", "Q",
                 "--tp", "2", "--port", "1")
        s = Board(self.root).snapshot()
        self.assertIn("mem", s["system"])
        self.assertEqual([(f["number"], f["title"], f["depth"], f["runs"])
                          for f in s["files"]],
                         [("1", "الجبر", 0, 1), ("1.1", "فرع", 1, 1)])
        run = [r for r in s["rows"] if r["kind"] == "run"][0]
        self.assertEqual([p["title"] for p in run["steps"]], ["الجبر", "فرع"])
        self.assertNotIn("record", run)
        srv = [r for r in s["rows"] if r["kind"] == "server"][0]
        self.assertEqual((srv["engine"], srv["settings"]), ("vllm", ["tp=2"]))
        self.assertNotIn("cfg", srv)
        cells = row_cells(srv, "ar")
        self.assertEqual((cells[1], cells[2], cells[3]),
                         ("big", "■ متوقف", ":1"))
        self.assertEqual(row_cells(run, "en")[3], "2/2")
        pairs, lines = detail_lines(srv, "en")
        self.assertIn(("Settings", "tp=2"), pairs)
        self.assertTrue(lines[0].startswith("$ vllm serve Q"))
        pairs, lines = detail_lines(run, "ar")
        self.assertIn(("الخطوات", "2/2"), pairs)
        self.assertIn("الجبر", lines[0])

    def test_keys_act_on_servers(self):
        helper = test_monitor.Board("stand_in")
        helper.root, helper.addCleanup = self.root, self.addCleanup
        proc, port = helper.stand_in()
        self.cli("server", "add", "sg", "--engine", "sglang", "--model",
                 "m1", "--port", str(port))
        state = self.root / "servers" / "sg"
        state.mkdir(parents=True)
        (state / "pid").write_text(str(proc.pid))
        board = Board(self.root)
        self.assertEqual(board.act("sg", "f"), (200, "flush sg: 200"))
        code, msg = board.act("sg", "t", "mem=0.5")
        self.assertEqual(code, 200, msg)
        cfg = json.loads((self.root / "servers.json").read_text())
        self.assertEqual(cfg["servers"]["sg"]["mem"], 0.5)
        self.assertEqual(board.act("sg", "t", "mem=9")[0], 409)
        self.assertEqual(board.act("nope", "f")[0], 404)
        self.assertEqual(board.act("sg", "q")[0], 400)
        log = board.snapshot()["log"]
        self.assertTrue(any(x["action"] == "board f" for x in log))
        self.assertTrue(any(x["action"] == "tune" for x in log))
        self.assertEqual(board.act("sg", "x")[0], 200)
        proc.wait(timeout=20)

    def test_read_only(self):
        self.cli("server", "add", "big", "--engine", "vllm", "--model", "Q")
        board = Board(self.root, read_only=True)
        self.assertEqual(board.act("big", "s")[0], 403)
        self.assertTrue(board.snapshot()["read_only"])


@unittest.skipUnless(can_show(), "needs Tkinter and a display")
class Window(Base):
    def test_the_window(self):
        from tawjihlib.dashboard import Window as W
        self.cli("new", "الجبر")
        self.fill("1", instructions="first")
        self.cli("path", "p", "1")
        self.cli("run", "p", "--input", "x", "--backend", "fake")
        self.cli("server", "add", "big", "--engine", "vllm", "--model", "Q",
                 "--port", "1")
        w = W(Board(self.root), "ar", 0.5)
        def close():
            try:
                w.root.destroy()
            except tkinter.TclError:
                pass                    # the test closed it already
        self.addCleanup(close)
        for _ in range(60):
            w.root.update()
            if w.state:
                break
            time.sleep(0.05)
        self.assertIsNotNone(w.state)
        w.draw()
        self.assertEqual(len(w.runs.get_children()), 2)
        self.assertEqual(len(w.files.get_children()), 1)
        self.assertEqual(w.root.title(), "توجيه · لوحة التحكم")
        w.move(1)
        self.assertTrue(w.sel_key.startswith("server:big"))
        self.assertIn("vllm", w.det.get("1.0", "end"))
        w.act("s")                                   # read: start fails
        w.switch()
        self.assertEqual(w.lang, "en")
        self.assertEqual(w.root.title(), "tawjih · dashboard")
        self.assertEqual(w.runs.heading("1")["text"], "Path / server")
        w.root.destroy()


if __name__ == "__main__":
    unittest.main()
