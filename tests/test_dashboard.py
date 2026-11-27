"""Tests for tawjih dashboard: the page has no outside address, nothing is
answered without the token, the state is the project's and the system's,
and the keys act on servers as in top (or not at all with --read-only)."""
import json
import re
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base  # noqa: E402
import test_monitor  # noqa: E402
from tawjihlib.dashboard import PAGE, Board, make_server  # noqa: E402


class Dashboard(Base):
    def start(self, read_only=False):
        board = Board(self.root, 8000, read_only)
        httpd = make_server(board, "127.0.0.1", 0, "tok")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.base = "http://127.0.0.1:%d" % httpd.server_address[1]
        return board

    def call(self, path, body=None, token="tok"):
        req = urllib.request.Request(
            self.base + path, method="POST" if body is not None else "GET",
            data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + token} if token else {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read().decode("utf-8"), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8"), dict(e.headers)

    def test_the_page_needs_nothing_from_outside(self):
        self.assertIsNone(re.search(r"(?i)\b(https?:)?//[a-z0-9.-]+\.[a-z]",
                                    PAGE))
        for word in ("<script src", "<link", "@import", "fonts.g"):
            self.assertNotIn(word, PAGE)
        self.start()
        code, text, headers = self.call("/", token=None)
        self.assertEqual(code, 200)
        self.assertIn('dir="rtl"', text)
        self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
        self.assertIn("connect-src 'self'", headers["Content-Security-Policy"])

    def test_the_language_chosen_at_start(self):
        for lang, want_dir, forced in ((None, "rtl", "false"),
                                       ("en", "ltr", "true"),
                                       ("ar", "rtl", "true")):
            board = Board(self.root, 8000, False, lang)
            self.assertIn('lang="%s" dir="%s"' % (lang or "ar", want_dir),
                          board.page)
            self.assertIn("FORCED = %s;" % forced, board.page)
            self.assertNotIn("__LANG__", board.page)
        code, out, err = self.cli("top", "--once", "--lang", "ar", "--width",
                                  "100", "--height", "28")
        self.assertEqual(code, 0, err)
        self.assertIn("المعالج", out)
        self.assertNotIn("المعالج", self.cli("top", "--once", "--lang", "en",
                                             "--width", "100", "--height",
                                             "28")[1])

    def test_nothing_without_the_token(self):
        self.start()
        self.assertEqual(self.call("/api/state", token=None)[0], 401)
        self.assertEqual(self.call("/api/state", token="wrong")[0], 401)
        self.assertEqual(self.call("/api/action", {"server": "x", "key": "s"},
                                   token=None)[0], 401)
        self.assertEqual(self.call("/nothing")[0], 404)

    def test_state(self):
        self.cli("new", "الجبر")
        self.cli("new", "فرع", "--parent", "1")
        self.fill("1", instructions="first")
        self.fill("1.1", instructions="second")
        self.cli("path", "p", "1 TO 1.1")
        self.assertEqual(self.cli("run", "p", "--input", "x", "--backend",
                                  "fake")[0], 0)
        self.cli("server", "add", "big", "--engine", "vllm", "--model", "Q",
                 "--tp", "2", "--port", "1")
        self.start()
        code, text, _ = self.call("/api/state")
        self.assertEqual(code, 200)
        s = json.loads(text)
        self.assertIn("mem", s["system"])
        self.assertEqual([(f["number"], f["title"], f["depth"], f["runs"])
                          for f in s["files"]],
                         [("1", "الجبر", 0, 1), ("1.1", "فرع", 1, 1)])
        kinds = [(r["kind"], r["name"], r["status"]) for r in s["rows"]]
        self.assertIn(("run", "p", "done"), kinds)
        self.assertIn(("server", "big", "stopped"), kinds)
        run = [r for r in s["rows"] if r["kind"] == "run"][0]
        self.assertEqual([p["title"] for p in run["steps"]], ["الجبر", "فرع"])
        self.assertNotIn("record", run)
        srv = [r for r in s["rows"] if r["kind"] == "server"][0]
        self.assertEqual((srv["engine"], srv["settings"]), ("vllm", ["tp=2"]))
        self.assertNotIn("cfg", srv)

    def test_keys_act_on_servers(self):
        helper = test_monitor.Board("stand_in")
        helper.root, helper.addCleanup = self.root, self.addCleanup
        proc, port = helper.stand_in()
        self.cli("server", "add", "sg", "--engine", "sglang", "--model",
                 "m1", "--port", str(port))
        state = self.root / "servers" / "sg"
        state.mkdir(parents=True)
        (state / "pid").write_text(str(proc.pid))
        self.start()
        code, text, _ = self.call("/api/action", {"server": "sg", "key": "f"})
        self.assertEqual((code, json.loads(text)["message"]),
                         (200, "flush sg: 200"))
        code, text, _ = self.call("/api/action", {"server": "sg", "key": "t",
                                                  "text": "mem=0.5"})
        self.assertEqual(code, 200, text)
        cfg = json.loads((self.root / "servers.json").read_text())
        self.assertEqual(cfg["servers"]["sg"]["mem"], 0.5)
        self.assertEqual(self.call("/api/action", {"server": "sg",
                                                   "key": "t",
                                                   "text": "mem=9"})[0], 409)
        self.assertEqual(self.call("/api/action", {"server": "nope",
                                                   "key": "f"})[0], 404)
        self.assertEqual(self.call("/api/action", {"server": "sg",
                                                   "key": "q"})[0], 400)
        log = json.loads(self.call("/api/state")[1])["log"]
        self.assertTrue(any(x["action"] == "board f" for x in log))
        self.assertTrue(any(x["action"] == "tune" for x in log))
        code, text, _ = self.call("/api/action", {"server": "sg", "key": "x"})
        self.assertEqual(code, 200, text)
        proc.wait(timeout=20)

    def test_read_only(self):
        self.cli("server", "add", "big", "--engine", "vllm", "--model", "Q")
        self.start(read_only=True)
        code, text, _ = self.call("/api/action", {"server": "big",
                                                  "key": "s"})
        self.assertEqual(code, 403)
        self.assertTrue(json.loads(self.call("/api/state")[1])["read_only"])


if __name__ == "__main__":
    unittest.main()
