"""Durable runs through the MCP server: tawjih_run with an idempotency key,
tawjih_run_status, and a client that disconnects mid-run (the tests
suggested by jiangxingfan1-coder). The model is a local HTTP stand-in held
on an explicit signal, not on a guessed sleep, so the disconnect happens
while the run is certainly active."""
import http.server
import json
import re
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base, tawjih  # noqa: E402

PROGRAM = str(Path(tawjih.__file__).resolve())


class HeldModel:
    """An OpenAI-compatible stand-in that answers only once released."""
    def __init__(self):
        self.hits, self.arrived, self.release = 0, threading.Event(), \
            threading.Event()
        owner = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                self.rfile.read(n)
                owner.hits += 1
                owner.arrived.set()
                owner.release.wait(60)
                body = json.dumps({"model": "held-1",
                                   "choices": [{"message": {
                                       "content": "the answer"}}],
                                   "usage": {"prompt_tokens": 1,
                                             "completion_tokens": 1}})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body.encode())

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d/v1" % self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.release.set()
        self.httpd.shutdown()


class Client:
    """A real stdio session with the server as a separate program."""
    def __init__(self, project, *extra):
        self.p = subprocess.Popen(
            [sys.executable, PROGRAM, "-p", project, "mcp", "--allow-run",
             "--allow-code", "--backend", "held", "--human-approval",
             "never", *extra],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8")
        self.mid = 0
        self.send("initialize", {"protocolVersion": "2025-06-18"})

    def send(self, method, params):
        self.mid += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.mid,
                                       "method": method,
                                       "params": params}) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def tool(self, tool_name, **args):
        r = self.send("tools/call", {"name": tool_name, "arguments": args})
        return r["result"]["content"][0]["text"], r["result"]["isError"]

    def disconnect(self):
        self.p.kill()                     # no goodbye, as a crash would
        self.p.wait(10)
        self.p.stdin.close()
        self.p.stdout.close()

    def close(self):
        self.p.stdin.close()
        self.p.wait(30)
        self.p.stdout.close()


class Durable(Base):
    def setUp(self):
        super().setUp()
        self.model = HeldModel()
        self.addCleanup(self.model.close)
        self.tree()
        self.counter = self.root / "executed.txt"
        self.fill("1.1.1", instructions="answer")
        self.fill("2", code="open(%r, 'a').write('x\\n')\noutput = input"
                  % str(self.counter))
        self.cli("path", "p", "1.1.1", "TO", "2")
        code, _, err = self.cli("provider", "add", "held", "--kind",
                                "openai", "--base-url", self.model.url,
                                "--model", "m")
        self.assertEqual(code, 0, err)

    def runs(self):
        d = self.root / "mcp-runs"
        return sorted(d.glob("m*.json")) if d.is_dir() else []

    def wait_done(self, client, rid):
        for _ in range(300):
            text, err = client.tool("tawjih_run_status", id=rid)
            if "working" not in text.splitlines()[0]:
                return text, err
            time.sleep(0.1)
        self.fail("run %s did not finish" % rid)

    def test_disconnect_then_a_fresh_client_picks_up_the_run(self):
        a = Client(self.dir)
        text, err = a.tool("tawjih_run", name="p", input="q", key="k1",
                           wait=0)
        self.assertFalse(err, text)
        rid = text.split()[1].rstrip(":")
        self.assertRegex(rid, r"^m\d{8}-\d{6}-[0-9a-f]{8}$")
        # the id given out can be read at once
        text, err = a.tool("tawjih_run_status", id=rid)
        self.assertIn("working", text)
        self.assertTrue(self.model.arrived.wait(30))  # the run is active
        a.disconnect()

        b = Client(self.dir)
        try:
            # same key, same input: the same run, and nothing new starts
            text, err = b.tool("tawjih_run", name="p", input="q", key="k1",
                               wait=0)
            self.assertFalse(err, text)
            self.assertIn(rid, text)
            self.assertIn("nothing new started", text)
            # same key, another input: a conflict, and nothing starts
            text, err = b.tool("tawjih_run", name="p", input="other",
                               key="k1", wait=0)
            self.assertTrue(err)
            self.assertIn("idempotency conflict", text)
            self.assertEqual(len(self.runs()), 1)
            self.model.release.set()
            text, err = self.wait_done(b, rid)
            self.assertFalse(err, text)
            self.assertIn("completed", text)
            self.assertIn("the answer", text)
            # once: one run, one model call, the code step ran once
            self.assertEqual(len(self.runs()), 1)
            self.assertEqual(self.model.hits, 1)
            self.assertEqual(self.counter.read_text(), "x\n")
            # asking again after the end returns the result, still once
            text, err = b.tool("tawjih_run", name="p", input="q", key="k1")
            self.assertIn("completed", text)
            self.assertEqual(self.model.hits, 1)
            self.assertEqual(self.counter.read_text(), "x\n")
            run = json.loads(self.runs()[0].read_text(encoding="utf-8"))
            self.assertEqual((run["status"], run["key"]), ("completed",
                                                           "k1"))
        finally:
            self.model.release.set()
            b.close()

    def test_a_short_run_answers_within_wait(self):
        self.model.release.set()
        c = Client(self.dir)
        try:
            text, err = c.tool("tawjih_run", name="p", input="q", key="k2",
                               wait=60)
            self.assertFalse(err, text)
            self.assertIn("completed", text)
            self.assertIn("the answer", text)
            # without a key, a run is as before: no id, no file
            text, err = c.tool("tawjih_run", name="p", input="q")
            self.assertFalse(err, text)
            self.assertEqual(len(self.runs()), 1)
            self.assertEqual(self.counter.read_text(), "x\nx\n")
        finally:
            c.close()

    def test_bad_requests_leave_the_key_free(self):
        c = Client(self.dir)
        try:
            text, err = c.tool("tawjih_run", name="nosuch", input="q",
                               key="k3", wait=0)
            self.assertTrue(err)
            self.assertIn("no path named nosuch", text)
            self.assertEqual(self.runs(), [])
            text, err = c.tool("tawjih_run_status",
                               id="m20260101-000000-00000000")
            self.assertTrue(err)
            r = c.send("tools/call", {"name": "tawjih_run", "arguments": {
                "name": "p", "input": "q", "key": "bad key"}})
            self.assertTrue(r["result"]["isError"])
            self.assertIn("key is not valid", r["result"]["content"][0]
                          ["text"])
            self.assertEqual(self.runs(), [])
        finally:
            c.close()

    def test_approval_then_the_same_key_asks_no_one(self):
        self.model.release.set()
        c = Client(self.dir, "--human-approval", "run")
        try:
            text, err = c.tool("tawjih_run", name="p", input="q", key="k4",
                               wait=0)
            self.assertIn("PENDING", text)
            self.assertEqual(self.runs(), [])     # nothing until approved
            rid = re.search(r"approve (r\d{8}-\d{6}-[0-9a-f]{4})",
                            text).group(1)
            code, out, err = self.cli("approve", rid, "--yes")
            self.assertEqual(code, 0, err + out)
            self.assertEqual(len(self.runs()), 1)
            run = self.runs()[0].stem
            for _ in range(300):
                text, _ = c.tool("tawjih_run_status", id=run)
                if "working" not in text.splitlines()[0]:
                    break
                time.sleep(0.1)
            self.assertIn("completed", text)
            # the key is known now: the same request returns that run,
            # and no person is asked again
            text, err = c.tool("tawjih_run", name="p", input="q", key="k4")
            self.assertNotIn("PENDING", text)
            self.assertIn(run, text)
            self.assertEqual(self.model.hits, 1)
        finally:
            c.close()


class Status(Base):
    """A worker that died is reported as failed, not working forever."""
    def test_dead_worker(self):
        self.tree()
        d = self.root / "mcp-runs"
        d.mkdir()
        rid = "m20260101-000000-0000beef"
        (d / (rid + ".json")).write_text(json.dumps({
            "id": rid, "status": "working", "pid": 999999999}),
            encoding="utf-8")
        srv = tawjih.Server(self.dir, tawjih.build_parser, allow_run=True,
                            human="always")
        self.assertFalse(srv.needs_approval("tawjih_run_status"))
        text, code = srv.call("tawjih_run_status", {"id": rid})
        self.assertEqual(code, 1)
        self.assertIn("failed", text)
        self.assertIn("ended without finishing", text)


if __name__ == "__main__":
    unittest.main()
