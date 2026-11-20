"""Tests for the MCP server (tawjih.py mcp): the protocol, the three levels
of permission, argument checking, proposals instead of edits, the log, and
that nothing but protocol messages reaches standard output."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base, tawjih  # noqa: E402

PROGRAM = str(Path(tawjih.__file__).resolve())


class Server(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.fill("1.1.1", human="original", instructions="answer")

    def server(self, **kw):
        return tawjih.Server(self.dir, tawjih.build_parser, **kw)

    def call(self, srv, name, args=None, mid=1):
        return srv.handle({"jsonrpc": "2.0", "id": mid,
                           "method": "tools/call",
                           "params": {"name": name,
                                      "arguments": args or {}}})

    def text(self, resp):
        return resp["result"]["content"][0]["text"]

    def test_initialize_and_levels(self):
        srv = self.server()
        r = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": "2025-03-26"}})
        self.assertEqual(r["result"]["protocolVersion"], "2025-03-26")
        self.assertEqual(r["result"]["serverInfo"]["name"], "tawjih")
        r = srv.handle({"jsonrpc": "2.0", "id": 2, "method": "initialize",
                        "params": {"protocolVersion": "1999-01-01"}})
        self.assertEqual(r["result"]["protocolVersion"], "2025-06-18")
        names = {t["name"] for t in srv.tool_list()}
        self.assertIn("tawjih_show", names)
        self.assertNotIn("tawjih_new", names)
        self.assertNotIn("tawjih_run", names)
        r = self.call(srv, "tawjih_new", {"title": "x"})
        self.assertEqual(r["error"]["code"], -32602)
        w = {t["name"] for t in self.server(allow_write=True).tool_list()}
        self.assertIn("tawjih_propose", w)
        self.assertNotIn("tawjih_run", w)
        every = {t["name"] for t in self.server(
            allow_write=True, allow_run=True).tool_list()}
        self.assertIn("tawjih_run", every)
        # nothing that removes, applies, unlocks or rolls back exists
        for bad in ("rm", "delete", "apply", "unlock", "rollback", "gc",
                    "redact", "shell", "exec"):
            self.assertFalse(any(bad in n for n in every), bad)

    def test_arguments_are_checked(self):
        srv = self.server(allow_write=True)
        for args, why in (({"ref": "--help"}, "not valid"),
                          ({"ref": "../../etc/passwd"}, "not valid"),
                          ({"ref": "1", "path": "/etc"}, "unknown argument"),
                          ({"ref": 5}, "must be a string"),
                          ({}, "missing argument"),
                          ({"ref": "1" * 70}, "too long")):
            r = self.call(srv, "tawjih_show", args)
            self.assertTrue(r["result"]["isError"], args)
            self.assertIn(why, self.text(r))
        r = self.call(srv, "tawjih_propose", {"ref": "1", "side": "secrets",
                                              "text": "x"})
        self.assertIn("must be one of", self.text(r))

    def test_client_text_is_never_an_option(self):
        srv = self.server(allow_write=True)
        r = self.call(srv, "tawjih_new", {"title": "--yes", "parent": "2"})
        self.assertFalse(r["result"]["isError"], self.text(r))
        self.assertIn("--yes", [n.title for n in self.store().nodes.values()])
        r = self.call(srv, "tawjih_search", {"query": "--semantic"})
        self.assertFalse(r["result"]["isError"], self.text(r))
        r = self.call(srv, "tawjih_define_path",
                      {"name": "p", "statements": "1 TO --yes"})
        self.assertTrue(r["result"]["isError"])

    def test_sides_are_only_proposed(self):
        srv = self.server(allow_write=True)
        r = self.call(srv, "tawjih_propose", {"ref": "1.1.1", "side": "human",
                                              "text": "changed by an AI"})
        self.assertFalse(r["result"]["isError"])
        self.assertEqual(self.store().nodes[self.store().resolve(
            "1.1.1")].human, "original")
        self.assertIn("+changed by an AI", self.text(
            self.call(srv, "tawjih_staged")))
        self.cli("stage", "--apply")                # a person applies it
        self.assertEqual(self.store().nodes[self.store().resolve(
            "1.1.1")].human, "changed by an AI")
        self.cli("lock", "2")
        r = self.call(srv, "tawjih_propose", {"ref": "2", "side": "human",
                                              "text": "x"})
        self.assertTrue(r["result"]["isError"])
        self.assertIn("locked", self.text(r))
        r = self.call(srv, "tawjih_contract", {"ref": "2", "model": "m"})
        self.assertTrue(r["result"]["isError"])

    def test_paths_are_replaced_only_when_asked(self):
        srv = self.server(allow_write=True)
        ok = self.call(srv, "tawjih_define_path",
                       {"name": "p", "statements": "1.1.1 TO 2\n2 TO 1.1"})
        self.assertFalse(ok["result"]["isError"], self.text(ok))
        r = self.call(srv, "tawjih_define_path",
                      {"name": "p", "statements": "1 TO 2"})
        self.assertIn("replace=true", self.text(r))
        r = self.call(srv, "tawjih_define_path",
                      {"name": "p", "statements": "1 TO 2", "replace": True})
        self.assertFalse(r["result"]["isError"])

    def test_runs_use_the_fixed_provider_and_deny_approval(self):
        self.cli("path", "p", "1.1.1")
        srv = self.server(allow_run=True, backend="fake", model="m-start")
        r = self.call(srv, "tawjih_run", {"name": "p", "input": "q"})
        self.assertFalse(r["result"]["isError"], self.text(r))
        self.assertEqual(self.models[-1], "m-start")
        self.cli("contract", "1.1.1", "--approve", "before")
        r = self.call(srv, "tawjih_run", {"name": "p", "input": "q"})
        self.assertTrue(r["result"]["isError"])
        self.assertIn("not approved", self.text(r))

    def test_calls_are_logged_without_their_text(self):
        srv = self.server(allow_write=True)
        self.call(srv, "tawjih_propose", {"ref": "1.1.1", "side": "human",
                                          "text": "a private sentence"})
        self.call(srv, "tawjih_new", {"title": "x"})
        log = "".join(f.read_text(encoding="utf-8") for f in
                      (self.root / "history").glob("mcp-*.jsonl"))
        self.assertIn("tawjih_propose", log)
        self.assertNotIn("a private sentence", log)
        self.assertIn("<18 characters>", log)

    def test_protocol_details(self):
        srv = self.server()
        self.assertIsNone(srv.handle({"jsonrpc": "2.0",
                                      "method": "notifications/initialized"}))
        self.assertEqual(srv.handle({"jsonrpc": "2.0", "id": 3,
                                     "method": "nothing"})["error"]["code"],
                         -32601)
        self.assertEqual(srv.handle({"id": 1})["error"]["code"], -32600)
        self.assertEqual(srv.handle({"jsonrpc": "2.0", "id": 4,
                                     "method": "ping"})["result"], {})


class Stdio(Base):
    """The server as a program: every line on standard output is a
    protocol message, even when commands print."""
    def test_only_protocol_on_stdout(self):
        self.tree()
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18"}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "tawjih_tree", "arguments": {}}},
                {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                 "params": {"name": "tawjih_check", "arguments": {}}}]
        text = "\n".join(json.dumps(m) for m in msgs) + "\nnot json\n"
        r = subprocess.run([sys.executable, PROGRAM, "-p", self.dir, "mcp"],
                           input=text, capture_output=True, text=True,
                           encoding="utf-8", timeout=60)
        lines = [l for l in r.stdout.splitlines() if l.strip()]
        replies = [json.loads(l) for l in lines]    # every line is JSON
        self.assertEqual([m.get("id") for m in replies], [1, 2, 3, 4, None])
        self.assertIn("الرياضيات", replies[2]["result"]["content"][0]["text"])
        self.assertEqual(replies[-1]["error"]["code"], -32700)
        self.assertIn("MCP server", r.stderr)


if __name__ == "__main__":
    unittest.main()
