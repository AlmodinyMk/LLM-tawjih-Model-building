"""Tests for the 0.13.1 additions: named providers (providers.json) with retries and
fallback chains, schema contracts, IF/UNTIL key = value, AMONG ... BY key,
and the project's restrict-only policy (tawjih.policy.json)."""
import http.server
import json
import os
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base  # noqa: E402
import tawjih  # noqa: E402
from tawjihlib import providers  # noqa: E402


class FakeServer:
    """An OpenAI-style /chat/completions server on this computer. `plan`
    is a list of (status, answer) served in turn; the last one repeats."""

    def __init__(self, plan, retry_after=None):
        self.plan, self.hits, self.models = list(plan), 0, []
        owner = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                owner.models.append(body.get("model"))
                owner.hits += 1
                status, answer = owner.plan.pop(0) if len(owner.plan) > 1 \
                    else owner.plan[0]
                self.send_response(status)
                if status != 200 and retry_after is not None:
                    self.send_header("Retry-After", str(retry_after))
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if status == 200:
                    out = {"choices": [{"message": {"content": answer}}],
                           "usage": {"prompt_tokens": 1,
                                     "completion_tokens": 1}}
                else:
                    out = {"error": "busy"}
                self.wfile.write(json.dumps(out).encode())

        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d/v1" % self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class Providers(Base):
    def setUp(self):
        super().setUp()
        self.servers = []
        self.sleeps = []
        self._sleep = providers.time.sleep
        providers.time.sleep = self.sleeps.append

    def tearDown(self):
        providers.time.sleep = self._sleep
        for s in self.servers:
            s.close()
        super().tearDown()

    def server(self, plan, **kw):
        s = FakeServer(plan, **kw)
        self.servers.append(s)
        return s

    def add(self, name, url, *extra):
        if "--model" not in extra:
            extra += ("--model", "m0")
        code, out, err = self.cli("provider", "add", name, "--kind", "openai",
                                  "--base-url", url, *extra)
        self.assertEqual(code, 0, err)
        return out

    def test_key_is_never_stored_only_its_variable(self):
        s = self.server([(200, "ok")])
        self.add("local", s.url, "--api-key-env", "MY_KEY", "--model", "m1")
        data = json.loads((self.root / "providers.json").read_text())
        self.assertEqual(data["providers"]["local"]["api_key_env"], "MY_KEY")
        self.assertEqual(data["default"], "local")      # the first added
        code, _, err = self.cli("provider", "add", "x", "--kind", "openai",
                                "--api-key-env", "sk-abc123=")
        self.assertNotEqual(code, 0)
        self.assertIn("NAME of an environment variable", err)
        out = self.cli("provider", "list")[1]
        self.assertIn("key in MY_KEY: NOT set", out)

    def test_retries_a_busy_server_then_answers(self):
        s = self.server([(503, ""), (429, ""), (200, "answer")])
        self.add("local", s.url, "--retries", "3", "--backoff-seconds", "1")
        out, usage = providers.call_model(self.dir, "local", "hi")
        self.assertEqual(out, "answer")
        self.assertEqual(usage["attempts"], 3)
        self.assertEqual(usage["provider"], "local")
        self.assertEqual(self.sleeps, [1.0, 2.0])      # exponential

    def test_retry_after_is_honoured_and_capped(self):
        s = self.server([(429, ""), (200, "ok")], retry_after=5)
        self.add("local", s.url, "--retries", "1")
        providers.call_model(self.dir, "local", "hi")
        self.assertEqual(self.sleeps, [5.0])
        s2 = self.server([(429, ""), (200, "ok")], retry_after=999)
        self.add("far", s2.url, "--retries", "1")
        providers.call_model(self.dir, "far", "hi")
        self.assertEqual(self.sleeps[-1], 60.0)

    def test_a_client_error_is_not_retried(self):
        s = self.server([(400, "")])
        self.add("local", s.url, "--retries", "3")
        with self.assertRaises(providers.ProviderError):
            providers.call_model(self.dir, "local", "hi")
        self.assertEqual(s.hits, 1)

    def test_fallback_chain_and_its_model(self):
        bad = self.server([(500, "")])
        good = self.server([(200, "from b")])
        self.add("a", bad.url, "--retries", "1", "--model", "model-a")
        self.add("b", good.url, "--model", "model-b")
        code, out, err = self.cli("provider", "fallback", "a", "b")
        self.assertEqual(code, 0, err)
        out, usage = providers.call_model(self.dir, "a", "hi", "step-model")
        self.assertEqual(out, "from b")
        self.assertEqual(usage["provider"], "b")
        self.assertEqual(bad.hits, 2)
        self.assertEqual(bad.models, ["step-model", "step-model"])
        self.assertEqual(good.models, ["model-b"])    # its own model first
        with self.assertRaises(providers.ProviderError):
            providers.call_model(self.dir, "a", "hi", fallback=False)
        self.assertIn("cannot repeat", self.cli("provider", "fallback", "a",
                                                "b", "b")[2])
        self.cli("provider", "fallback", "a")
        self.assertNotIn("a", json.loads(
            (self.root / "providers.json").read_text())["fallback"])

    def test_a_path_runs_on_the_default_provider(self):
        s = self.server([(200, "from the server")])
        self.add("local", s.url)
        self.cli("new", "A")
        self.fill("1", instructions="answer")
        self.cli("path", "p", "1")
        code, out, err = self.cli("run", "p", "--input", "x")
        self.assertEqual(code, 0, err)
        self.assertIn("from the server", out)
        code, _, err = self.cli("run", "p", "--input", "x", "--backend",
                                "nowhere")
        self.assertNotEqual(code, 0)
        self.assertIn("unknown provider 'nowhere'", err)

    def test_model_order_with_a_provider(self):
        s = self.server([(200, "ok")])
        self.add("local", s.url, "--model", "prov-model")
        self.cli("new", "A")
        self.fill("1", instructions="answer")
        self.cli("path", "p", "1")
        os.environ["TAWJIH_MODEL"] = "env-model"
        try:
            self.cli("run", "p", "--input", "x")
            self.fill("1", model="file-model")
            self.cli("run", "p", "--input", "x")
            self.cli("run", "p", "--input", "x", "--model", "cli-model")
        finally:
            os.environ.pop("TAWJIH_MODEL", None)
        self.assertEqual(s.models, ["prov-model", "file-model", "cli-model"])

    def test_provider_test_records_the_result(self):
        s = self.server([(200, '{"status": "ok"}')])
        self.add("local", s.url)
        code, out, err = self.cli("provider", "test", "local", "--json")
        self.assertEqual(code, 0, err)
        self.assertTrue(json.loads(out)["ok"])
        s.plan = [(200, "something else")]
        self.assertEqual(self.cli("provider", "test", "local")[0], 1)
        lines = (self.root / "runs" / "provider-tests.jsonl").read_text(
            ).splitlines()
        self.assertEqual([json.loads(x)["ok"] for x in lines], [True, False])
        self.assertEqual(self.cli("provider", "test", "all")[0], 1)

    def test_rm_and_default(self):
        s = self.server([(200, "ok")])
        self.add("a", s.url)
        self.add("b", s.url)
        self.cli("provider", "fallback", "a", "b")
        code, _, err = self.cli("provider", "rm", "b")
        self.assertIn("fallback of a", err)
        self.assertEqual(self.cli("provider", "rm", "b", "--yes")[0], 0)
        data = json.loads((self.root / "providers.json").read_text())
        self.assertEqual(data["fallback"]["a"], [])
        self.assertEqual(self.cli("provider", "default", "dry")[0], 0)
        self.assertEqual(providers.default_backend(self.dir), "dry")
        self.assertNotEqual(self.cli("provider", "policy", "zz",
                                     "--retries", "1")[0], 0)


class Policy(Base):
    def test_policy_only_restricts(self):
        self.cli("new", "A")
        self.fill("1", code="output = 'ran'")
        self.cli("path", "p", "1")
        code, out, err = self.cli("policy", "set", "--allow-code", "false")
        self.assertEqual(code, 0, err)
        code, _, err = self.cli("run", "p", "--input", "x", "--allow-code")
        self.assertNotEqual(code, 0)
        self.assertIn("does not allow --allow-code", err)
        code, _, err = self.cli("sandbox", "1", "--input", "x")
        self.assertIn("does not allow running code", err)
        # loosening needs --yes
        code, _, err = self.cli("policy", "set", "--allow-code", "true")
        self.assertIn("loosens", err)
        self.assertEqual(self.cli("policy", "set", "--allow-code", "true",
                                  "--yes")[0], 0)
        self.assertEqual(self.cli("run", "p", "--input", "x",
                                  "--allow-code")[0], 0)
        self.assertIn("add --yes", self.cli("policy", "reset")[2])
        self.assertEqual(self.cli("policy", "reset", "--yes")[0], 0)
        self.assertFalse((self.root / "tawjih.policy.json").exists())

    def test_network_local_only(self):
        self.cli("policy", "set", "--allow-network", "local")
        self.cli("provider", "add", "here", "--kind", "openai",
                 "--base-url", "http://127.0.0.1:9/v1")
        self.cli("provider", "add", "far", "--kind", "openai",
                 "--base-url", "https://api.example.com/v1")
        with self.assertRaises(tawjih.Die):
            providers.check_network(self.dir, providers.provider_chain(
                self.dir, "far"))
        providers.check_network(self.dir, providers.provider_chain(
            self.dir, "here"))
        # a far fallback behind a local provider is refused too
        self.cli("provider", "fallback", "here", "far")
        with self.assertRaises(tawjih.Die):
            providers.check_network(self.dir, providers.provider_chain(
                self.dir, "here"))
        self.cli("policy", "set", "--allow-network", "false")
        with self.assertRaises(tawjih.Die):
            providers.check_network(self.dir, providers.provider_chain(
                self.dir, "here", False))
        providers.check_network(self.dir, providers.provider_chain(
            self.dir, "dry"))

    def test_sandbox_and_approval(self):
        self.cli("policy", "set", "--sandbox", "true", "--human-approval",
                 "run")
        self.assertTrue(providers.load_policy(self.dir)["sandbox"])
        code, _, err = self.cli("mcp", "--human-approval", "never")
        self.assertNotEqual(code, 0)
        self.assertIn("--human-approval run at least", err)
        (self.root / "tawjih.policy.json").write_text('{"sandbox": "yes"}')
        self.assertIn("cannot be", self.cli("policy", "show")[2])


class Schemas(Base):
    SCHEMA = {"type": "object", "required": ["status", "score"],
              "properties": {"status": {"enum": ["pass", "fail"]},
                             "score": {"type": "number", "minimum": 0,
                                       "maximum": 1}},
              "additionalProperties": False}

    def setUp(self):
        super().setUp()
        self.cli("new", "A")
        f = self.root / "verdict.json"
        f.write_text(json.dumps(self.SCHEMA))
        code, out, err = self.cli("contract", "1", "--produces-schema",
                                  str(f))
        self.assertEqual(code, 0, err)
        self.fill("1", instructions="judge")
        self.cli("path", "p", "1")

    def run_p(self, reply):
        self.replies = [reply]
        return self.cli("run", "p", "--input", "x", "--backend", "fake")

    def test_schema_is_installed_and_named(self):
        self.assertTrue((self.root / "schemas" / "verdict.json").is_file())
        n = self.store().nodes[self.store().resolve("1")]
        self.assertEqual(n.produces, "schema(verdict)")

    def test_valid_and_invalid_answers(self):
        self.assertEqual(self.run_p('{"status": "pass", "score": 0.5}')[0],
                         0)
        code, _, err = self.run_p('{"status": "maybe", "score": 2}')
        self.assertNotEqual(code, 0)
        self.assertIn("status", err)
        code, _, err = self.run_p('{"status": "pass", "score": 1, "x": 1}')
        self.assertNotEqual(code, 0)

    def test_unsupported_keyword_and_missing_schema(self):
        bad = self.root / "bad.json"
        bad.write_text(json.dumps({"oneOf": [{"type": "string"}]}))
        self.cli("new", "B")
        code, _, err = self.cli("contract", "2", "--expects-schema",
                                str(bad))
        self.assertNotEqual(code, 0)
        self.assertIn("oneOf", err)
        code, _, err = self.cli("contract", "2", "--expects", "schema(none)")
        self.assertNotEqual(code, 0)
        os.remove(self.root / "schemas" / "verdict.json")
        out = self.cli("check")[1] + self.cli("check")[2]
        self.assertIn("verdict", out)


class KeyValue(Base):
    def setUp(self):
        super().setUp()
        for t in ("A", "B", "C"):
            self.cli("new", t)
        self.fill("1", instructions="classify")
        self.fill("2", code="output = 'group'")
        self.fill("3", code="output = 'single'")

    def go(self, reply, *words):
        self.cli("path", "p", *words)
        self.replies = [reply] if isinstance(reply, str) else list(reply)
        return self.cli("run", "p", "--input", "x", "--backend", "fake",
                        "--allow-code")

    def test_if_key_equals_value(self):
        words = ("1", "TO", "2", "IF", "is_group", "=", "true", "ELSE", "3")
        code, out, err = self.go('{"is_group": true}', *words)
        self.assertEqual(code, 0, err)
        self.assertIn("group", out)
        code, out, err = self.go('{"is_group": false}', *words)
        self.assertIn("single", out)
        code, out, err = self.go("not json", *words)
        self.assertIn("single", out)      # no JSON: the condition is false

    def test_spellings_of_key_value(self):
        for w in (("kind=a",), ("kind", "=", "a"), ("kind=", "a"),
                  ("kind", "=a")):
            code, _, err = self.cli("path", "q", "1", "TO", "2", "IF", *w)
            self.assertEqual(code, 0, (w, err))
            self.assertIn("kind=a", (self.root / "paths" / "q.path"
                                     ).read_text().replace(" = ", "="))

    def test_until_key_equals_value(self):
        code, out, err = self.go(['{"status": "fail"}', '{"status": "pass"}'],
                                 "1", "RETRY", "2", "UNTIL", "status=pass")
        self.assertEqual(code, 0, err)
        self.assertIn("pass", out)

    def test_among_by_key(self):
        words = ("1", "AMONG", "2", "3", "BY", "route")
        code, out, err = self.go('{"route": "C"}', *words)
        self.assertEqual(code, 0, err)
        self.assertIn("single", out)
        code, _, err = self.go('{"route": "Z"}', *words)
        self.assertNotEqual(code, 0)
        self.assertIn("route", err)
        code, _, err = self.go('{"other": "C"}', *words)
        self.assertNotEqual(code, 0)
        # Arabic: حسب
        code, out, err = self.go('{"route": "B"}', "1", "بين", "2", "3",
                                 "حسب", "route")
        self.assertEqual(code, 0, err)
        self.assertIn("group", out)


if __name__ == "__main__":
    unittest.main()
