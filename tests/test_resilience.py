"""Rate limits, circuit breakers, injected trouble (--chaos) and health."""
import json
import time
import unittest

from test_tawjih import Base
from tawjihlib import providers as P


class Rate(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(P.parse_rate("60/min"), (60, 60.0))
        self.assertEqual(P.parse_rate(" 2 / s "), (2, 1.0))
        self.assertEqual(P.parse_rate("1000/h"), (1000, 3600.0))
        self.assertEqual(P.parse_rate("5/10s"), (5, 10.0))
        for bad in ("", "60", "0/min", "3/x", "a/s", "-1/s"):
            with self.assertRaises(ValueError):
                P.parse_rate(bad)

    def test_waits(self):
        lim = P._Limiter()
        t0 = time.monotonic()
        waited = [lim.wait("x", "2/s") for _ in range(3)]
        self.assertEqual(waited[:2], [0.0, 0.0])
        self.assertGreater(waited[2], 0.5)
        self.assertGreaterEqual(time.monotonic() - t0, 0.85)   # clock steps
        self.assertEqual(lim.wait("y", "2/s"), 0.0)     # its own window


class Chaos(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(P.parse_chaos("fail@1.2"),
                         {"what": "fail", "target": "1.2", "status": 503,
                          "spec": "fail@1.2"})
        self.assertEqual(P.parse_chaos("fail=429@provider:a")["status"], 429)
        self.assertEqual(P.parse_chaos("slow=3s@*")["seconds"], 3.0)
        self.assertEqual(P.parse_chaos("slow=250ms@step:2")["seconds"], 0.25)
        for bad in ("fail", "slow@1", "boom@1", "fail=3s@1", "fail=1.5@1"):
            with self.assertRaises(ValueError):
                P.parse_chaos(bad)


class Providers(Base):
    def setUp(self):
        super().setUp()
        self.addCleanup(P.CHAOS.clear)
        for name in ("main", "spare"):
            code, _, err = self.cli("provider", "add", name, "--kind", "fake",
                                    "--retries", "0", "--backoff-seconds", "0")
            self.assertEqual(code, 0, err)
        self.cli("provider", "default", "main")
        self.cli("provider", "fallback", "main", "spare")

    def call(self):
        return P.call_model(self.dir, "main", "hello")

    def test_rate_and_breaker_saved(self):
        code, out, err = self.cli("provider", "set", "main", "--rate", "60/min",
                                  "--breaker", "2/30")
        self.assertEqual(code, 0, err)
        self.assertIn("rate 60/min", out)
        self.assertIn("breaker 2 failures / 30 s", out)
        cfg = json.loads((self.root / "providers.json").read_text())[
            "providers"]["main"]
        self.assertEqual(cfg["breaker"], {"failures": 2, "cooldown": 30.0})
        code, _, err = self.cli("provider", "set", "main", "--rate", "9/x")
        self.assertNotEqual(code, 0)
        self.assertIn("N/UNIT", err)
        self.cli("provider", "set", "main", "--rate", "none", "--breaker", "0")
        cfg = json.loads((self.root / "providers.json").read_text())[
            "providers"]["main"]
        self.assertNotIn("rate", cfg)
        self.assertNotIn("breaker", cfg)

    def test_chaos_falls_back_and_opens_the_breaker(self):
        self.cli("provider", "set", "main", "--breaker", "2/30")
        P.CHAOS[:] = [P.parse_chaos("fail@provider:main")]
        for k in range(2):
            answer, usage = self.call()
            self.assertEqual(usage["provider"], "spare")
            self.assertEqual(usage["chaos"], ["fail@provider:main"])
        self.assertTrue(P.breaker_open(self.dir, "main") > 25)
        P.CHAOS.clear()
        _, usage = self.call()          # open: main is not even tried
        self.assertEqual(usage["provider"], "spare")
        self.assertEqual(usage["breaker_skipped"], ["main"])
        self.assertNotIn("chaos", usage)
        code, out, _ = self.cli("provider", "breaker")
        self.assertIn("OPEN", out)
        self.cli("provider", "breaker", "main", "--reset")
        _, usage = self.call()
        self.assertEqual(usage["provider"], "main")

    def test_half_open_success_closes(self):
        self.cli("provider", "set", "main", "--breaker", "1/1")
        P.CHAOS[:] = [P.parse_chaos("fail@provider:main")]
        self.call()
        P.CHAOS.clear()
        self.assertTrue(P.breaker_open(self.dir, "main"))
        data = P.load_breakers(self.dir)
        data["main"]["open_until"] = time.time() - 1     # cooldown over
        P._save_breakers(self.dir, data)
        _, usage = self.call()          # one try is let through, and works
        self.assertEqual(usage["provider"], "main")
        self.assertEqual(P.load_breakers(self.dir), {})

    def test_without_a_breaker_nothing_is_counted(self):
        P.CHAOS[:] = [P.parse_chaos("fail@provider:main")]
        self.call()
        self.assertFalse((self.root / "history" / "breakers.json").exists())

    def test_slow(self):
        P.CHAOS[:] = [P.parse_chaos("slow=300ms@*")]
        t0 = time.monotonic()
        _, usage = self.call()
        self.assertGreaterEqual(time.monotonic() - t0, 0.25)   # clock steps
        self.assertEqual(usage["provider"], "main")

    def test_run_is_marked(self):
        self.make_path()
        # on one provider: the step goes on with the fallback
        self.replies = ["21", "explained"]
        code, _, err = self.cli("run", "p", "--allow-code", "--input", "21",
                                "--chaos", "fail@provider:main")
        self.assertEqual(code, 0, err)
        self.assertIn("chaos: an injected failure", err)
        rec = json.loads(sorted((self.root / "history").glob(
            "production-p-*.json"))[-1].read_text(encoding="utf-8"))
        self.assertEqual(rec["chaos"], ["fail@provider:main"])
        step = [s for s in rec["steps"] if s["number"] == "1.1.1"][0]
        self.assertEqual(step["usage"]["chaos"], ["fail@provider:main"])
        self.assertEqual(P.CHAOS, [])       # nothing left for the next run
        # on a step: every provider fails for it, so the run stops there
        code, _, err = self.cli("run", "p", "--allow-code", "--input", "21",
                                "--chaos", "fail@step:1")
        self.assertNotEqual(code, 0)
        self.assertIn("every provider failed", err)
        code, _, err = self.cli("run", "p", "--chaos", "boom")
        self.assertNotEqual(code, 0)
        self.assertIn("chaos is", err)


class Health(Base):
    def test_report(self):
        self.cli("provider", "add", "here", "--kind", "ollama", "--base-url",
                 "http://127.0.0.1:9")                 # nothing listens
        code, out, err = self.cli("health", "--json")
        self.assertEqual(code, 1, err)
        rep = json.loads(out)
        rows = rep["sections"]["providers"]
        self.assertTrue(any(r["state"] == "FAIL" and "does not answer" in
                            r["text"] for r in rows), rows)
        self.assertEqual(rep["failures"], 1)
        libs = {r["name"] for r in rep["sections"]["libraries"]}
        self.assertTrue({"torch", "vllm", "tkinter"} <= libs)

    def test_policy_and_keys(self):
        self.cli("provider", "add", "far", "--kind", "anthropic",
                 "--api-key-env", "TAWJIH_TEST_NO_SUCH_KEY")
        self.cli("policy", "set", "--allow-network", "local")
        code, out, _ = self.cli("health")
        self.assertEqual(code, 1)
        self.assertIn("allows only this computer", out)
        self.cli("policy", "set", "--allow-network", "true", "--yes")
        code, out, _ = self.cli("health", "--local")
        self.assertIn("key in TAWJIH_TEST_NO_SUCH_KEY: NOT set", out)
        self.assertIn("not asked (--local)", out)

    def test_built_in_is_healthy(self):
        code, out, err = self.cli("صحة", "--local")
        self.assertEqual(code, 0, err + out)
        self.assertIn("built in (dry)", out)


if __name__ == "__main__":
    unittest.main()
