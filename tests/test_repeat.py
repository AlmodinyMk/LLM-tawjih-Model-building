"""Tests for eval --repeat N --min-pass K: the live tier of an evaluation,
where a model that does not always answer the same is asked N times and a
case passes on K of N (as suggested by leonardo-tcarvalho), and for the
model version the provider says answered."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base, tawjih  # noqa: E402
from tawjihlib import providers  # noqa: E402


class Repeat(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.fill("1.1.1", instructions="answer yes or no")
        self.cli("path", "p", "1.1.1")
        (self.root / "paths" / "p.cases.jsonl").write_text(
            '{"name": "one", "input": "q", "expect": "yes"}\n',
            encoding="utf-8")
        self.calls = 0

        def varied(prompt, model=""):
            self.calls += 1
            providers._note_usage(3, 1, "m-2026-11-01")
            return self.replies.pop(0) if self.replies else "no"
        tawjih.BACKENDS["varied"] = varied
        self.addCleanup(tawjih.BACKENDS.pop, "varied", None)

    def last(self):
        recs = sorted((self.root / "runs").glob("eval-p-*.json"),
                      key=lambda f: f.stat().st_mtime_ns)
        return json.loads(recs[-1].read_text(encoding="utf-8"))

    def test_k_of_n(self):
        self.replies = ["yes", "no", "yes"]
        code, out, err = self.cli("eval", "p", "--backend", "varied",
                                  "--repeat", "3", "--min-pass", "2")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.calls, 3)            # asked each time
        self.assertIn("2 of 3 runs passed (need 2)", out)
        rec = self.last()
        case = rec["cases"][0]
        self.assertTrue(case["pass"])
        self.assertEqual(case["passes"], 2)
        self.assertEqual([r["pass"] for r in case["runs"]],
                         [True, False, True])
        self.assertEqual((rec["repeat"], rec["min_pass"]), (3, 2))
        self.assertEqual(rec["served"], ["m-2026-11-01"])
        self.assertEqual(case["tokens_in"], 9)

    def test_fails_below_k_and_default_is_all(self):
        self.replies = ["yes", "no", "no"]
        code, out, _ = self.cli("eval", "p", "--backend", "varied",
                                "--repeat", "3", "--min-pass", "2")
        self.assertEqual(code, 1)
        self.assertIn("1 of 3 runs passed (need 2); first failure", out)
        self.replies = ["yes", "yes", "no"]
        code, out, _ = self.cli("eval", "p", "--backend", "varied",
                                "--repeat", "3")
        self.assertEqual(code, 1)                  # K defaults to N
        self.assertIn("(need 3)", out)

    def test_one_run_is_unchanged(self):
        self.replies = ["yes"]
        code, out, err = self.cli("eval", "p", "--backend", "varied")
        self.assertEqual(code, 0, err)
        case = self.last()["cases"][0]
        self.assertNotIn("runs", case)
        self.assertNotIn("passes", case)

    def test_checks(self):
        code, _, err = self.cli("eval", "p", "--backend", "varied",
                                "--repeat", "3", "--min-pass", "4")
        self.assertEqual(code, 1)
        self.assertIn("--min-pass", err)
        code, _, err = self.cli("eval", "p", "--backend", "varied",
                                "--repeat", "2", "--resume")
        self.assertEqual(code, 1)
        self.assertIn("--resume", err)

    def test_served_model_in_run_record(self):
        self.cli("produce", "q", "1.1.1", "--input", "x", "--backend",
                 "varied")
        rec = sorted((self.root / "history").glob("production-q-*.json"))[-1]
        data = json.loads(rec.read_text(encoding="utf-8"))
        self.assertEqual(data["usage"]["served"], ["m-2026-11-01"])
        self.assertEqual(data["steps"][0]["usage"]["served"],
                         "m-2026-11-01")


if __name__ == "__main__":
    unittest.main()
