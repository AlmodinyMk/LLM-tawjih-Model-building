"""Tests for 0.9: the examples side, split and weight, the dataset compiler
with its manifest, and the evaluation family."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base, tawjih  # noqa: E402

EXAMPLES = """Q: Solve 2x + 4 = 10
A: Subtract 4 from both sides: 2x = 6, so x = 3.
E: x = 3
BAD: x = 7

س: ما ناتج 3 + 4؟
ج: 7
"""


class ExamplesSide(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.id = self.store().resolve("1.1.1")

    def test_parse_examples(self):
        exs, probs = tawjih.parse_examples(EXAMPLES + "\nQ: more\nlines\nA: x\n"
                                           "continued\n")
        self.assertEqual(probs, [])
        self.assertEqual(len(exs), 3)
        self.assertEqual((exs[0]["e"], exs[0]["bad"]), ("x = 3", ["x = 7"]))
        self.assertEqual((exs[1]["q"], exs[1]["a"]), ("ما ناتج 3 + 4؟", "7"))
        self.assertEqual((exs[2]["q"], exs[2]["a"]), ("more\nlines",
                                                      "x\ncontinued"))
        _, probs = tawjih.parse_examples("A: no question\nQ: only q\n")
        self.assertEqual(len(probs), 2)

    def test_file_format_keeps_old_files_and_reads_anti_examples(self):
        self.fill("1.1.1", human="h")
        st = self.store()
        old = st.nodes[self.id].dump()           # a file made before 0.15
        self.assertNotIn("=== examples ===", old)
        st.remove_file(self.id)
        f = self.root / "files" / (self.id + ".md")
        f.write_text(old + "=== examples ===\n"
                     "Q: a\nA: b\n=== anti_examples ===\nQ: a\nBAD: c\n",
                     encoding="utf-8")
        n = self.store().nodes[self.id]
        self.assertIn("BAD: c", n.examples)
        self.assertEqual(n.reading, "")

    def test_contract_split_weight_and_check(self):
        self.cli("contract", "1.1.1", "--split", "eval", "--weight", "2")
        n = self.store().nodes[self.id]
        self.assertEqual((n.split, n.weight), ("eval", "2"))
        self.assertIn("split is", self.cli("contract", "1.1.1", "--split",
                                           "test")[2])
        self.assertIn("weight is", self.cli("contract", "1.1.1", "--weight",
                                            "-1")[2])
        self.fill("1.1.1", examples=EXAMPLES)
        self.fill("2", examples="Q: Solve 2x + 4 = 10\nA: x = 3")
        out = self.cli("check")[1]
        self.assertIn("in train and eval files", out)
        self.fill("1.1.1", examples="A: orphan answer")
        self.assertIn("before any question", self.cli("check")[1])

    def test_edit_writes_examples_and_tracks_their_age(self):
        self.fill("1.1.1", human="h1")
        self.replies = [EXAMPLES]
        code, out, err = self.cli("edit", "1.1.1", "--side", "examples",
                                  "--ask", "write examples", "--backend",
                                  "fake", "--yes")
        self.assertEqual(code, 0, err)
        n = self.store().nodes[self.id]
        self.assertTrue(n.examples_sha)
        self.assertFalse(n.examples_stale())
        self.fill("1.1.1", human="h2")
        self.assertIn("examples older than", self.cli("status")[1])
        self.cli("confirm", "1.1.1", "--examples")
        self.assertFalse(self.store().nodes[self.id].examples_stale())
        self.replies = ["A: no question"]
        self.assertIn("not readable", self.cli(
            "edit", "1.1.1", "--side", "examples", "--ask", "x", "--backend",
            "fake", "--yes")[2])


class Dataset(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.fill("1.1.1", examples=EXAMPLES, instructions="Show the steps.")
        self.fill("2", examples="Q: held out\nA: secret", split="eval")
        self.fill("1.1", examples="Q: ignored\nA: x", split="ignore")

    def test_examples_to_jsonl_with_preferences_and_manifest(self):
        out = self.root / "train.jsonl"
        code, o, err = self.cli("dataset", "--source", "examples", "-o",
                                str(out), "--dpo", str(self.root / "dpo.jsonl"),
                                "--with-rules")
        self.assertEqual(code, 0, err)
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8")
                .splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows[0]["messages"][0]["content"].startswith(
            "Show the steps.\n\nSolve"))
        text = out.read_text(encoding="utf-8")
        self.assertNotIn("secret", text)
        self.assertNotIn("ignored", text)
        pairs = [json.loads(l) for l in (self.root / "dpo.jsonl").read_text(
            encoding="utf-8").splitlines()]
        self.assertEqual(pairs[0]["rejected"], "x = 7")
        man = json.loads((self.root / "train.jsonl.manifest.json").read_text())
        st = self.store()
        self.assertEqual(list(man["from"]), [st.resolve("1.1.1")])
        self.assertEqual(len(man["from"][st.resolve("1.1.1")]), 64)
        self.cli("dataset", "--source", "examples", "--split", "eval", "-o",
                 str(out))
        self.assertIn("secret", out.read_text(encoding="utf-8"))

    def test_weight_repeats(self):
        self.cli("contract", "1.1.1", "--weight", "3")
        out = self.root / "w.jsonl"
        self.cli("dataset", "--source", "examples", "-o", str(out),
                 "--repeat-by-weight", "--meta")
        rows = out.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 6)
        self.assertEqual(json.loads(rows[0])["meta"]["weight"], 3.0)


class Evaluation(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.fill("1.1.1", instructions="answer")
        self.cli("produce", "p", "1.1.1", "--plan")
        self.answers = {}

        def fake(prompt, model=""):
            self.prompts.append(prompt)
            if prompt.startswith("You are grading"):
                return "PASS\nsame meaning" if "seven" in prompt else \
                    "FAIL\nwrong"
            for q, a in self.answers.items():
                if q in prompt:
                    return a
            return "I do not know"
        tawjih.BACKENDS["fake"] = fake

    def cases(self, *cases):
        (self.root / "paths" / "p.cases.jsonl").write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in cases),
            encoding="utf-8")

    def ev(self, *extra):
        return self.cli("eval", "p", "--backend", "fake", *extra)

    def test_init_and_extract(self):
        code, out, err = self.cli("eval", "init", "p")
        self.assertEqual(code, 0, err)
        self.assertIn("already exists", self.cli("eval", "init", "p")[2])
        self.fill("2", split="eval", examples=EXAMPLES)
        self.fill("1.1", examples="Q: train only\nA: t")
        code, out, err = self.cli("eval", "extract", "p")
        self.assertIn("2 case(s) added", out)
        self.assertIn("graded by a judge", out)
        cases = tawjih._read_cases(self.root / "paths" / "p.cases.jsonl")
        self.assertEqual([c["input"] for c in cases],
                         ["Solve 2x + 4 = 10", "ما ناتج 3 + 4؟"])
        self.assertEqual(cases[0]["expect"], ["x = 3"])
        self.assertEqual(cases[1]["match"], "judge")
        self.assertIn("no new cases", self.cli("eval", "extract", "p")[1])

    def test_judge_fail_fast_direct_compare_and_report(self):
        self.answers = {"three plus four": "seven", "capital": "Paris"}
        self.cases({"name": "sum", "input": "three plus four",
                    "expect": "7", "match": "judge"},
                   {"name": "capital", "input": "capital of France",
                    "expect": "Paris"},
                   {"name": "hard", "input": "unknown", "expect": "x"})
        code, out, err = self.ev()
        self.assertIn("SKIP sum", out)
        self.assertIn("1 of 2 passed", out)
        code, out, err = self.ev("--judge", "j")
        self.assertIn("PASS sum", out)
        self.assertIn("judge: same meaning", out)
        code, out, err = self.ev("--judge", "j", "--fail-fast")
        self.assertIn("--- expected", out)
        self.assertIn("--- output\nI do not know", out)
        self.answers = {"capital": "Paris"}
        code, out, err = self.ev("--direct")
        self.assertIn("1 of 2 passed (50%), 1 skipped", out)
        rec = tawjih._latest(self.store(), "p", "direct")
        self.assertEqual(rec["mode"], "direct")
        code, out, err = self.cli("eval", "compare", "p")
        self.assertIn("only by path 1", out)
        code, out, err = self.cli("eval", "report", "p", "-o",
                                  str(self.root / "r.md"))
        report = (self.root / "r.md").read_text(encoding="utf-8")
        self.assertIn("# Evaluation of p", report)
        self.assertIn("The model asked directly", report)
        self.assertIn("| hard | **fail** |", report)

    def test_baseline_diff_and_regressions(self):
        self.answers = {"one": "1", "two": "2"}
        self.cases({"name": "one", "input": "one", "expect": "1"},
                   {"name": "two", "input": "two", "expect": "2"})
        self.assertEqual(self.ev()[0], 0)
        self.cli("eval", "baseline", "p", "--save")
        self.assertIn("2/2", self.cli("eval", "baseline", "p")[1])
        self.answers = {"one": "1", "two": "nothing"}
        code, out, err = self.ev()
        self.assertEqual(code, 1)
        self.assertIn("REGRESSION: two", out)
        code, out, err = self.cli("eval", "diff", "p")
        self.assertIn("now failing: two", out)
        self.assertEqual(code, 1)
        self.answers = {"one": "1", "two": "2"}
        self.ev()
        self.assertIn("now passing: two", self.cli("eval", "diff", "p")[1])

    def test_blame_and_promote(self):
        self.cli("new", "second", "--parent", "2")
        self.fill("2.1", instructions="restate")
        self.cli("produce", "p", "1.1.1", "TO", "2.1", "--plan")
        self.answers = {"restate": "nothing", "q1": "the answer is 42"}
        self.cases({"name": "lost", "input": "q1", "expect": "42"},
                   {"name": "never", "input": "q2", "expect": "99"})
        self.ev()
        out = self.cli("eval", "blame", "p")[1]
        self.assertIn("== never", out)
        self.assertIn("no step produced the expected text", out)
        self.assertIn("== lost", out)
        self.assertIn("has the expected text", out)
        self.assertIn("was lost after it", out)
        code, out, err = self.cli("eval", "promote", "p", "--to", "1.1",
                                  "--case", "never")
        self.assertEqual(code, 0, err)
        ex = self.store().nodes[self.store().resolve("1.1")].examples
        self.assertIn("Q: q2\nA: 99", ex)
        self.assertIn("BAD: nothing", ex)
        left = tawjih._read_cases(self.root / "paths" / "p.cases.jsonl")
        self.assertEqual([c["name"] for c in left], ["lost"])
        self.assertTrue((self.root / "paths" / "p.promoted.jsonl").is_file())
        self.fill("2", split="eval")
        self.assertIn("evaluation file", self.cli("eval", "promote", "p",
                                                  "--to", "2")[2])

    def test_synthesize_proposes_for_review(self):
        self.fill("1.1.1", human="Linear equations in one unknown.",
                  examples=EXAMPLES)
        self.fill("2", split="eval", examples="Q: hidden\nA: secret answer")
        reply = ('{"input": "Solve 0x = 5", "expect": ["no solution"]}\n'
                 'not json\n{"input": "Solve x = x", "expect": ["every"]}')

        def fake(prompt, model=""):
            self.prompts.append(prompt)
            return reply
        tawjih.BACKENDS["fake"] = fake
        code, out, err = self.cli("eval", "synthesize", "p", "--backend",
                                  "fake", "--count", "2")
        self.assertEqual(code, 0, err)
        self.assertIn("2 case(s) proposed", out)
        self.assertIn("1 unreadable", out)
        self.assertNotIn("secret answer", self.prompts[-1])
        self.assertFalse((self.root / "paths" / "p.cases.jsonl").exists())
        self.cli("eval", "synthesize", "p", "--accept")
        cases = tawjih._read_cases(self.root / "paths" / "p.cases.jsonl")
        self.assertEqual(len(cases), 2)
        self.assertTrue(cases[0]["synthesized"])

    def test_examples_dataset_from_eval_skips_direct_runs(self):
        self.answers = {"one": "1"}
        self.cases({"name": "one", "input": "one", "expect": "1"})
        self.ev()
        self.ev("--direct")
        out = self.root / "d.jsonl"
        self.cli("dataset", "-o", str(out))
        self.assertEqual(len(out.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
