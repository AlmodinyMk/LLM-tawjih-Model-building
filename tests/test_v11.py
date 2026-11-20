"""Tests for 0.11: the new words of the path language (PASS, LINK, ELSE,
EACH, APPROVE, CALL, USING, RETRY, UNTIL, PICK), the reserved PARALLEL and
CACHE, and the Arabic separators."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base  # noqa: E402


class Words(Base):
    def setUp(self):
        super().setUp()
        for t in ("A", "B", "C", "D", "E"):
            self.cli("new", t)
        self.ids = {t: self.store().resolve(str(k)) for k, t in
                    enumerate("ABCDE", 1)}

    def path(self, name, *words):
        code, out, err = self.cli("path", name, *words)
        self.assertEqual(code, 0, err)
        return self.store().load_flow(name)

    def run_path(self, name, *extra):
        code, out, err = self.cli("run", name, "--input", "REQ",
                                  "--allow-code", "--backend", "fake",
                                  *extra)
        return code, out, err

    def edges(self, flow):
        i = {v: k for k, v in self.ids.items()}
        return sorted((i[e["src"]], i[e["dst"]], bool(e.get("order")))
                      for e in flow.edges)

    def test_pass_and_arabic_words_are_synonyms(self):
        f1 = self.path("p1", "1", "PASS", "2", "AND", "3")
        f2 = self.path("p2", "1", "مرر", "2", "و", "3")
        f3 = self.path("p3", "1", "TO", "2", "AND", "3")
        self.assertEqual(self.edges(f1), self.edges(f3))
        self.assertEqual(self.edges(f2), self.edges(f3))
        self.assertIn("TO", (self.root / "paths" / "p1.path").read_text())

    def test_arabic_separators(self):
        f = self.path("p", "1", "إلى", "2", "؛", "2", "إلى", "3")
        self.assertEqual(self.edges(f), [("A", "B", False),
                                         ("B", "C", False)])

    def test_link_orders_without_passing_the_output(self):
        self.fill("1", code="output = 'from A'")
        self.fill("2", code="output = 'B got: ' + input")
        f = self.path("p", "1", "LINK", "2")
        self.assertEqual(self.edges(f), [("A", "B", True)])
        code, out, err = self.run_path("p")
        self.assertEqual(code, 0, err)
        self.assertIn("B got: REQ", out)
        self.assertEqual(f.order(), [self.ids["A"], self.ids["B"]])
        g = self.cli("graph", "--path", "p")[1]
        self.assertIn("LINK", g)
        self.assertIn("LINK: after it", self.cli("produce", "p",
                                                 "--plan")[1])
        self.assertIn("LINK only orders", self.cli("path", "q", "1", "LINK",
                                                   "2", "IF", "x")[2])

    def test_else(self):
        self.fill("1", code="output = input")
        self.fill("2", code="output = 'yes'")
        self.fill("3", code="output = 'no'")
        self.path("p", "1", "TO", "2", "IF", "ok", "ELSE", "3")
        for given, want, other in (('{"ok": true}', "yes", "no"),
                                   ('{"ok": false}', "no", "yes")):
            code, out, err = self.cli("run", "p", "--input", given,
                                      "--allow-code", "--backend", "fake")
            self.assertEqual(code, 0, err)
            self.assertEqual(out.strip(), want)
        self.assertIn("ELSE comes once, after an IF",
                      self.cli("path", "q", "1", "TO", "2", "ELSE", "3")[2])
        f = self.path("r", "1", "إلى", "2", "إذا", "ok", "وإلا", "3", "إلى",
                      "4")
        self.assertIn(("B", "D", False), self.edges(f))
        self.assertIn(("C", "D", False), self.edges(f))

    def test_each_and_pick(self):
        self.fill("1", code='output = \'{"rows": [{"n": 2}, {"n": 5}], '
                  '"x": 1}\'')
        self.fill("2", code="import json\n"
                  "output = str(json.loads(input)['n'] * 10)")
        self.fill("3", code="output = 'got ' + input")
        self.path("p", "1", "PICK", "rows", "TO", "EACH", "2", "TO", "3")
        code, out, err = self.run_path("p")
        self.assertEqual(code, 0, err)
        self.assertIn('got {"items": [20, 50]}', out)
        self.path("q", "1", "PICK", "missing", "TO", "3")
        code, _, err = self.run_path("q")
        self.assertNotEqual(code, 0)
        self.assertIn("PICK", err)

    def test_approve_word_asks_before_the_step(self):
        self.fill("1", code="output = 'x'")
        self.path("p", "APPROVE", "1")
        code, _, err = self.run_path("p", "--approve", "deny")
        self.assertNotEqual(code, 0)
        self.assertIn("not approved before the step ran", err)
        self.assertEqual(self.run_path("p", "--approve", "auto")[0], 0)

    def test_using_chooses_the_model_in_this_path_only(self):
        self.fill("1", instructions="answer", model="header-model")
        self.path("p", "1", "USING", "path-model")
        self.path("q", "1")
        self.run_path("p")
        self.run_path("q")
        self.assertEqual(self.models, ["path-model", "header-model"])
        self.cli("run", "p", "--input", "x", "--backend", "fake", "--model",
                 "cli-model")
        self.assertEqual(self.models[-1], "cli-model")

    def test_retry_until(self):
        self.fill("1", instructions="answer")
        self.path("p", "1", "RETRY", "3", "UNTIL", "done")
        self.replies = ["not yet", "still no", "done!"]
        code, out, err = self.run_path("p")
        self.assertEqual(code, 0, err)
        self.assertEqual(out.strip(), "done!")
        self.assertIn("Your previous answer was not accepted",
                      self.prompts[1])
        self.replies = ["no", "no"]
        self.path("q", "1", "RETRY", "1", "UNTIL", "done")
        code, _, err = self.run_path("q")
        self.assertNotEqual(code, 0)
        self.assertIn("UNTIL", err)
        self.assertIn("UNTIL needs RETRY", self.cli("path", "r", "1",
                                                    "UNTIL", "done")[2])
        self.assertIn("RETRY takes a number", self.cli("path", "r", "1",
                                                       "RETRY", "x")[2])

    def test_call_splices_a_saved_path(self):
        self.fill("1", code="output = input + '>A'")
        self.fill("2", code="output = input + '>B'")
        self.fill("3", code="output = input + '>C'")
        self.fill("4", code="output = input + '>D'")
        self.path("sub", "2", "TO", "3")
        f = self.path("main", "1", "TO", "CALL", "sub", "TO", "4")
        self.assertEqual(self.edges(f), [("A", "B", False),
                                         ("B", "C", False),
                                         ("C", "D", False)])
        self.assertIn("CALL sub", (self.root / "paths" / "main.path")
                      .read_text())
        code, out, err = self.run_path("main")
        self.assertEqual(code, 0, err)
        self.assertEqual(out.strip(), "REQ>A>B>C>D")
        # the called path is read each time: a change reaches the caller
        self.path("sub", "2", "TO", "5")
        self.assertIn(self.ids["E"], self.store().load_flow("main").nodes)
        self.assertIn("in this path and in the called path",
                      self.cli("path", "x", "2", "TO", "CALL", "sub")[2])
        self.path("loop1", "1")
        (self.root / "paths" / "loop1.path").write_text("CALL loop1\n")
        self.assertIn("calls itself", self.cli("run", "loop1", "--input",
                                               "x")[2])
        self.assertIn("no path named", self.cli("path", "y", "1", "TO",
                                                "CALL", "nothing")[2])
        # removing a file used only by the called path is refused too
        err = self.cli("rm", "5", "--yes")[2]
        self.assertIn("main", err)
        self.assertIn("sub", err)

    def test_settings_survive_saving(self):
        self.path("p", "EACH", "1", "USING", "m1", "RETRY", "2", "UNTIL",
                  "NOT", "error", "PICK", "k", "TO", "APPROVE", "2")
        steps = self.store().load_flow("p").steps
        self.assertEqual(steps[self.ids["A"]], {
            "each": "yes", "using": "m1", "retry": 2,
            "until": [True, "error"], "pick": "k"})
        self.assertEqual(steps[self.ids["B"]], {"approve": "before"})
        plan = self.cli("produce", "p", "--plan")[1]
        self.assertIn("model=m1", plan)
        self.assertIn("UNTIL NOT error", plan)

    def test_steps_with_settings_and_no_connecting_word_are_a_sequence(self):
        f = self.path("p", "1", "USING", "m", ",", "2", ",", "3")
        self.assertEqual(self.edges(f), [("A", "B", False),
                                         ("B", "C", False)])

    def test_parallel_and_cache_are_reserved_with_a_reason(self):
        err = self.cli("path", "p", "1", "TO", "PARALLEL", "2", "3")[2]
        self.assertIn("already independent branches", err)
        err = self.cli("path", "p", "1", "CACHE", "TO", "2")[2]
        self.assertIn("--resume", err)
        self.assertIn("word of the path language",
                      self.cli("alias", "1", "using")[2])

    def test_contract_check_ignores_links_and_picked_outputs(self):
        self.cli("contract", "1", "--produces", "json(rows)")
        self.cli("contract", "2", "--expects", "text")
        self.cli("contract", "3", "--expects", "json(z)")
        self.path("p", "1", "PICK", "rows", "TO", "3")
        self.path("q", "1", "LINK", "3")
        errors = [e for e in self.store().check()[0] if "mismatch" in e]
        self.assertEqual(errors, [])
        self.path("r", "1", "TO", "3")
        self.assertTrue(any("mismatch" in e for e in self.store().check()[0]))


if __name__ == "__main__":
    unittest.main()


class Script(Base):
    """The command do: management statements in English or Arabic."""
    def setUp(self):
        super().setUp()
        self.tree()

    def test_translation(self):
        import tawjih
        todo, problems = tawjih.translate_script(
            'أنشئ "تحليل النص" تحت 1.1 / عدّل 1.1.1 جانب آلي بواسطة ollama '
            '"اكتب التعليمات" / انقل 2 في 1 ; انسخ 1.1 تحت 2 / ادمج 2 في 1 '
            '/ احذف 1.1.1 / افحص 1 / نظف\nمسار p 1 إلى 2 , 2 مرر 1.1\n'
            '# a comment\nCREATE x UNDER 1 FROM tpl\n'
            'EDIT 1 SIDE HUMAN USING llama3 shorter')
        self.assertEqual(problems, [])
        argv = [t[1] for t in todo]
        self.assertEqual(argv[0], ["new", "تحليل النص", "--parent", "1.1"])
        self.assertEqual(argv[1], ["edit", "1.1.1", "--side", "instructions",
                                   "--ask", "اكتب التعليمات", "--backend",
                                   "ollama", "--stage"])
        self.assertEqual(argv[2], ["move", "2", "--to", "1"])
        self.assertEqual(argv[3], ["copy", "1.1", "--to", "2"])
        self.assertEqual(argv[4], ["merge", "1", "2"])
        self.assertEqual(argv[5], ["rm", "1.1.1"])
        self.assertEqual(argv[6], ["check", "--branch", "1"])
        self.assertEqual(argv[7], ["gc"])
        self.assertEqual(argv[8], ["path", "p", "1", "إلى", "2", ",", "2",
                                   "مرر", "1.1"])
        self.assertEqual(argv[9], ["scaffold", "tpl", "--title", "x",
                                   "--parent", "1"])
        self.assertEqual(argv[10][-5:], ["--backend", "dry", "--model",
                                         "llama3", "--stage"])
        self.assertEqual([t[2] for t in todo].count(True), 3)

    def test_problems_stop_everything(self):
        code, out, err = self.cli("do", "أنشئ جديد تحت 1 / 1.1 مرر 2 / "
                                  "عدّل 1 جانب آلي", "--run")
        self.assertNotEqual(code, 0)
        self.assertIn("statement 2", err)
        self.assertIn("EDIT needs what to ask", err)
        self.assertEqual(len(self.store().nodes), 4)    # nothing ran

    def test_plan_then_run_with_yes_for_removal(self):
        code, out, _ = self.cli("do", "CREATE new UNDER 2 / DELETE 1.1.1")
        self.assertIn("nothing was run", out)
        self.assertIn("(needs --yes)", out)
        self.assertEqual(len(self.store().nodes), 4)
        code, out, err = self.cli("do", "CREATE new UNDER 2 / DELETE 1.1.1 / "
                                  "CHECK", "--run")
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.store().nodes), 5)        # not removed
        self.cli("do", "DELETE 1.1.1", "--run", "--yes")
        self.assertEqual(len(self.store().nodes), 4)

    def test_a_failing_statement_stops_the_rest(self):
        self.cli("lock", "2")
        code, out, err = self.cli("do", "DELETE 2 / CREATE late", "--run",
                                  "--yes")
        self.assertNotEqual(code, 0)
        self.assertIn("stopped at statement 1", err)
        self.assertNotIn("late", [n.title for n in
                                  self.store().nodes.values()])

    def test_script_file(self):
        f = self.root / "ops.txt"
        f.write_text("# build\nCREATE Notes UNDER 1\nPATH p 1 TO 2\n",
                     encoding="utf-8")
        code, out, err = self.cli("do", "--file", str(f), "--run")
        self.assertEqual(code, 0, err)
        self.assertTrue((self.root / "paths" / "p.path").is_file())
