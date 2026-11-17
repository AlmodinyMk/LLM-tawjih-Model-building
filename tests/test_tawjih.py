"""Tests for tawjih. Run:  python -m unittest discover -s tests -v"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tawjih  # noqa: E402

HUMAN = "المفهوم البشري السري"
READING = "مقطع الكتاب السري"


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="tawjih-test-")
        self.root = Path(self.dir)
        self.prompts = []
        self.replies = []

        def fake(prompt):
            self.prompts.append(prompt)
            return self.replies.pop(0) if self.replies else "ok"
        tawjih.BACKENDS["fake"] = fake
        self.cli("init")

    def tearDown(self):
        tawjih.BACKENDS.pop("fake", None)
        shutil.rmtree(self.dir, ignore_errors=True)

    def cli(self, *args, stdin=""):
        out, err = io.StringIO(), io.StringIO()
        old_argv, old_stdin = sys.argv, sys.stdin
        sys.argv = ["tawjih.py", "-p", self.dir] + list(args)
        sys.stdin = io.StringIO(stdin)
        code = 0
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                tawjih.main()
        except SystemExit as e:
            code = e.code or 0
        finally:
            sys.argv, sys.stdin = old_argv, old_stdin
        return code, out.getvalue(), err.getvalue()

    def store(self):
        return tawjih.Store(self.dir)

    def fill(self, ref, **sides):
        st = self.store()
        n = st.nodes[st.resolve(ref)]
        for k, v in sides.items():
            setattr(n, k, v)
        st.save(n)
        return n.id

    def tree(self):
        """1 general / 1.1 section / 1.1.1 topic, and a second tree 2."""
        self.cli("new", "الرياضيات")
        self.cli("new", "الجبر", "--parent", "1")
        self.cli("new", "المعادلات", "--parent", "1.1", "--level", "موضوع")
        self.cli("new", "البرمجة")


    def make_path(self):
        self.tree()
        self.fill("1", human=HUMAN)
        self.fill("1.1.1", human=HUMAN, reading=READING,
                  instructions="استخرج العدد")
        self.fill("2", human=HUMAN, reading=READING,
                  code="vars['n'] = int(input)\noutput = str(vars['n'] * 2)")
        self.cli("new", "شرح", "--parent", "2")
        self.fill("2.1", instructions="اشرح النتيجة", reading=READING)
        self.cli("path", "p", "1.1.1", "2", "2.1")


class FileFormat(Base):
    def test_roundtrip_all_sides(self):
        n = tawjih.Node(id="abc123", title="عنوان: فرعي", level=4,
                        parent="p00001", order=3, links=[("x00001", "uses")],
                        human="سطر ١\nسطر ٢", instructions="افعل كذا",
                        code="output = 1", reading="## مقطع\nنص")
        m = tawjih.Node.parse(n.dump())
        for k in ("id", "title", "level", "parent", "order", "links",
                  "human", "instructions", "code", "reading"):
            self.assertEqual(getattr(n, k), getattr(m, k), k)

    def test_old_two_sided_file_still_loads(self):
        text = ("---\nid: abc123\ntitle: t\nlevel: 1\nparent: \norder: 1\n"
                "links: \n---\n=== human ===\nh\n=== instructions ===\ni\n"
                "=== code ===\n")
        n = tawjih.Node.parse(text)
        self.assertEqual((n.human, n.instructions, n.code, n.reading),
                         ("h", "i", "", ""))

    def test_id_must_match_file_name(self):
        self.tree()
        st = self.store()
        nid = st.resolve("1")
        os.replace(str(st.file_of(nid)), str(st.fdir / "zzzzzz.md"))
        code, _, err = self.cli("tree")
        self.assertEqual(code, 1)
        self.assertIn("does not match", err)


class Trees(Base):
    def test_numbers_ids_and_levels(self):
        self.tree()
        st = self.store()
        topic = st.resolve("1.1.1")
        self.assertEqual(st.nodes[topic].level, 4)
        self.assertEqual(st.number(topic), "1.1.1")
        self.assertEqual(st.resolve(topic), topic)
        self.assertEqual(st.nodes[st.resolve("2")].title, "البرمجة")
        code, _, err = self.cli("new", "x", "--parent", "1.1", "--level", "1")
        self.assertEqual(code, 1)
        self.assertIn("deeper", err)
        code, _, _ = self.cli("show", "9.9")
        self.assertEqual(code, 1)

    def test_check_reports_broken_structure(self):
        self.tree()
        st = self.store()
        a, b = st.nodes[st.resolve("1")], st.nodes[st.resolve("1.1")]
        c = st.nodes[st.resolve("2")]
        a.parent = b.id                       # cycle
        st.save(a)
        c.parent = "nosuch"                   # missing parent
        c.code = "def broken(:"               # syntax error
        st.save(c)
        code, out, _ = self.cli("check")
        self.assertEqual(code, 1)
        for word in ("cycle", "does not exist", "syntax error"):
            self.assertIn(word, out)

    def test_link_and_path_names(self):
        self.tree()
        self.assertEqual(self.cli("link", "1.1.1", "2", "--type", "uses")[0], 0)
        st = self.store()
        self.assertEqual(st.nodes[st.resolve("1.1.1")].links,
                         [(st.resolve("2"), "uses")])
        code, _, err = self.cli("path", "../evil", "1")
        self.assertEqual(code, 1)
        self.assertFalse((self.root / "evil.path").exists())


class Run(Base):
    def test_only_machine_side_reaches_the_model(self):
        self.make_path()
        self.replies = ["21", "تم"]
        code, out, _ = self.cli("run", "p", "--input", "واحد وعشرون",
                                "--backend", "fake", "--allow-code")
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), "تم")
        self.assertEqual(len(self.prompts), 2)
        for p in self.prompts:
            self.assertNotIn(HUMAN, p)
            self.assertNotIn(READING, p)
        self.assertIn("استخرج العدد", self.prompts[0])
        self.assertIn("42", self.prompts[1])              # code output
        self.assertIn("واحد وعشرون", self.prompts[1])     # original request
        log = next((self.root / "runs").glob("p-*.md")).read_text(encoding="utf-8")
        self.assertNotIn(HUMAN, log)

    def test_code_needs_explicit_permission(self):
        self.make_path()
        code, _, err = self.cli("run", "p", "--input", "x", "--backend", "fake")
        self.assertEqual(code, 1)
        self.assertIn("--allow-code", err)
        self.assertEqual(self.prompts, [])

    def test_failing_code_stops_with_a_log(self):
        self.make_path()
        self.replies = ["ليس عدداً"]
        code, _, err = self.cli("run", "p", "--input", "x", "--backend",
                                "fake", "--allow-code")
        self.assertEqual(code, 1)
        self.assertIn("step 2", err)
        log = next((self.root / "runs").glob("p-*.md")).read_text(encoding="utf-8")
        self.assertIn("FAILED", log)


class Produce(Base):
    def test_trees_lists_roots(self):
        self.make_path()
        code, out, _ = self.cli("trees")
        self.assertEqual(code, 0)
        lines = out.strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("الرياضيات", lines[0])
        self.assertIn("3 file(s), 1 with a machine side", lines[0])
        self.assertIn("2 file(s), 2 with a machine side", lines[1])

    def test_cross_tree_production_runs_saves_and_documents(self):
        self.make_path()
        self.replies = ["21", "تم"]
        code, out, err = self.cli("produce", "prod", "1.1.1", "2", "2.1",
                                  "--input", "واحد وعشرون", "--backend",
                                  "fake", "--allow-code")
        self.assertEqual(code, 0, err)
        self.assertEqual(out.strip(), "تم")
        self.assertIn("across 2 tree(s)", err)
        self.assertIn("handoff", err)
        for p in self.prompts:
            self.assertNotIn(HUMAN, p)
            self.assertNotIn(READING, p)
        res = list((self.root / "runs").glob("prod-*.result.txt"))
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].read_text(encoding="utf-8").strip(), "تم")
        recs = list((self.root / "history").glob("production-prod-*.json"))
        self.assertEqual(len(recs), 1)
        rec = json.loads(recs[0].read_text(encoding="utf-8"))
        self.assertEqual(rec["status"], "done")
        self.assertTrue(rec["cross_tree"])
        self.assertEqual(rec["handoffs"], 1)
        self.assertEqual([s["status"] for s in rec["steps"]], ["done"] * 3)
        st = self.store()
        self.assertEqual(rec["steps"][1]["machine_sha"],
                         tawjih.machine_side(st.nodes[st.resolve("2")])[2])
        # the selection is saved and can be produced again by name
        self.replies = ["5", "ثانية"]
        code, out, _ = self.cli("produce", "prod", "--input", "خمسة",
                                "--backend", "fake", "--allow-code")
        self.assertEqual((code, out.strip()), (0, "ثانية"))
        self.assertEqual(len(list((self.root / "history").glob("production-prod-*.json"))), 2)

    def test_production_is_strict_before_it_starts(self):
        self.make_path()
        code, _, err = self.cli("produce", "bad", "1", "2", "--backend", "fake",
                                "--allow-code")
        self.assertEqual(code, 1)
        self.assertIn("empty machine side", err)
        code, _, err = self.cli("produce", "p", "--backend", "fake")
        self.assertEqual(code, 1)
        self.assertIn("--allow-code", err)
        self.assertEqual(self.prompts, [])
        self.assertFalse((self.root / "runs").exists()
                         and list((self.root / "runs").iterdir()))
        code, out, _ = self.cli("produce", "p", "--plan")
        self.assertEqual(code, 0)
        self.assertIn("instructions", out)
        self.assertEqual(self.prompts, [])

    def test_failed_production_is_documented(self):
        self.make_path()
        self.replies = ["ليس عدداً"]
        code, _, err = self.cli("produce", "p", "--backend", "fake",
                                "--allow-code", "--input", "x")
        self.assertEqual(code, 1)
        rec = json.loads(next((self.root / "history").glob(
            "production-p-*.json")).read_text(encoding="utf-8"))
        self.assertEqual((rec["status"], rec["failed_step"]), ("failed", 2))
        self.assertFalse(list((self.root / "runs").glob("*.result.txt")))


class Branching(Base):
    """Paths written with TO, AND, IF, AMONG, THROUGH."""

    def setUp(self):
        super().setUp()
        # tree 1: 1, 1.1 (a), 1.2 (b), 1.3 (c)   tree 2: 2, 2.1 (d), 2.2 (e)
        self.cli("new", "شجرة أولى")
        self.cli("new", "شجرة ثانية")
        for parent, name in (("1", "a"), ("1", "b"), ("1", "c"),
                             ("2", "d"), ("2", "e")):
            self.cli("new", name, "--parent", parent)
        for ref, name in (("1.1", "a"), ("1.2", "b"), ("1.3", "c"),
                          ("2.1", "d"), ("2.2", "e")):
            self.fill(ref, code="output = '%s<' + input + '>'" % name)

    def produce(self, *words, inp="x"):
        code, out, err = self.cli("produce", "f", *words, "--input", inp,
                                  "--backend", "fake", "--allow-code")
        rec = None
        recs = sorted((self.root / "history").glob("production-f-*.json"),
                      key=lambda f: f.stat().st_mtime_ns)
        if recs:
            rec = json.loads(recs[-1].read_text(encoding="utf-8"))
        return code, out.strip(), err, rec

    def status(self, rec):
        return {s["title"]: s["status"] for s in rec["steps"]}

    def test_output_branches_to_several_files_and_joins_again(self):
        code, out, err, rec = self.produce(
            "1.1", "TO", "1.2", "AND", "2.1", "TO", "2.2")
        self.assertEqual(code, 0, err)
        self.assertTrue(out.startswith("e<"))
        self.assertIn("b<a<x>>", out)          # both branches got a's output
        self.assertIn("d<a<x>>", out)          # and both reached e
        self.assertEqual(self.status(rec), dict.fromkeys("abde", "done"))
        self.assertEqual(rec["handoffs"], 2)   # a->d and b->e cross trees
        self.assertEqual(len(rec["connections"]), 4)

    def test_if_routes_by_the_content_of_the_output(self):
        words = ("1.1", "TO", "1.2", "IF", "خطأ", ",",
                 "1.1", "TO", "1.3", "IF", "NOT", "خطأ")
        code, out, _, rec = self.produce(*words, inp="فيه خطأ")
        self.assertEqual((code, out), (0, "b<a<فيه خطأ>>"))
        self.assertEqual(self.status(rec)["c"], "skipped")
        code, out, _, rec = self.produce(inp="سليم")     # saved, reused
        self.assertEqual((code, out), (0, "c<a<سليم>>"))
        self.assertEqual(self.status(rec)["b"], "skipped")

    def test_among_runs_only_the_file_the_output_names(self):
        self.fill("1.1", code="output = 'اختر ' + input")
        code, out, err, rec = self.produce("1.1", "AMONG", "1.2", "1.3", "2.1",
                                           "TO", "2.2", inp="2.1")
        self.assertEqual(code, 0, err)
        self.assertEqual(out, "e<d<اختر 2.1>>")
        st = self.status(rec)
        self.assertEqual((st["b"], st["c"], st["d"]),
                         ("skipped", "skipped", "done"))
        code, _, err, rec = self.produce(inp="لا شيء")
        self.assertEqual(code, 1)
        self.assertIn("names none of", err)
        self.assertEqual(rec["status"], "failed")

    def test_among_matches_whole_words_only(self):
        self.fill("1.1", code="output = 'ابدأ من database ثم ' + input")
        code, out, err, _ = self.produce("1.1", "AMONG", "1.2", "1.3",
                                         inp="c")
        self.assertEqual(code, 0, err)
        self.assertTrue(out.startswith("c<"))      # not 'a' inside database

    def test_titles_are_one_line_and_if_word_has_no_hash(self):
        self.cli("new", "سطر\nparent: zzz")
        st = self.store()
        self.assertEqual(st.nodes[st.resolve("3")].title, "سطر parent: zzz")
        code, _, err, _ = self.produce("1.1", "TO", "1.2", "IF", "#وسم")
        self.assertEqual(code, 1)
        self.assertIn("#", err)

    def test_through_passes_by_intermediate_files(self):
        code, out, _, _ = self.produce("1.1", "TO", "2.2", "THROUGH",
                                       "1.2", "2.1")
        self.assertEqual((code, out), (0, "e<d<b<a<x>>>>"))

    def test_arabic_words_and_slash_separator(self):
        code, out, err, _ = self.produce("1.1", "إلى", "1.2", "و", "1.3", "/",
                                         "1.2", "و", "1.3", "إلى", "2.1")
        self.assertEqual(code, 0, err)
        self.assertTrue(out.startswith("d<"))
        self.assertIn("c<a<x>>", out)

    def test_several_final_files_are_all_in_the_result(self):
        code, out, _, _ = self.produce("1.1", "TO", "1.2", "AND", "1.3")
        self.assertEqual(code, 0)
        self.assertIn("b<a<x>>", out)
        self.assertIn("c<a<x>>", out)

    def test_bad_paths_are_refused_before_anything_runs(self):
        for words, msg in ((("1.1", "TO", "1.2", "TO", "1.1"), "loop"),
                           (("1.1", "TO"), "missing"),
                           (("1.1", "1.2", "TO", "1.3"), "expected TO"),
                           (("1.1", "AND", "1.2", "AMONG", "1.3", "2.1"),
                            "exactly one"),
                           (("1.1", "TO", "9.9"), "no file numbered")):
            code, _, err, rec = self.produce(*words)
            self.assertEqual(code, 1, words)
            self.assertIn(msg, err)
            self.assertIsNone(rec)
        self.assertFalse((self.root / "paths" / "f.path").exists())

    def test_saved_path_file_is_readable_and_checked(self):
        self.produce("1.1", "TO", "1.2", "IF", "كلمتان معاً")
        text = (self.root / "paths" / "f.path").read_text(encoding="utf-8")
        st = self.store()
        self.assertIn("%s TO %s IF" % (st.resolve("1.1"), st.resolve("1.2")),
                      text)
        self.assertEqual(st.load_flow("f").edges[0]["cond"],
                         [False, "كلمتان معاً"])
        self.assertEqual(self.cli("check")[0], 0)
        (self.root / "paths" / "g.path").write_text(
            "%s TO nosuch\n" % st.resolve("1.1"), encoding="utf-8")
        code, out, _ = self.cli("check")
        self.assertEqual(code, 1)
        self.assertIn("path g", out)


class CacheAndResume(Base):
    def setUp(self):
        super().setUp()
        self._model = os.environ.pop("TAWJIH_MODEL", None)
        self.make_path()

    def tearDown(self):
        os.environ.pop("TAWJIH_MODEL", None)
        if self._model is not None:
            os.environ["TAWJIH_MODEL"] = self._model
        super().tearDown()

    def go(self, *extra, replies=()):
        self.prompts.clear()
        self.replies = list(replies)
        code, out, err = self.cli("produce", "p", "--input", "واحد وعشرون",
                                  "--backend", "fake", "--allow-code", *extra)
        recs = sorted((self.root / "history").glob("production-p-*.json"),
                      key=lambda f: f.stat().st_mtime_ns)
        rec = json.loads(recs[-1].read_text(encoding="utf-8")) if recs else {}
        return code, out.strip(), err, rec

    def cached(self):
        return sorted((self.root / "cache").glob("*.json")) \
            if (self.root / "cache").is_dir() else []

    def test_resume_reuses_unchanged_steps_only(self):
        code, out, _, rec = self.go(replies=["21", "تم"])
        self.assertEqual((code, out, len(self.prompts)), (0, "تم", 2))
        self.assertEqual((len(self.cached()), rec["from_cache"]), (2, 0))
        # same path, no --resume: the model is asked again
        code, out, _, rec = self.go(replies=["21", "ثانية"])
        self.assertEqual((out, len(self.prompts), rec["from_cache"]),
                         ("ثانية", 2, 0))
        # --resume: nothing changed, so nothing is asked
        code, out, err, rec = self.go("--resume")
        self.assertEqual((code, out, len(self.prompts)), (0, "ثانية", 0))
        self.assertEqual(rec["from_cache"], 2)
        self.assertIn("taken from the cache", err)
        # change the last step's instructions: only that step is asked
        self.fill("2.1", instructions="اشرح النتيجة بإيجاز")
        code, out, _, rec = self.go("--resume", replies=["موجز"])
        self.assertEqual((code, out, len(self.prompts)), (0, "موجز", 1))
        self.assertEqual([s.get("cached", False) for s in rec["steps"]],
                         [True, False, False])
        self.assertIn("42", self.prompts[0])        # code ran again

    def test_resume_after_a_failed_model_call(self):
        calls = []

        def flaky(prompt):
            calls.append(prompt)
            if len(calls) == 2:
                raise RuntimeError("HTTP 503")
            return "21" if len(calls) == 1 else "تم"
        normal = tawjih.BACKENDS["fake"]
        tawjih.BACKENDS["fake"] = flaky
        try:
            code, _, err, rec = self.go()
            self.assertEqual((code, rec["failed_step"], len(calls)), (1, 3, 2))
            code, out, _, rec = self.go("--resume")
        finally:
            tawjih.BACKENDS["fake"] = normal
        self.assertEqual((code, out), (0, "تم"))
        self.assertEqual(len(calls), 3)             # step 1 was not asked again
        self.assertEqual(rec["from_cache"], 1)

    def test_redo_replaces_the_answer_that_broke_a_later_step(self):
        code, _, err, rec = self.go(replies=["ليس عدداً"])
        self.assertEqual((code, rec["failed_step"]), (1, 2))
        self.assertIn("--resume --redo 1.1.1", err)
        code, _, err, rec = self.go("--resume")         # same bad answer again
        self.assertEqual((code, len(self.prompts), rec["from_cache"]),
                         (1, 0, 1))
        code, out, _, rec = self.go("--resume", "--redo", "1.1.1",
                                    replies=["21", "تم"])
        self.assertEqual((code, out, len(self.prompts)), (0, "تم", 2))
        code, out, _, rec = self.go("--resume")         # the good one is kept
        self.assertEqual((code, out, len(self.prompts)), (0, "تم", 0))
        for extra, msg in ((("--redo", "1.1.1"), "goes with --resume"),
                           (("--resume", "--redo", "1"), "not in this path")):
            code, _, err, _ = self.go(*extra)
            self.assertEqual(code, 1)
            self.assertIn(msg, err)

    def test_code_output_set_to_empty_stops_but_no_output_passes_on(self):
        self.fill("2", code="vars['seen'] = input")      # sets no output
        code, out, _, _ = self.go(replies=["21", "تم"])
        self.assertEqual(code, 0)
        self.assertIn("# Input\n21", self.prompts[1])    # input passed on
        self.fill("2", code="output = ''")
        code, _, err, rec = self.go(replies=["21"])
        self.assertEqual((code, rec["failed_step"]), (1, 2))
        self.assertIn("produced no output", err)

    def test_cache_command(self):
        self.go(replies=["21", "تم"])
        code, out, _ = self.cli("cache")
        self.assertIn("2 kept answer(s)", out)
        code, out, _ = self.cli("cache", "--clear")
        self.assertIn("removed 2", out)
        self.assertEqual(self.cached(), [])
        code, out, _, _ = self.go("--resume", replies=["21", "تم"])
        self.assertEqual(len(self.prompts), 2)

    def test_no_cache_and_dry_keep_nothing(self):
        code, _, _, _ = self.go("--no-cache", replies=["21", "تم"])
        self.assertEqual((code, self.cached()), (0, []))
        code, _, err, _ = self.go("--no-cache", "--resume")
        self.assertEqual(code, 1)
        self.assertIn("use one of them", err)
        self.cli("produce", "p", "--input", "7", "--backend", "dry",
                 "--allow-code")
        self.assertEqual(self.cached(), [])

    def test_empty_answer_stops_the_run(self):
        code, _, err, rec = self.go(replies=["   "])
        self.assertEqual((code, rec["status"], rec["failed_step"]),
                         (1, "failed", 1))
        self.assertIn("empty answer", err)
        self.assertEqual((len(self.prompts), self.cached()), (1, []))
        self.assertFalse(list((self.root / "runs").glob("*.result.txt")))

    def test_model_option_sets_the_model_and_separates_the_cache(self):
        code, _, _, rec = self.go("--model", "m1", replies=["21", "تم"])
        self.assertEqual((code, rec["model"]), (0, "m1"))
        self.assertEqual(os.environ["TAWJIH_MODEL"], "m1")
        code, out, _, rec = self.go("--resume", "--model", "m2",
                                    replies=["21", "من m2"])
        self.assertEqual((out, len(self.prompts), rec["from_cache"]),
                         ("من m2", 2, 0))
        code, out, _, _ = self.go("--resume", "--model", "m1")
        self.assertEqual((out, len(self.prompts)), ("تم", 0))

    def test_plan_shows_sizes(self):
        code, out, _ = self.cli("produce", "p", "--plan")
        self.assertEqual(code, 0)
        self.assertIn("size: instructions %d characters"
                      % len("استخرج العدد"), out)
        self.assertIn("size: code ", out)
        self.assertIn("sent to the model: %d characters of instructions in "
                      "2 step(s)" % (len("استخرج العدد") + len("اشرح النتيجة")),
                      out)


class Edit(Base):
    def test_edit_shows_all_sides_and_saves_with_backup(self):
        self.tree()
        nid = self.fill("1.1.1", human=HUMAN, reading=READING, code="output = 1")
        self.replies = ["```python\noutput = 2\n```"]
        code, out, _ = self.cli("edit", "1.1.1", "--side", "code", "--ask",
                                "غيّر", "--backend", "fake", "--yes")
        self.assertEqual(code, 0)
        self.assertIn(HUMAN, self.prompts[0])
        self.assertIn(READING, self.prompts[0])
        self.assertEqual(self.store().nodes[nid].code, "output = 2")
        self.assertEqual(len(list((self.root / "history").glob(nid + "-*.md"))), 1)

    def test_edit_needs_confirmation_and_rejects_bad_code(self):
        self.tree()
        nid = self.fill("1.1.1", code="output = 1")
        self.replies = ["output = 2"]
        self.cli("edit", "1.1.1", "--side", "code", "--ask", "x",
                 "--backend", "fake", stdin="n\n")
        self.assertEqual(self.store().nodes[nid].code, "output = 1")
        self.replies = ["def broken(:"]
        code, _, err = self.cli("edit", "1.1.1", "--side", "code", "--ask",
                                "x", "--backend", "fake", "--yes")
        self.assertEqual(code, 1)
        self.assertIn("syntax error", err)
        self.assertEqual(self.store().nodes[nid].code, "output = 1")


class Library(Base):
    def book(self, text):
        f = self.root / "book.txt"
        f.write_text(text, encoding="utf-8")
        return str(f)

    def test_deposit_filter_accept_fill_the_reading_side(self):
        self.tree()
        target = self.store().resolve("1.1.1")
        path = self.book("# ناشر\nحقوق\n# قانون\nنص القانون\n# غريب\nشيء\n")
        code, out, _ = self.cli("deposit", path)
        self.assertEqual(code, 0)
        bid = out.split()[1]
        self.assertEqual(self.cli("deposit", path)[0], 1)      # duplicate
        self.replies = [
            json.dumps({"keep": False, "reason": "publisher"}),
            json.dumps({"keep": True, "target": target, "title": "القانون"}),
            json.dumps({"keep": True, "target": "nosuch", "title": "غريب"}),
        ]
        code, out, _ = self.cli("filter", bid, "--backend", "fake")
        self.assertEqual(code, 0)
        self.assertIn("1 kept, 1 dropped, 1 need review", out)
        self.assertEqual(self.store().nodes[target].reading, "")   # untouched
        self.assertEqual(self.cli("accept", bid)[0], 0)
        st = self.store()
        self.assertEqual(len(st.nodes), 4)                         # no new files
        reading = st.nodes[target].reading
        self.assertIn("نص القانون", reading)
        self.assertIn("[source: book.txt, book %s, passage 2]" % bid, reading)
        self.assertNotIn("شيء", reading)
        self.assertEqual(self.cli("accept", bid)[0], 1)            # only once
        code, out, _ = self.cli("read", "1", "--all")
        self.assertIn("نص القانون", out)


class Editors(Base):
    def test_open_saves_each_side_edited_in_an_editor(self):
        self.tree()
        nid = self.fill("1.1.1", human="قديم", instructions="تعليمات",
                        reading=READING)
        script = self.root / "fake_editor.py"
        script.write_text(
            "import sys\n"
            "for f in sys.argv[1:]:\n"
            "    t = open(f, encoding='utf-8-sig').read()\n"
            "    if f.endswith('human.txt'):\n"
            "        t = 'جديد\\n'\n"
            "    if f.endswith('machine.txt'):\n"
            "        t = t + 'output = 7\\n'\n"
            "    open(f, 'w', encoding='utf-8').write(t)\n"
            "open(sys.argv[1] + '.seen', 'w').write(str(len(sys.argv) - 1))\n",
            encoding="utf-8")
        editor = '"%s" "%s"' % (sys.executable, script)
        code, out, err = self.cli("open", "1.1.1", "--editor", editor)
        self.assertEqual(code, 0, err)
        self.assertIn("human, machine, reading", out)
        n = self.store().nodes[nid]
        self.assertEqual(n.human, "جديد")
        self.assertEqual(n.instructions, "تعليمات")
        self.assertEqual(n.code, "output = 7")
        self.assertEqual(n.reading, READING)
        self.assertFalse((self.root / "work").exists())
        self.assertEqual(len(list((self.root / "history").glob("*.md"))), 1)

    def test_vscode_names_and_flags(self):
        for spec in ("vscode", "code", "code --wait"):
            cmd, name = tawjih.editor_command(spec)
            self.assertEqual(name, "code")
            self.assertEqual(cmd.count("--wait"), 1)
            self.assertIn("--new-window", cmd)
        cmd, name = tawjih.editor_command("vim")
        self.assertEqual((name, len(cmd)), ("vim", 1))

    def test_editor_names_from_full_paths(self):
        for spec in (r'"C:\\Program Files\\VS Code\\bin\\code.cmd"',
                     "/usr/share/code/bin/code", "Code.EXE"):
            cmd, name = tawjih.editor_command(spec)
            self.assertEqual(name, "code", spec)
            self.assertIn("--wait", cmd)

    @unittest.skipUnless(os.name == "nt", "Windows starts editors via .cmd")
    def test_windows_editor_given_as_a_cmd_file_on_the_path(self):
        self.tree()
        nid = self.fill("1.1.1", human="old")
        bindir = self.root / "bin"
        bindir.mkdir()
        (bindir / "fakeed.py").write_text(
            "import sys\n"
            "for f in sys.argv[1:]:\n"
            "    if f.endswith('human.txt'):\n"
            "        open(f, 'w', encoding='utf-8').write('new\\n')\n",
            encoding="utf-8")
        (bindir / "fakeed.cmd").write_text(
            '@"%s" "%s" %%*\r\n' % (sys.executable, bindir / "fakeed.py"))
        old = os.environ["PATH"]
        os.environ["PATH"] = str(bindir) + os.pathsep + old
        try:
            code, out, err = self.cli("open", "1.1.1", "--editor", "fakeed")
        finally:
            os.environ["PATH"] = old
        self.assertEqual(code, 0, err)
        self.assertEqual(self.store().nodes[nid].human, "new")

    def test_unfinished_edit_is_not_overwritten(self):
        self.tree()
        nid = self.fill("1.1.1", human="saved")
        left = self.root / "work" / nid
        left.mkdir(parents=True)
        (left / "human.txt").write_text("unsaved work", encoding="utf-8")
        editor = '"%s" -c pass' % sys.executable
        code, _, err = self.cli("open", "1.1.1", "--editor", editor)
        self.assertEqual(code, 1)
        self.assertIn("--discard", err)
        self.assertEqual((left / "human.txt").read_text(encoding="utf-8"),
                         "unsaved work")
        code, out, err = self.cli("open", "1.1.1", "--editor", editor,
                                  "--discard")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.store().nodes[nid].human, "saved")

    def test_split_opens_the_machine_side_as_two_typed_files(self):
        self.tree()
        nid = self.fill("1.1.1", human="مفهوم", instructions="تعليمات",
                        code="output = 1")
        script = self.root / "fake_code.py"
        script.write_text(
            "import sys, os\n"
            "names = sorted(os.path.basename(f) for f in sys.argv[1:])\n"
            "open(os.path.join(os.path.dirname(sys.argv[1]), '..', '..',"
            " 'seen.txt'), 'w').write(' '.join(names))\n"
            "for f in sys.argv[1:]:\n"
            "    t = open(f, encoding='utf-8').read()\n"
            "    if f.endswith('code.py'):\n"
            "        t = 'output = 2\\n'\n"
            "    if f.endswith('instructions.md'):\n"
            "        t = t + 'سطر جديد\\n'\n"
            "    open(f, 'w', encoding='utf-8').write(t)\n",
            encoding="utf-8")
        editor = '"%s" "%s"' % (sys.executable, script)
        code, out, err = self.cli("open", "1.1.1", "--split", "--editor",
                                  editor)
        self.assertEqual(code, 0, err)
        self.assertEqual((self.root / "seen.txt").read_text(),
                         "code.py human.md instructions.md")
        n = self.store().nodes[nid]
        self.assertEqual((n.human, n.instructions, n.code),
                         ("مفهوم", "تعليمات\nسطر جديد", "output = 2"))
        self.assertIn("saved: instructions, code", out)

    def test_reading_editor_opens_only_when_there_is_a_book(self):
        self.tree()
        _, out, _ = self.cli("open", "1", "--editor",
                             '"%s" -c pass' % sys.executable)
        self.assertIn("opening human, machine with", out)
        self.assertIn("no changes", out)


if __name__ == "__main__":
    unittest.main()
