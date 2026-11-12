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
        self.replies = ["21", "تم"]
        code, _, err = self.cli("run", "p", "--input", "x", "--backend", "fake")
        self.assertEqual(code, 0)
        self.assertIn("code skipped", err)
        self.assertNotIn("42", self.prompts[1])

    def test_failing_code_stops_with_a_log(self):
        self.make_path()
        self.replies = ["ليس عدداً"]
        code, _, err = self.cli("run", "p", "--input", "x", "--backend",
                                "fake", "--allow-code")
        self.assertEqual(code, 1)
        self.assertIn("step 2", err)
        log = next((self.root / "runs").glob("p-*.md")).read_text(encoding="utf-8")
        self.assertIn("FAILED", log)


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

    def test_reading_editor_opens_only_when_there_is_a_book(self):
        self.tree()
        _, out, _ = self.cli("open", "1", "--editor",
                             '"%s" -c pass' % sys.executable)
        self.assertIn("opening human, machine with", out)
        self.assertIn("no changes", out)


if __name__ == "__main__":
    unittest.main()
