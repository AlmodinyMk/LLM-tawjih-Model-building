"""Tests for the commands added in 0.7: restructuring, versions, state,
search, outputs and services. Run:  python -m unittest discover -s tests"""
import argparse
import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base, tawjih  # noqa: E402


class Restructure(Base):
    def setUp(self):
        super().setUp()
        self.tree()                     # 1 / 1.1 / 1.1.1 ; 2
        st = self.store()
        self.top, self.sec, self.topic, self.other = (
            st.resolve(r) for r in ("1", "1.1", "1.1.1", "2"))

    def test_move_keeps_ids_shifts_levels_and_paths_still_work(self):
        self.fill("1.1.1", instructions="x")
        self.cli("produce", "p", "1.1.1", "--plan")
        code, out, err = self.cli("move", "1.1", "--to", "2")
        self.assertEqual(code, 0, err)
        st = self.store()
        self.assertEqual(st.number(self.sec), "2.1")
        self.assertEqual(st.number(self.topic), "2.1.1")
        self.assertEqual((st.nodes[self.sec].level, st.nodes[self.topic].level),
                         (2, 4))                  # topic kept its gap of 2
        self.assertIn("paths name files by id and need no change: p", out)
        self.assertEqual(self.cli("check")[0], 0)
        self.assertTrue(list((self.root / "history").glob(self.sec + "-*.md")))

    def test_move_refuses_its_own_branch_and_too_deep(self):
        self.assertIn("under itself", self.cli("move", "1", "--to", "1.1")[2])
        for t in ("a", "b", "c"):
            self.cli("new", t, "--parent", "2" if t == "a" else
                     {"b": "2.1", "c": "2.1.1"}[t])
        code, _, err = self.cli("move", "1.1", "--to", "2.1.1.1")
        self.assertEqual(code, 1)
        self.assertIn("deeper than level 6", err)

    def test_move_to_root_and_position(self):
        self.cli("move", "1.1", "--root")
        st = self.store()
        self.assertEqual(st.nodes[self.sec].parent, "")
        self.assertEqual(st.number(self.sec), "3")
        self.cli("move", "3", "--root", "--position", "1")
        self.assertEqual(self.store().number(self.sec), "1")

    def test_rename_updates_among_deciders(self):
        self.cli("new", "Router", "--parent", "2")
        self.cli("new", "Other", "--parent", "2")
        st = self.store()
        r, o = st.resolve("2.1"), st.resolve("2.2")
        self.fill("2.1", instructions="Name the file: المعادلات or Other")
        self.fill("1.1.1", instructions="solve")
        self.fill("2.2", instructions="x", code="")
        self.cli("produce", "route", "2.1", "AMONG", "1.1.1", "2.2", "--plan")
        self.fill("1", instructions="Do not confuse with المعادلات")
        code, out, err = self.cli("rename", "1.1.1", "Equations")
        self.assertEqual(code, 0, err)
        st = self.store()
        self.assertEqual(st.nodes[self.topic].title, "Equations")
        self.assertEqual(st.nodes[r].instructions, "Name the file: Equations or Other")
        self.assertIn("mentions the old title", out)
        self.assertEqual(st.nodes[self.top].instructions,
                         "Do not confuse with المعادلات")
        self.assertIsNotNone(o)

    def test_copy_a_file_and_a_branch(self):
        self.fill("1.1.1", human="h", code="output = 1", produces="text")
        self.cli("tag", "1.1.1", "algebra")
        self.cli("link", "1.1.1", "1.1")
        code, out, err = self.cli("copy", "1.1.1", "--to", "2", "--title", "Copy")
        self.assertEqual(code, 0, err)
        st = self.store()
        c = st.nodes[st.resolve("2.1")]
        self.assertNotEqual(c.id, self.topic)
        self.assertEqual((c.title, c.human, c.code, c.produces, c.tags, c.level),
                         ("Copy", "h", "output = 1", "text", ["algebra"], 2))
        self.assertEqual(c.links[0][0], self.sec)           # outside: kept
        self.cli("copy", "1.1", "--to", "2", "--branch")
        st = self.store()
        sec2 = st.resolve("2.2")
        kid = st.children(sec2)[0]
        self.assertEqual(kid.title, "المعادلات")
        self.assertEqual(kid.links[0][0], sec2)            # inside: remapped

    def test_rm_is_careful(self):
        self.fill("1.1.1", instructions="x")
        self.cli("produce", "p", "1.1.1", "--plan")
        code, _, err = self.cli("rm", "1.1.1", "--yes")
        self.assertIn("used by path(s) p", err)
        code, _, err = self.cli("rm", "1.1", "--yes")
        self.assertIn("move them first", err)
        (self.root / "paths" / "p.path").unlink()
        self.cli("link", "2", "1.1.1")
        self.assertIn("--unlink", self.cli("rm", "1.1.1", "--yes")[2])
        code, out, _ = self.cli("rm", "1.1.1", "--unlink")
        self.assertIn("nothing removed", out)
        self.assertIn(self.topic, self.store().nodes)
        code, out, err = self.cli("rm", "1.1.1", "--unlink", "--yes")
        self.assertEqual(code, 0, err)
        st = self.store()
        self.assertNotIn(self.topic, st.nodes)
        self.assertEqual(st.nodes[self.other].links, [])
        code, out, err = self.cli("revert", self.topic)
        self.assertEqual(code, 0, err)
        self.assertIn("restored", out)
        self.assertEqual(self.store().number(self.topic), "1.1.1")

    def test_rm_branch(self):
        code, out, err = self.cli("rm", "1.1", "--branch", "--yes")
        self.assertEqual(code, 0, err)
        st = self.store()
        self.assertFalse({self.sec, self.topic} & set(st.nodes))

    def test_split_and_merge(self):
        reading = "## A\n[source: b]\none\n\n## B\n[source: b]\ntwo\n\n## C\nthree"
        self.fill("1.1.1", human="h", reading=reading, code="output = 1")
        st = self.store()
        n = st.nodes[self.topic]
        n.built_sha = n.source_digest()
        st.save(n)
        out = self.cli("split", "1.1.1")[1]
        self.assertIn("2  ## B", out)
        code, out, err = self.cli("split", "1.1.1", "--passages", "2-3",
                                  "--title", "Second half")
        self.assertEqual(code, 0, err)
        self.assertIn("now stale", out)
        st = self.store()
        new = st.nodes[st.resolve("1.1.2")]
        self.assertEqual(new.title, "Second half")
        self.assertTrue(new.reading.startswith("## B"))
        self.assertNotIn("## B", st.nodes[self.topic].reading)
        self.assertIn("every passage", self.cli("split", "1.1.2", "--passages",
                                                "1-2", "--title", "x")[2])
        # merge it back
        self.fill("1.1.2", human="second")
        self.cli("link", "2", "1.1.2")
        code, out, err = self.cli("merge", "1.1.1", "1.1.2")
        self.assertEqual(code, 0, err)
        st = self.store()
        self.assertNotIn(new.id, st.nodes)
        a = st.nodes[self.topic]
        self.assertIn("## B", a.reading)
        self.assertIn("second", a.human)
        self.assertEqual(st.nodes[self.other].links[0][0], self.topic)

    def test_merge_refuses_a_machine_side(self):
        self.fill("2", instructions="x")
        code, _, err = self.cli("merge", "1", "2")
        self.assertIn("machine side", err)
        code, out, err = self.cli("merge", "1.1.1", "1.1", "--keep")
        self.assertEqual(code, 0, err)


class Versions(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.id = self.store().resolve("1.1.1")
        for text in ("first", "second"):
            self.replies = [text]
            self.cli("edit", "1.1.1", "--side", "human", "--ask", "x",
                     "--backend", "fake", "--yes")

    def test_log_and_diff(self):
        out = self.cli("log", "1.1.1")[1]
        self.assertIn("current", out)
        self.assertIn("then changed: human", out)
        code, out, err = self.cli("diff", "1.1.1")
        self.assertIn("-first", out)
        self.assertIn("+second", out)
        self.assertIn("versions run from 1", self.cli("diff", "1.1.1",
                                                      "--version", "99")[2])

    def test_revert_a_side_and_revert_the_revert(self):
        code, out, err = self.cli("revert", "1.1.1", "--side", "human")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.store().nodes[self.id].human, "first")
        self.cli("revert", "1.1.1", "--side", "human")
        self.assertEqual(self.store().nodes[self.id].human, "second")

    def test_reverting_the_machine_side_brings_back_its_build_digest(self):
        self.fill("1.1.1", code="output = 1")
        st = self.store()
        n = st.nodes[self.id]
        n.built_sha = n.source_digest()
        st.save(n)
        self.replies = ["output = 2"]
        self.cli("edit", "1.1.1", "--side", "code", "--ask", "x", "--backend",
                 "fake", "--yes")
        self.replies = ["third"]
        self.cli("edit", "1.1.1", "--side", "human", "--ask", "x", "--backend",
                 "fake", "--yes")
        self.cli("revert", "1.1.1", "--version", "2", "--side", "code")
        n = self.store().nodes[self.id]
        self.assertEqual(n.code, "output = 1")
        self.assertTrue(n.is_stale())     # built against "second", now "third"


class State(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.id = self.store().resolve("1.1.1")

    def test_status_lists_only_what_is_unfinished(self):
        for r in ("1", "1.1", "1.1.1", "2"):
            self.fill(r, human="h")
        self.cli("new", "x", "--parent", "2")
        self.fill("2.1", human="h")
        self.assertIn("nothing unfinished", self.cli("status")[1])
        self.fill("1.1.1", code="output = 1")
        self.cli("produce", "p", "1.1.1", "TO", "2", "--plan")
        out = self.cli("status")[1]
        self.assertIn("in a path but without a machine side: 1", out)
        self.assertIn("machine side without contracts: 1", out)

    def test_tags_and_search_by_tag(self):
        self.cli("tag", "1.1.1", "algebra,core")
        self.cli("tag", "2", "algebra")
        out = self.cli("tag")[1]
        self.assertIn("algebra (2)", out)
        self.assertEqual(self.store().nodes[self.id].tags, ["algebra", "core"])
        self.assertIn("1 matching", self.cli("search", "ال", "--tag", "core")[1])
        self.cli("untag", "1.1.1", "core")
        self.assertEqual(self.store().nodes[self.id].tags, ["algebra"])
        self.assertEqual(self.cli("tag", "2", "bad tag!")[0], 1)

    def test_sync_finds_edits_made_outside(self):
        self.fill("1.1.1", human="h", code="output = 1")
        st = self.store()
        n = st.nodes[self.id]
        n.built_sha = n.source_digest()
        st.save(n)
        f = self.root / "files" / self.id / "machine.md"
        f.write_text(f.read_text(encoding="utf-8").replace("output = 1",
                                                           "output = 2"),
                     encoding="utf-8")
        self.assertIn("changed outside Tawjih", self.cli("status")[1])
        out = self.cli("sync")[1]
        self.assertIn("changed outside Tawjih: 1.1.1", out)
        self.assertIn("machine", out)
        self.assertIn("in sync", self.cli("sync")[1])
        h = self.root / "files" / self.id / "human.md"
        h.write_text(h.read_text(encoding="utf-8").replace("\nh\n", "\nH2\n"),
                     encoding="utf-8")
        out = self.cli("sync")[1]
        self.assertIn("now stale", out)
        f.write_text(f.read_text(encoding="utf-8").replace("output = 2",
                                                           "output = 3"),
                     encoding="utf-8")
        self.cli("sync", "--confirm-machine")
        self.assertFalse(self.store().nodes[self.id].is_stale())

    def test_a_project_from_before_0_7_starts_from_its_current_state(self):
        (self.root / "state.json").unlink()
        self.fill("2", human="changed by a command")
        self.assertIn("in sync", self.cli("sync")[1])

    def test_sync_cleans_work(self):
        (self.root / "work" / self.id).mkdir(parents=True)
        (self.root / "work" / self.id / "human.txt").write_text("x")
        self.assertIn("unfinished edit", self.cli("sync")[1])
        self.cli("sync", "--clean-work")
        self.assertFalse((self.root / "work").exists())

    def test_refs(self):
        self.cli("link", "2", "1.1.1", "--type", "uses")
        self.fill("1.1.1", instructions="x")
        self.fill("2", instructions="see المعادلات")
        self.cli("produce", "p", "2", "TO", "1.1.1", "--plan")
        out = self.cli("refs", "1.1.1")[1]
        self.assertIn("linked from: 2", out)
        self.assertIn("path p: from 2", out)
        self.assertIn("machine side mentions its title: 2", out)

    def test_clean(self):
        st = self.store()
        n = st.nodes[self.id]
        n.links = [("abcdef", "related")]
        st.save(n)
        (self.root / "work" / "x").mkdir(parents=True)
        out = self.cli("clean")[1]
        self.assertIn("broken link(s)", out)
        self.assertIn("unfinished edit work/x", out)
        self.assertIn("nothing changed", out)
        self.cli("clean", "--yes")
        self.assertEqual(self.store().nodes[self.id].links, [])
        self.assertFalse((self.root / "work").exists())


class Audit(Base):
    def test_findings(self):
        found = tawjih.audit_code(
            "import subprocess as sp\nimport os\nfrom shutil import rmtree\n"
            "from pathlib import Path\n"
            "sp.run(['ls'])\nos.remove('x')\nrmtree('d')\n"
            "open('f', 'w').write('x')\nPath('p').unlink()\neval('1')\n"
            "output = vars.get('a')\n")
        msgs = " ".join(m for _, _, m in found)
        for want in ("imports subprocess", "subprocess.run", "os.remove",
                     "shutil.rmtree", "opens a file for writing", "unlink",
                     "eval"):
            self.assertIn(want, msgs)
        self.assertNotIn("vars.get", msgs)
        self.assertEqual(tawjih.audit_code("import json\noutput = "
                                           "json.dumps({'a': 1})"), [])

    def test_command_and_warning_before_a_run(self):
        self.tree()
        self.fill("2", code="import os\nos.system('echo hi')\noutput = input")
        code, out, _ = self.cli("audit")
        self.assertEqual(code, 1)
        self.assertIn("os.system: runs a shell command", out)
        self.fill("2", code="output = input")
        self.assertEqual(self.cli("audit", "2")[0], 0)
        self.fill("2", code="import os\nos.remove('nothing-here')\noutput = input")
        code, out, err = self.cli("produce", "p", "2", "--input", "x",
                                  "--backend", "fake", "--allow-code")
        self.assertIn("audit:", err)


class RankedSearch(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.fill("1.1.1", reading="## A\ncyclic groups and generators\n\n"
                  "## B\nthe weather today")
        self.fill("2", human="programming with groups of files")

    def test_rank_orders_by_relevance(self):
        out = self.cli("search", "cyclic generators", "--rank")[1]
        first = out.splitlines()[0]
        self.assertIn("المعادلات", first)
        self.assertIn("(reading)", first)
        self.assertNotIn("weather", out)

    def test_semantic_with_an_index(self):
        calls = []

        def emb(texts, model):
            calls.append(len(texts))
            return [[t.count("cyclic"), t.count("weather"), t.count("files"),
                     1.0] for t in texts]
        tawjih.EMBEDDERS["fake"] = emb
        try:
            self.assertIn("no semantic index", self.cli(
                "search", "cyclic", "--semantic")[2])
            code, out, err = self.cli("index", "--backend", "fake")
            self.assertEqual(code, 0, err)
            first_batch = sum(calls)
            self.cli("index", "--backend", "fake")
            self.assertEqual(sum(calls), first_batch)       # nothing new
            out = self.cli("search", "weather", "--semantic", "--top", "1")[1]
            self.assertIn("the weather today", out)
        finally:
            tawjih.EMBEDDERS.pop("fake", None)


class Outputs(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.fill("1.1.1", human="An <equation> & more",
                  reading="## Passage\n[source: book.txt]\nText of the book",
                  instructions="solve", code="output = 1")

    def test_export_markdown_and_html(self):
        out = self.cli("export")[1]
        self.assertIn("# 1 الرياضيات", out)
        self.assertIn("### 1.1.1 المعادلات", out)
        self.assertIn("> **Passage**", out)
        self.assertIn("> [source: book.txt]", out)
        self.assertNotIn("solve", out)
        self.assertIn("solve", self.cli("export", "--machine")[1])
        f = self.root / "book.html"
        self.cli("export", "--branch", "1.1", "-o", str(f))
        html = f.read_text(encoding="utf-8")
        self.assertIn("<h3 dir=\"auto\">1.1.1 المعادلات</h3>", html)
        self.assertIn("<p><strong>Passage</strong><br>[source: book.txt]", html)
        self.assertIn("An &lt;equation&gt; &amp; more", html)
        self.assertIn("<blockquote", html)
        self.assertNotIn("البرمجة", html)
        self.assertNotIn("MathJax", html)
        self.fill("1.1.1", human="The group $\\mathbb{Z}_6$")
        self.cli("export", "-o", str(f))
        self.assertIn("mathjax@3", f.read_text(encoding="utf-8"))
        self.cli("export", "-o", str(f), "--no-mathjax")
        self.assertNotIn("mathjax", f.read_text(encoding="utf-8"))

    def test_pack_and_unpack(self):
        self.cli("produce", "p", "1.1.1", "--plan")
        pack = self.root / "b.zip"
        code, out, err = self.cli("pack", "--branch", "1", "-o", str(pack))
        self.assertEqual(code, 0, err)
        self.assertIn("3 file(s) and 1 path(s)", out)
        self.assertIn("already exist", self.cli("unpack", str(pack))[2])
        code, out, err = self.cli("unpack", str(pack), "--new-ids",
                                  "--parent", "2")
        self.assertEqual(code, 0, err)
        st = self.store()
        copy = st.nodes[st.resolve("2.1.1.1")]
        self.assertEqual(copy.reading.split("\n")[0], "## Passage")
        self.assertEqual(copy.level, 5)
        new_path = (self.root / "paths" / "p-2.path").read_text()
        self.assertIn(copy.id, new_path)
        self.assertEqual(self.cli("check")[0], 0)
        bad = self.root / "bad.zip"
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("../evil.md", "x")
        self.assertIn("not a Tawjih pack", self.cli("unpack", str(bad))[2])
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("manifest.json", json.dumps({"files": [], "paths": ["../x"]}))
        self.assertIn("not valid", self.cli("unpack", str(bad))[2])

    def test_unpack_into_another_project(self):
        pack = self.root / "b.zip"
        self.cli("pack", "--branch", "1.1", "-o", str(pack))
        other = Path(self.dir) / "other"
        tawjih_args = ["tawjih.py", "-p", str(other)]
        old = sys.argv
        try:
            sys.argv = tawjih_args + ["init"]
            tawjih.main()
            sys.argv = tawjih_args + ["unpack", str(pack)]
            tawjih.main()
        finally:
            sys.argv = old
        st = tawjih.Store(other)
        self.assertEqual(len(st.nodes), 2)
        self.assertEqual([n.title for n in st.children("")], ["الجبر"])

    def docx(self):
        W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" ' \
            'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"'
        doc = ('<w:document %s><w:body>'
               '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Sets</w:t></w:r></w:p>'
               '<w:p><w:r><w:t>A set is a collection</w:t></w:r><m:oMath><m:r><m:t>x∈A</m:t></m:r></m:oMath></w:p>'
               '<w:p><w:pPr><w:numPr/></w:pPr><w:r><w:t>first</w:t></w:r></w:p>'
               '<w:p><w:pPr><w:numPr/></w:pPr><w:r><w:t>second</w:t></w:r></w:p>'
               '<w:p><w:pPr><w:pStyle w:val="2"/></w:pPr><w:r><w:t>العلاقات</w:t></w:r></w:p>'
               '<w:tbl><w:tr><w:tc><w:p><w:r><w:t>a</w:t></w:r></w:p></w:tc>'
               '<w:tc><w:p><w:r><w:t>b</w:t></w:r></w:p></w:tc></w:tr></w:tbl>'
               '</w:body></w:document>' % W)
        styles = ('<w:styles %s><w:style w:styleId="Heading1"><w:name w:val="heading 1"/></w:style>'
                  '<w:style w:styleId="2"><w:name w:val="عنوان 2"/></w:style></w:styles>' % W)
        f = self.root / "d.docx"
        with zipfile.ZipFile(f, "w") as z:
            z.writestr("word/document.xml", doc)
            z.writestr("word/styles.xml", styles)
        return f

    def test_ingest_docx_and_deposit(self):
        f = self.docx()
        code, out, err = self.cli("ingest", str(f), "--deposit", "--with",
                                  "builtin")
        self.assertEqual(code, 0, err)
        md = (self.root / "d.md").read_text(encoding="utf-8")
        self.assertIn("# Sets", md)
        self.assertIn("## العلاقات", md)
        self.assertIn("A set is a collection$x∈A$", md)
        self.assertIn("- first\n- second", md)
        self.assertIn("| a | b |", md)
        self.assertIn("1 formula(s) were flattened", out)
        self.assertIn("deposited", out)

    def test_ingest_epub_and_pdf_without_the_tool(self):
        f = self.root / "b.epub"
        with zipfile.ZipFile(f, "w") as z:
            z.writestr("META-INF/container.xml",
                       '<container><rootfiles><rootfile full-path="OEBPS/c.opf"/>'
                       '</rootfiles></container>')
            z.writestr("OEBPS/c.opf", '<package><manifest><item id="a" '
                       'href="a.xhtml"/></manifest><spine><itemref idref="a"/>'
                       '</spine></package>')
            z.writestr("OEBPS/a.xhtml", "<html><body><h1>Chapter</h1><p>Text"
                       "</p></body></html>")
        code, out, err = self.cli("ingest", str(f), "--with", "builtin")
        self.assertEqual(code, 0, err)
        self.assertIn("# Chapter", (self.root / "b.md").read_text())
        pdf = self.root / "x.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        real = tawjih.shutil.which
        tawjih.shutil.which = lambda name: None
        try:
            self.assertIn("no converter for .pdf is installed",
                          self.cli("ingest", str(pdf))[2])
        finally:
            tawjih.shutil.which = real
        self.assertIn("ingest x.pdf --deposit", self.cli("deposit", str(pdf))[2])


class Serve(Base):
    def test_api(self):
        self.tree()
        self.fill("1.1.1", instructions="answer")
        self.fill("2", code="output = input")
        self.cli("produce", "ask", "1.1.1", "--plan")
        self.cli("produce", "coded", "2", "--plan")
        ready = threading.Event()
        box = {}

        def on_ready(server):
            box["server"] = server
            ready.set()
        a = argparse.Namespace(project=self.dir, token="secret", paths=None,
                               host="127.0.0.1", port=0, backend="fake",
                               allow_code=False, resume=False,
                               code_timeout=5.0, retries=None, approve=None,
                               _ready=on_ready)
        t = threading.Thread(target=lambda: tawjih.cmd_serve(a), daemon=True)
        old_out = sys.stdout
        sys.stdout = open(self.root / "serve.out", "w")
        try:
            t.start()
            self.assertTrue(ready.wait(5))
            port = box["server"].server_address[1]
            base = "http://127.0.0.1:%d" % port

            def call(path, body=None, token="secret"):
                req = urllib.request.Request(
                    base + path, data=None if body is None else
                    json.dumps(body).encode(), headers={
                        "Authorization": "Bearer " + token,
                        "Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=10) as r:
                        return r.status, json.loads(r.read())
                except urllib.error.HTTPError as e:
                    return e.code, json.loads(e.read())
            self.assertEqual(call("/health", token="")[0], 200)
            self.assertEqual(call("/paths", token="wrong")[0], 401)
            code, data = call("/paths")
            self.assertEqual({d["name"] for d in data}, {"ask", "coded"})
            self.replies = ["the answer"]
            code, data = call("/paths/ask", {"input": "q"})
            self.assertEqual((code, data["result"]), (200, "the answer"))
            self.assertEqual(call("/paths/coded", {"input": "q"})[0], 403)
            self.assertEqual(call("/paths/none", {"input": "q"})[0], 404)
            self.assertEqual(call("/paths/ask", {"x": 1})[0], 400)
        finally:
            if "server" in box:
                box["server"].shutdown()
            sys.stdout.close()
            sys.stdout = old_out


class EvalAndDataset(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.fill("2", code="output = input.upper()")
        self.fill("1.1.1", instructions="explain")
        self.cli("produce", "up", "2", "TO", "1.1.1", "--plan")
        cases = [{"input": "abc", "expect": "ANSWER ABC"},
                 {"input": "x", "expect": ["no such"], "name": "fails"},
                 {"input": "y", "expect": {"k": 1}},
                 {"input": "z", "expect": "(?i)^answer z", "match": "regex"}]
        (self.root / "paths" / "up.cases.jsonl").write_text(
            "\n".join(json.dumps(c) for c in cases), encoding="utf-8")

    def reply(self, prompt, model=""):
        if '"k"' in prompt or "\nY" in prompt:
            return '{"k": 1}'
        return "answer " + prompt.rsplit("\n", 1)[-1].strip()

    def test_eval_reports_and_records(self):
        tawjih.BACKENDS["fake"] = self.reply
        self.assertIn("--allow-code", self.cli("eval", "up", "--backend",
                                               "fake")[2])
        code, out, err = self.cli("eval", "up", "--backend", "fake",
                                  "--allow-code")
        self.assertEqual(code, 1, err)
        self.assertIn("FAIL fails", out)
        self.assertIn("3 of 4 passed", out)
        rec = json.loads(next((self.root / "runs").glob("eval-up-*.json"))
                         .read_text(encoding="utf-8"))
        self.assertEqual([c["pass"] for c in rec["cases"]],
                         [True, False, True, True])

    def test_dataset_from_eval_runs_and_steps(self):
        tawjih.BACKENDS["fake"] = self.reply
        self.cli("eval", "up", "--backend", "fake", "--allow-code")
        f = self.root / "d.jsonl"
        code, out, err = self.cli("dataset", "-o", str(f), "--meta")
        self.assertEqual(code, 0, err)
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["messages"][0], {"role": "user", "content": "abc"})
        self.assertTrue(rows[0]["meta"]["pass"])
        self.cli("dataset", "-o", str(f), "--source", "runs", "--format",
                 "prompt")
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
        self.assertIn({"prompt": "abc", "completion": "answer ABC"}, rows)
        self.cli("dataset", "-o", str(f), "--source", "steps")
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(rows and "explain" in rows[0]["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
