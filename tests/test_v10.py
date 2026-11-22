"""Tests for 0.10: lock, alias, review, stage, mount, sandbox, FOR EACH,
trace, release, snapshot, seal, inherit, template, scaffold, gc, compact,
format, archive, glossary, doctor, schedule and trigger."""
import json
import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base, tawjih  # noqa: E402


class Governance(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.id = self.store().resolve("1.1.1")

    def test_lock_blocks_change_and_removal_but_not_unlock(self):
        self.assertEqual(self.cli("lock", "1", "--branch")[0], 0)
        self.assertTrue(self.store().nodes[self.id].locked)
        with self.assertRaises(tawjih.Die):
            self.fill("1.1.1", human="new")
        code, _, err = self.cli("rm", "1.1.1", "--yes")
        self.assertNotEqual(code, 0)
        self.cli("unlock", "1.1.1")
        self.fill("1.1.1", human="new")
        self.assertEqual(self.store().nodes[self.id].human, "new")
        self.assertEqual(self.cli("freeze", "2")[0], 0)

    def test_alias_resolves_and_rejects_clashes(self):
        self.cli("alias", "1.1.1", "eqs")
        self.assertEqual(self.store().resolve("EQS"), self.id)
        self.assertIn("already an alias", self.cli("alias", "2", "eqs")[2])
        self.assertIn("neither an id", self.cli("alias", "2", "1.2")[2])
        self.assertIn("eqs", self.cli("alias")[1])
        self.cli("show", "eqs")
        self.cli("unalias", "1.1.1", "eqs")
        self.assertEqual(self.store().nodes[self.id].aliases, [])

    def test_review_tracks_changes_after_review(self):
        out = self.cli("review")[1]
        self.assertIn("never reviewed: 4", out)
        self.cli("review", "--ok", "1", "--branch")
        out = self.cli("review")[1]
        self.assertIn("never reviewed: 1", out)
        self.fill("1.1.1", human="changed")
        self.assertIn("changed since reviewed: 1", self.cli("review")[1])

    def test_edit_stage_then_apply(self):
        self.fill("1.1.1", human="h")
        self.replies = ["new rules"]
        code, out, err = self.cli("edit", "1.1.1", "--side", "instructions",
                                  "--ask", "write", "--backend", "fake",
                                  "--stage")
        self.assertEqual(code, 0, err)
        self.assertIn("staged", out)
        self.assertEqual(self.store().nodes[self.id].instructions, "")
        self.assertIn("+new rules", self.cli("stage", "--show")[1])
        self.assertIn("applied", self.cli("stage", "--apply")[1])
        n = self.store().nodes[self.id]
        self.assertEqual(n.instructions, "new rules")
        self.assertFalse(n.is_stale())
        self.assertIn("nothing staged", self.cli("stage")[1])

    def test_stage_refuses_when_side_changed_meanwhile(self):
        n = self.store().nodes[self.id]
        tawjih.stage_change(self.store(), n, "human", "proposed")
        self.fill("1.1.1", human="edited by hand")
        self.assertIn("skipped", self.cli("stage", "--apply")[1])
        self.cli("stage", "--drop")
        self.assertIn("nothing staged", self.cli("stage")[1])

    def test_inherit_fills_missing_fields_only(self):
        self.cli("contract", "1", "--model", "m1", "--split", "eval")
        self.cli("contract", "1.1", "--model", "own")
        self.cli("tag", "1", "math")
        self.cli("inherit", "1")
        st = self.store()
        self.assertEqual(st.nodes[st.resolve("1.1")].model, "own")
        self.assertEqual(st.nodes[self.id].model, "m1")
        self.assertEqual(st.nodes[self.id].split, "eval")
        self.assertIn("math", st.nodes[self.id].tags)
        self.cli("inherit", "1", "--fields", "model", "--overwrite")
        self.assertEqual(self.store().nodes[st.resolve("1.1")].model, "m1")
        self.assertIn("cannot inherit", self.cli("inherit", "1", "--fields",
                                                 "title")[2])

    def test_template_and_scaffold(self):
        self.fill("1.1", human="section text", instructions="do")
        self.cli("template", "save", "sec", "1.1", "--branch")
        self.assertIn("2 file(s)", self.cli("template", "list")[1])
        code, out, err = self.cli("scaffold", "sec", "--parent", "2",
                                  "--title", "Copy")
        self.assertEqual(code, 0, err)
        st = self.store()
        new = st.nodes[st.resolve("2.1")]
        self.assertEqual((new.title, new.human, new.level), ("Copy",
                                                            "section text", 2))
        self.assertEqual(len(st.children(new.id)), 1)
        self.cli("template", "rm", "sec")
        self.assertIn("no templates", self.cli("template", "list")[1])

    def test_format_and_compact(self):
        self.fill("1.1.1", human="a  \n\n\n\nb", tags=["x", "x"])
        code, out, _ = self.cli("format", "--check")
        self.assertEqual(code, 1)
        self.assertIn("would tidy", out)
        self.cli("format")
        n = self.store().nodes[self.id]
        self.assertEqual((n.human, n.tags), ("a\n\nb", ["x"]))
        self.assertEqual(self.cli("format", "--check")[0], 0)
        st = self.store()
        n = st.nodes[st.resolve("2")]
        n.order = 7
        st.save(n)
        self.cli("compact")
        self.assertEqual(self.store().nodes[n.id].order, 2)

    def test_archive_and_restore(self):
        self.cli("archive", "1.1")
        self.assertIn(self.id, self.store().nodes)       # no --yes
        out = self.cli("archive", "1.1", "--yes")[1]
        self.assertIn("archived 2", out)
        self.assertNotIn(self.id, self.store().nodes)
        self.assertIn("الجبر", self.cli("archive", "--list")[1])
        sid = out.split("--restore ")[1].strip()
        self.cli("archive", "--restore", sid)
        st = self.store()
        self.assertEqual(st.resolve("1.1.1"), self.id)
        self.assertIn("nothing archived", self.cli("archive", "--list")[1])

    def test_archive_refuses_files_used_by_paths(self):
        self.fill("1.1.1", instructions="x")
        self.cli("path", "p", "1.1.1")
        self.assertIn("used by path", self.cli("archive", "1.1", "--yes")[2])

    def test_glossary_and_conflicts(self):
        self.fill("1", human="**Ring**: a set\n\nfield: a ring with inverses")
        self.fill("2", human="field: a place on a form")
        out = self.cli("glossary")[1]
        self.assertIn("**Ring**", out)
        out = self.cli("glossary", "--conflicts")[1]
        self.assertIn("field", out)
        self.assertNotIn("Ring", out)

    def test_gc_keeps_newest_versions(self):
        for k in range(5):
            self.fill("1.1.1", human="v%d" % k)
            tawjih._backup(self.store(), self.store().nodes[self.id])
        hist = list((self.root / "history").glob(self.id + "-*.md"))
        self.assertGreaterEqual(len(hist), 5)
        self.assertIn("would remove", self.cli("gc", "--keep", "2")[1])
        self.cli("gc", "--keep", "2", "--yes")
        self.assertEqual(len(list((self.root / "history").glob(
            self.id + "-*.md"))), 2)


class Releases(Base):
    def setUp(self):
        super().setUp()
        self.make_path()

    def test_create_verify_diff_rollback(self):
        self.assertEqual(self.cli("release", "create", "v1")[0], 0)
        self.assertIn("v1", self.cli("release", "list")[1])
        self.assertIn("intact", self.cli("release", "verify", "v1")[1])
        self.assertIn("no differences", self.cli("release", "diff", "v1")[1])
        self.fill("2", human="changed")
        out = self.cli("release", "diff", "v1")[1]
        self.assertIn("changed", out)
        self.assertIn("human", out)
        self.assertIn("--yes", self.cli("release", "rollback", "v1")[1])
        out = self.cli("release", "rollback", "v1", "--yes")[1]
        self.assertIn("rolled back", out)
        st = self.store()
        self.assertNotEqual(st.nodes[st.resolve("2")].human, "changed")
        self.assertEqual(len(list((self.root / "releases").glob(
            "before-rollback-*.zip"))), 1)
        self.assertIn("already exists", self.cli("release", "create", "v1")[2])

    def test_snapshot_and_seal(self):
        self.assertIn("snapshot-", self.cli("snapshot")[1])
        self.cli("seal")
        self.assertEqual(self.cli("verify")[0], 0)
        self.fill("1", human="tampered")
        code, out, _ = self.cli("verify")
        self.assertEqual(code, 1)
        self.assertIn("changed files/", out)

    def test_trace_impact_and_output(self):
        out = self.cli("trace", "--impact", "1.1.1")[1]
        self.assertIn("path p: 2 step(s) after it", out)
        self.cli("release", "create", "r1")
        self.assertIn("in release(s): r1", self.cli("trace", "--impact",
                                                     "2")[1])
        self.fill("2", reading="[source: Book, ch. 1]\n" + "text")
        self.replies = ["21", "explained"]
        code, out, err = self.cli("run", "p", "--input", "21",
                                  "--backend", "fake", "--allow-code")
        self.assertEqual(code, 0, err)
        rec = sorted((self.root / "history").glob("production-*.json"))[-1]
        out = self.cli("trace", "--output", str(rec))[1]
        self.assertIn("run of path p", out)
        self.assertIn("reading from", out)
        self.assertEqual(self.cli("trace", "--cost")[0], 0)


class Mounts(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        import tempfile
        self.odir = tempfile.mkdtemp(prefix="tawjih-other-")
        old = self.dir
        self.dir = self.odir
        self.cli("init")
        self.cli("new", "Shared rules")
        st = self.store()
        self.sid = st.resolve("1")
        self.fill("1", instructions="shared step")
        self.dir = old

    def tearDown(self):
        import shutil
        shutil.rmtree(self.odir, ignore_errors=True)
        super().tearDown()

    def test_mount_read_only_and_usable_in_paths(self):
        code, out, err = self.cli("mount", self.odir, "--name", "lib")
        self.assertEqual(code, 0, err)
        self.assertIn("1 file(s)", out)
        st = self.store()
        self.assertEqual(st.nodes[self.sid].mount, "lib")
        with self.assertRaises(tawjih.Die):
            st.save(st.nodes[self.sid])
        self.fill("1.1.1", instructions="local step")
        self.assertEqual(self.cli("path", "p", "1.1.1", self.sid)[0], 0)
        code, out, err = self.cli("run", "p", "--backend", "fake")
        self.assertEqual(code, 0, err)
        self.assertIn("not unmounted", self.cli("unmount", "lib")[2])
        (self.root / "paths" / "p.path").unlink()
        self.assertIn("unmounted", self.cli("unmount", "lib")[1])
        self.assertNotIn(self.sid, self.store().nodes)


class SandboxAndEach(Base):
    def setUp(self):
        super().setUp()
        self.tree()

    @unittest.skipIf(os.name == "nt", "resource limits are POSIX only")
    def test_sandbox_runs_code_apart(self):
        self.fill("2", code="output = str(int(input) * 3)")
        code, out, err = self.cli("sandbox", "2", "--input", "5")
        self.assertEqual(code, 0, err)
        self.assertEqual(out.strip(), "15")
        self.fill("2", code="x = 'a' * (2000 * 1024 * 1024)\noutput = 'no'")
        code, out, err = self.cli("sandbox", "2", "--memory", "64")
        self.assertNotEqual(code, 0)

    def test_for_each_runs_once_per_item(self):
        self.fill("2", code="import json\n"
                  "output = str(json.loads(input)['n'] * 2)")
        self.cli("contract", "2", "--each", "rows")
        self.assertEqual(self.store().nodes[self.store().resolve("2")].each,
                         "rows")
        self.cli("path", "p", "2")
        code, out, err = self.cli("run", "p", "--allow-code", "--input",
                                  json.dumps({"rows": [{"n": 1}, {"n": 4}]}))
        self.assertEqual(code, 0, err)
        result = sorted((self.root / "runs").rglob("*result*"))
        text = result[-1].read_text(encoding="utf-8") if result else out
        self.assertIn('"items": [2, 8]', text)
        self.assertIn("each is", self.cli("contract", "2", "--each",
                                          "a b")[2])


class Doctor(Base):
    def setUp(self):
        super().setUp()
        self.tree()

    def test_healthy_then_problems_then_repair(self):
        self.assertIn("healthy", self.cli("doctor")[1])
        st = self.store()
        nid = st.resolve("1.1.1")
        (self.root / "files" / "bad000.md").write_text("garbage",
                                                       encoding="utf-8")
        f = self.root / "files" / nid / "head.json"
        f.write_text(f.read_text(encoding="utf-8").replace(
            st.nodes[nid].parent, "ffffff"), encoding="utf-8")
        (self.root / "state.json").write_text("{bad", encoding="utf-8")
        code, out, _ = self.cli("doctor")
        self.assertEqual(code, 1)
        self.assertIn("parent ffffff is missing", out)
        self.assertIn("state.json cannot be read", out)
        code, out, _ = self.cli("repair")
        self.assertEqual(code, 0)
        self.assertTrue((self.root / "quarantine").is_dir())
        self.assertIn("healthy", self.cli("doctor")[1])


class Schedules(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.fill("2", code="output = 'ran'")
        self.cli("path", "p", "2")

    def test_every_pause_resume_cancel(self):
        out = self.cli("schedule", "job", "--path", "p", "--every", "1m",
                       "--allow-code")[1]
        self.assertIn("schedule --run-due", out)
        self.assertIn("job: done", self.cli("schedule", "--run-due")[1])
        self.assertIn("nothing due", self.cli("schedule", "--run-due")[1])
        jobs = json.loads((self.root / "schedules.json").read_text())
        jobs["job"]["last_run_ts"] = time.time() - 120
        (self.root / "schedules.json").write_text(json.dumps(jobs))
        self.cli("pause", "job")
        self.assertIn("nothing due", self.cli("schedule", "--run-due")[1])
        self.cli("resume", "job")
        self.assertIn("job: done", self.cli("schedule", "--run-due")[1])
        self.assertIn("active", self.cli("schedule")[1])
        self.cli("cancel", "job")
        self.assertIn("nothing scheduled", self.cli("schedule")[1])
        self.assertIn("--every is", self.cli("schedule", "j", "--path", "p",
                                             "--every", "soon")[2])

    def test_trigger_on_change(self):
        self.cli("trigger", "t", "--path", "p", "--on-change", "1.1.1",
                 "--allow-code")
        self.assertIn("nothing due", self.cli("schedule", "--run-due")[1])
        self.fill("1.1.1", human="changed")
        self.assertIn("t: done", self.cli("schedule", "--run-due")[1])
        self.assertIn("nothing due", self.cli("schedule", "--run-due")[1])


if __name__ == "__main__":
    unittest.main()


class PackageLayout(unittest.TestCase):
    """tawjihlib is a chain: each module imports only from the ones before
    it, and only the standard library from outside."""
    def test_modules_import_only_earlier_ones_and_the_standard_library(self):
        import ast
        pkg = Path(tawjih.__file__).resolve().parent / "tawjihlib"
        order = list(tawjih.MODULES)
        self.assertEqual(sorted(order + ["__init__"]), sorted(
            [f.stem for f in pkg.glob("*.py")]))
        for k, m in enumerate(order):
            tree = ast.parse((pkg / (m + ".py")).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level:
                    self.assertIn(node.module, order[:k],
                                  "%s imports %s" % (m, node.module))
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [node.module] if isinstance(
                        node, ast.ImportFrom) else [a.name for a in node.names]
                    for name in names:
                        top = name.split(".")[0]
                        self.assertIn(top, sys.stdlib_module_names,
                                      "%s imports %s" % (m, name))

    def test_every_name_is_reachable_from_the_facade(self):
        import importlib
        for m in tawjih.MODULES:
            mod = importlib.import_module("tawjihlib." + m)
            for name in mod.__all__:
                self.assertIs(getattr(tawjih, name), getattr(mod, name))


class ReviewFixes(Base):
    """What the review of 0.10 found."""
    def setUp(self):
        super().setUp()
        self.tree()

    def test_a_release_cannot_write_outside_the_project(self):
        import zipfile
        rdir = self.root / "releases"
        rdir.mkdir()
        man = {"name": "evil", "files": {"../escaped.txt": "0"},
               "digest": "0"}
        with zipfile.ZipFile(rdir / "evil.zip", "w") as z:
            z.writestr("release.json", json.dumps(man))
            z.writestr("../escaped.txt", "x")
        code, _, err = self.cli("release", "rollback", "evil", "--yes")
        self.assertNotEqual(code, 0)
        self.assertIn("not safe", err)
        self.assertFalse((self.root.parent / "escaped.txt").exists())
        self.assertEqual(len(self.store().nodes), 4)
        self.assertIn("one word", self.cli("release", "verify",
                                           "../x")[2])

    def test_weight_zero_keeps_examples_out_of_training_data(self):
        self.fill("1.1.1", examples="Q: a\nA: b\nBAD: c")
        self.fill("2", examples="Q: d\nA: e")
        self.cli("contract", "2", "--weight", "0")
        out = self.root / "t.jsonl"
        self.cli("dataset", "--source", "examples", "-o", str(out),
                 "--dpo", str(self.root / "d.jsonl"))
        self.assertEqual(len(out.read_text().splitlines()), 1)
        man = json.loads(Path(str(out) + ".manifest.json").read_text())
        self.assertNotIn(self.store().resolve("2"), man["from"])

    def test_gc_by_age_keeps_the_newest_version(self):
        nid = self.store().resolve("1.1.1")
        for k in range(3):
            self.fill("1.1.1", human="v%d" % k)
            tawjih._backup(self.store(), self.store().nodes[nid])
        old = time.time() - 100 * 86400
        for f in (self.root / "history").glob(nid + "-*.md"):
            os.utime(f, (old, old))
        self.cli("gc", "--days", "30", "--yes")
        self.assertEqual(len(list((self.root / "history").glob(
            nid + "-*.md"))), 1)
