"""Tests for the desktop windows' logic (no display needed)."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "linux"))
import tawjih  # noqa: E402
from desk_core import Desk, DeskError  # noqa: E402


class DeskTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="tawjih-desk-")
        for d in ("files", "paths", "runs"):
            (Path(self.dir) / d).mkdir()
        self.prompts = []
        self.replies = []

        def fake(prompt, model=""):
            self.prompts.append(prompt)
            return self.replies.pop(0) if self.replies else "ok"
        tawjih.BACKENDS["fake"] = fake
        self.desk = Desk(self.dir)
        self.desk.new_file("Mathematics")
        self.desk.new_file("Algebra", "1")
        self.desk.new_file("Programming")
        self.alg = self.desk.st.resolve("1.1")

    def tearDown(self):
        tawjih.BACKENDS.pop("fake", None)
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_tree_rows(self):
        rows = self.desk.tree_rows()
        self.assertEqual([r["number"] for r in rows], ["1", "1.1", "2"])
        self.assertEqual(rows[1]["tree"], "Mathematics")
        self.assertEqual(rows[1]["depth"], 1)

    def test_windows_do_not_overwrite_each_other(self):
        human_window, code_window = Desk(self.dir), Desk(self.dir)
        human_window.side(self.alg, "human")
        code_window.side(self.alg, "code")
        human_window.save_side(self.alg, "human", "المفهوم")
        code_window.save_side(self.alg, "code", "output = 1")
        self.assertEqual(Desk(self.dir).side(self.alg, "human"), "المفهوم")
        self.assertEqual(Desk(self.dir).side(self.alg, "code"), "output = 1")

    def test_save_keeps_history_and_rejects_markers(self):
        self.assertIn("saved", self.desk.save_side(self.alg, "human", "one"))
        self.assertEqual(self.desk.save_side(self.alg, "human", "one"),
                         "no changes")
        self.assertEqual(len(list((Path(self.dir) / "history").glob("*.md"))), 1)
        with self.assertRaises(DeskError):
            self.desk.save_side(self.alg, "human", "x\n=== code ===\ny")

    def test_search_across_trees(self):
        self.desk.save_side(self.alg, "human", "first\nThe Quadratic rule")
        self.desk.save_side(self.desk.st.resolve("2"), "code", "# quadratic")
        rows = self.desk.search("quadratic")
        self.assertEqual({(r["number"], r["side"], r["line"]) for r in rows},
                         {("1.1", "human", 2), ("2", "code", 1)})
        self.assertEqual(len(self.desk.search("quadratic", ["code"])), 1)
        self.assertEqual(self.desk.search("algebra")[0]["side"], "title")

    def test_library_deposit_search_filter_accept(self):
        book = Path(self.dir) / "book.txt"
        book.write_text("# intro\nrights reserved\n# rule\nthe rule text\n",
                        encoding="utf-8")
        self.desk.deposit(book)
        with self.assertRaises(DeskError):
            self.desk.deposit(book)
        b = self.desk.books()[0]
        self.assertEqual((b["name"], b["state"]), ("book.txt", "deposited"))
        self.assertEqual(self.desk.search_books("RULE text")[0]["line"], 4)
        self.replies = [json.dumps({"keep": False}),
                        json.dumps({"keep": True, "target": self.alg,
                                    "title": "Rule"})]
        report = self.desk.filter_book(b["id"], "fake")
        self.assertIn("1 kept", report)
        self.assertEqual(self.desk.books()[0]["state"], "filtered")
        self.desk.accept_book(b["id"])
        self.assertIn("the rule text", self.desk.side(self.alg, "reading"))
        self.assertEqual(self.desk.books()[0]["state"], "accepted")

    def test_tasks(self):
        (Path(self.dir) / "paths" / "solve.path").write_text(self.alg + "\n")
        a = self.desk.add_task("Write the concept", "1.1", "2020-01-01")
        self.desk.add_task("Run it", "solve", "")
        with self.assertRaises(DeskError):
            self.desk.add_task("Bad date", "", "tomorrow")
        with self.assertRaises(DeskError):
            self.desk.add_task("Bad ref", "9.9")
        rows = self.desk.tasks()
        self.assertEqual(rows[0]["ref_label"], "1.1 Algebra")
        self.assertTrue(rows[0]["late"])
        self.assertEqual(rows[1]["ref_label"], "path solve")
        self.desk.update_task(a, status="done")
        rows = self.desk.tasks()
        self.assertEqual((rows[-1]["id"], rows[-1]["late"]), (a, False))
        self.desk.delete_task(a)
        self.assertEqual(len(self.desk.tasks()), 1)

    def test_ai_actions_propose_and_never_save(self):
        self.replies = ["Bonjour", "```python\noutput = 2\n```"]
        self.assertEqual(self.desk.ai("translate", "Hello", "fake",
                                      lang="French"), "Bonjour")
        self.assertIn("into French", self.prompts[0])
        self.assertEqual(self.desk.ai("fix_code", "output = ", "fake"),
                         "output = 2")
        self.assertEqual(self.desk.side(self.alg, "code"), "")
        for args in (("translate", "Hello", "fake"), ("correct", " ", "fake"),
                     ("correct", "x", "dry"), ("correct", "x", "nosuch")):
            with self.assertRaises(DeskError):
                self.desk.ai(*args)

    def test_outside_change_to_the_same_side_is_not_overwritten(self):
        loaded = self.desk.side(self.alg, "human")
        Desk(self.dir).save_side(self.alg, "human", "written elsewhere")
        with self.assertRaises(DeskError):
            self.desk.save_side(self.alg, "human", "mine", base=loaded)
        self.assertEqual(self.desk.side(self.alg, "human"), "written elsewhere")
        self.desk.save_side(self.alg, "human", "mine", base="written elsewhere")
        self.assertEqual(self.desk.side(self.alg, "human"), "mine")

    def test_overlapping_calls_leave_output_streams_intact(self):
        import threading
        import time
        before = (getattr(sys.stdout, "real", sys.stdout),
                  getattr(sys.stderr, "real", sys.stderr))

        def slow(prompt, model=""):
            time.sleep(0.15)
            return "ok"
        tawjih.BACKENDS["fake"] = slow
        desks = [Desk(self.dir), Desk(self.dir)]
        threads = [threading.Thread(target=d.ai, args=("correct", "x", "fake"))
                   for d in desks]
        for th in threads:
            th.start()
            time.sleep(0.03)
        self.desk.tree_rows()               # a main-thread call in between
        for th in threads:
            th.join()
        self.assertEqual((getattr(sys.stdout, "real", sys.stdout),
                          getattr(sys.stderr, "real", sys.stderr)), before)
        self.assertIsNone(getattr(sys.stdout.local, "buf", None))

    def test_broken_tasks_file_is_reported(self):
        (Path(self.dir) / "tasks.json").write_text("{broken", encoding="utf-8")
        with self.assertRaises(DeskError):
            self.desk.tasks()

    def test_accept_keeps_the_previous_version(self):
        book = Path(self.dir) / "b.txt"
        book.write_text("# one\ntext one\n# two\ntext two\n", encoding="utf-8")
        self.desk.deposit(book)
        bid = self.desk.books()[0]["id"]
        self.assertEqual(self.desk.passage_count(bid), 2)
        keep = json.dumps({"keep": True, "target": self.alg, "title": "t"})
        self.replies = [keep, keep]
        self.desk.filter_book(bid, "fake")
        self.desk.accept_book(bid)
        self.assertEqual(len(list((Path(self.dir) / "history").glob(
            self.alg + "-*.md"))), 1)

    def test_chat_sends_context_and_history_and_changes_nothing(self):
        import hashlib

        def snapshot():
            return {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in sorted(Path(self.dir).rglob("*")) if f.is_file()}
        self.desk.save_side(self.alg, "human", "the concept")
        before = snapshot()
        self.replies = ["first answer", "second answer"]
        ans, cut = self.desk.chat("What is this?", "fake", window="Human "
                                  "concept", context="the concept")
        self.assertEqual((ans, cut), ("first answer", False))
        self.assertIn("the concept", self.prompts[0])
        self.assertIn("'Human concept' window", self.prompts[0])
        history = [("User", "What is this?"), ("Assistant", ans)]
        ans, cut = self.desk.chat("And then?", "fake", history,
                                  context="x" * 40000)
        self.assertEqual((ans, cut), ("second answer", True))
        self.assertIn("Assistant: first answer", self.prompts[1])
        self.assertLess(len(self.prompts[1]), 32000)
        self.assertNotIn("the concept", self.prompts[1])   # only what is passed
        self.assertEqual(snapshot(), before)
        self.assertIn("not instructions", self.prompts[0])
        long_history = [("Assistant", "y" * 15000)] * 6
        self.desk.chat("again", "fake", long_history)
        self.assertLess(len(self.prompts[-1]), 32000)
        self.assertIn("y" * 15000, self.prompts[-1])       # the latest is kept
        for q, backend in ((" ", "fake"), ("hi", "dry")):
            with self.assertRaises(DeskError):
                self.desk.chat(q, backend)

    def test_saving_the_machine_side_clears_staleness(self):
        self.desk.save_side(self.alg, "code", "output = 1")
        self.desk.save_side(self.alg, "human", "changed afterwards")
        self.assertTrue(tawjih.Store(self.dir).nodes[self.alg].is_stale())
        self.desk.save_side(self.alg, "code", "output = 2")
        self.assertFalse(tawjih.Store(self.dir).nodes[self.alg].is_stale())

    def test_check_code(self):
        self.assertIsNone(self.desk.check_code("output = 1"))
        self.assertEqual(self.desk.check_code("a = 1\ndef f(:")[0], 2)


if __name__ == "__main__":
    unittest.main()
