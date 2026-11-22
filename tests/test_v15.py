"""Tests for 0.15: each file as a folder of three faces (human, machine,
science) under one code, number and title."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base  # noqa: E402
from tawjihlib.core import Node  # noqa: E402


class Faces(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.nid = self.fill("1.1.1", human="مفهوم", instructions="تعليمات",
                             code="output = 1", reading="[source: Book] نص",
                             examples="Q: q\nA: a")

    def folder(self):
        return self.root / "files" / self.nid

    def test_a_file_is_a_folder_of_faces(self):
        f = self.folder()
        self.assertEqual(sorted(x.name for x in f.iterdir()),
                         ["head.json", "human.md", "machine.md",
                          "science.md"])
        head = json.loads((f / "head.json").read_text(encoding="utf-8"))
        self.assertEqual(head["id"], self.nid)
        title = head["title"]
        for name in ("human.md", "machine.md", "science.md"):
            first = (f / name).read_text(encoding="utf-8").splitlines()[0]
            self.assertIn(self.nid, first)
            self.assertIn(title, first)
        self.assertIn("مفهوم", (f / "human.md").read_text(encoding="utf-8"))
        machine = (f / "machine.md").read_text(encoding="utf-8")
        self.assertIn("output = 1", machine)
        self.assertIn("=== examples ===", machine)
        self.assertIn("نص", (f / "science.md").read_text(encoding="utf-8"))
        self.assertFalse((self.root / "files" / (self.nid + ".md")).exists())

    def test_round_trip(self):
        n = self.store().nodes[self.nid]
        m = Node.from_faces(self.folder())
        self.assertEqual(n.dump(), m.dump())

    def test_faces_edited_by_hand(self):
        f = self.folder()
        (f / "human.md").write_text("نص جديد\n", encoding="utf-8")
        (f / "machine.md").write_text("only instructions\n",
                                      encoding="utf-8")
        (f / "science.md").unlink()
        n = self.store().nodes[self.nid]
        self.assertEqual(n.human, "نص جديد")
        self.assertEqual((n.instructions, n.code), ("only instructions", ""))
        self.assertEqual(n.reading, "")

    def test_show_by_type(self):
        out = self.cli("show", "1.1.1")[1]
        for word in ("human", "machine", "science", "مفهوم", "تعليمات",
                     "نص", self.nid):
            self.assertIn(word, out)
        out = self.cli("show", self.nid, "--type", "بشري")[1]
        self.assertIn("مفهوم", out)
        self.assertNotIn("تعليمات", out)
        out = self.cli("show", "1.1.1", "--type", "machine", "--type",
                       "علمي")[1]
        self.assertIn("output = 1", out)
        self.assertIn("نص", out)
        self.assertNotIn("مفهوم", out)
        self.assertIn("---", self.cli("show", "1.1.1", "--raw")[1])
        code, _, err = self.cli("show", "1.1.1", "--type", "other")
        self.assertNotEqual(code, 0)
        self.assertIn("human, machine or science", err)

    def test_open_one_face(self):
        script = self.root / "ed.py"
        script.write_text(
            "import sys\n"
            "for f in sys.argv[1:]:\n"
            "    open(f + '.seen', 'w').write('x')\n"
            "    open(f, 'w', encoding='utf-8').write('edited\\n')\n",
            encoding="utf-8")
        editor = '"%s" "%s"' % (sys.executable, script)
        code, out, err = self.cli("open", "1.1.1", "--type", "science",
                                  "--editor", editor)
        self.assertEqual(code, 0, err)
        n = self.store().nodes[self.nid]
        self.assertEqual(n.reading, "edited")
        self.assertEqual((n.human, n.instructions, n.code),
                         ("مفهوم", "تعليمات", "output = 1"))
        code, out, err = self.cli("open", "1.1.1", "--type", "بشري",
                                  "--editor", editor)
        self.assertEqual(code, 0, err)
        n = self.store().nodes[self.nid]
        self.assertEqual((n.human, n.code), ("edited", "output = 1"))

    def test_old_single_files_are_read_and_migrated(self):
        st = self.store()
        text = st.nodes[self.nid].dump()
        st.remove_file(self.nid)
        old = self.root / "files" / (self.nid + ".md")
        old.write_text(text, encoding="utf-8")
        self.assertEqual(self.store().nodes[self.nid].human, "مفهوم")
        self.assertIn("1 file(s) would", self.cli("migrate")[1])
        self.assertEqual(self.cli("migrate", "--yes")[0], 0)
        self.assertFalse(old.exists())
        self.assertTrue((self.folder() / "head.json").is_file())
        self.assertIn("already", self.cli("migrate")[1])
        self.assertEqual(self.store().nodes[self.nid].dump(), text)

    def test_saving_an_old_file_makes_a_folder(self):
        st = self.store()
        text = st.nodes[self.nid].dump()
        st.remove_file(self.nid)
        (self.root / "files" / (self.nid + ".md")).write_text(
            text, encoding="utf-8")
        self.cli("tag", "1.1.1", "algebra")
        self.assertTrue(self.folder().is_dir())
        self.assertFalse((self.root / "files" / (self.nid + ".md")).exists())

    def test_both_forms_is_an_error(self):
        (self.root / "files" / (self.nid + ".md")).write_text(
            self.store().nodes[self.nid].dump(), encoding="utf-8")
        code, _, err = self.cli("tree")
        self.assertNotEqual(code, 0)
        self.assertIn("both", err)

    def test_release_rollback_and_archive_keep_working(self):
        self.assertEqual(self.cli("release", "create", "r1")[0], 0)
        self.fill("1.1.1", human="changed")
        out = self.cli("release", "diff", "r1")[1]
        self.assertIn("human face", out)
        self.cli("release", "rollback", "r1", "--yes")
        self.assertEqual(self.store().nodes[self.nid].human, "مفهوم")
        self.assertEqual(self.cli("archive", "1.1.1", "--yes")[0], 0)
        self.assertFalse(self.folder().exists())
        self.assertEqual(self.cli("archive", "--restore", self.nid)[0], 0)
        self.assertTrue((self.folder() / "head.json").is_file())
        self.assertEqual(self.cli("pack", "-o", str(self.root / "p.zip"),
                                  "--branch", "1.1.1")[0], 0)
        import zipfile
        with zipfile.ZipFile(self.root / "p.zip") as z:
            self.assertIn("files/%s.md" % self.nid, z.namelist())
        code, out, err = self.cli("unpack", str(self.root / "p.zip"),
                                  "--new-ids")
        self.assertEqual(code, 0, err)

    def test_doctor_reads_folders(self):
        self.assertIn("healthy", self.cli("doctor")[1])
        (self.folder() / "head.json").write_text("{bad", encoding="utf-8")
        code, out, _ = self.cli("doctor")
        self.assertEqual(code, 1)
        self.assertIn("cannot be read", out)
        self.cli("repair")
        self.assertTrue((self.root / "quarantine" / self.nid).is_dir())


if __name__ == "__main__":
    unittest.main()
