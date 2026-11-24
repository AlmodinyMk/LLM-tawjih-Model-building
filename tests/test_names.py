"""Names from the command line or a client never reach another folder, and
reading a fenced answer stays linear (findings of CodeQL)."""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base, tawjih  # noqa: E402
from tawjihlib import core, engine  # noqa: E402


def quiet(test):
    """die() writes its message; these tests only want the refusal."""
    import contextlib
    import functools
    import io

    @functools.wraps(test)
    def run(self):
        with contextlib.redirect_stderr(io.StringIO()):
            test(self)
    return run


class Names(Base):
    BAD = ("../x", "..", "a/b", "/etc/passwd", "a\\b", "c:x", "", "a b")

    @quiet
    def test_path_file(self):
        self.assertEqual(core.path_file(self.root, "ok_1-2", ".path"),
                         (self.root / "ok_1-2.path").absolute())
        for name in self.BAD:
            with self.assertRaises(tawjih.Die, msg=name):
                core.path_file(self.root, name, ".path")

    @quiet
    def test_inside(self):
        with self.assertRaises(tawjih.Die):
            core.inside(self.root, "../outside.json")
        with self.assertRaises(tawjih.Die):
            engine._stamped(self.root / "runs", "../escaped", ".json")
        f = engine._stamped(self.root / "runs", "eval-p", ".json")
        self.assertEqual(f.parent, (self.root / "runs").absolute())

    def test_commands_refuse_names_outside(self):
        self.tree()
        outside = self.root.parent / ("x-%d" % id(self))
        outside.mkdir()
        self.addCleanup(lambda: [p.unlink() for p in outside.iterdir()]
                        and outside.rmdir())
        (outside / "e.cases.jsonl").write_text(
            '{"input": "q", "expect": "dry"}\n', encoding="utf-8")
        rel = "../" + outside.name + "/e"
        for argv in (["eval", rel, "--direct", "--backend", "fake"],
                     ["eval", rel, "--backend", "fake"],
                     ["run", rel, "--input", "q", "--backend", "fake"],
                     ["produce", rel, "1", "--input", "q", "--backend",
                      "fake"]):
            code, out, err = self.cli(*argv)
            self.assertEqual(code, 1, argv)
            self.assertIn("letters, digits", err + out, argv)
        self.assertEqual(sorted(p.name for p in outside.iterdir()),
                         ["e.cases.jsonl"])
        self.assertFalse(list(self.root.glob("runs/eval-*")))


class Fence(unittest.TestCase):
    def test_same_answers(self):
        cases = {"```json\n{\"a\": 1}\n```": "{\"a\": 1}\n",
                 "text ```\n{}\n``` more": "{}\n",
                 "```json {\"a\": 1}": None, "```\nopen": None,
                 "no fence": None}
        for text, want in cases.items():
            self.assertEqual(core._fenced(text), want, text)
        obj, _ = core._json_in("Here:\n```json\n{\"k\": 2}\n```\nDone.")
        self.assertEqual(obj, {"k": 2})

    def test_linear(self):
        hostile = "```a" * 200000          # quadratic for the old pattern
        t0 = time.monotonic()
        core._json_in(hostile)
        self.assertLess(time.monotonic() - t0, 2.0)


if __name__ == "__main__":
    unittest.main()
