"""Tests for whoami: the person's identity in their own settings, and the
records that note it (runs, approvals, reviews, evaluations, jobs)."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base, tawjih  # noqa: E402
from tawjihlib import core  # noqa: E402


class Identity(Base):
    def setUp(self):
        super().setUp()
        self.idfile = self.root / "settings" / "identity.json"
        self.env = mock.patch.dict(os.environ,
                                   {"TAWJIH_IDENTITY": str(self.idfile)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        super().tearDown()

    def last_record(self):
        recs = sorted((self.root / "history").glob("production-*.json"),
                      key=lambda f: f.stat().st_mtime_ns)
        return json.loads(recs[-1].read_text(encoding="utf-8"))

    def test_set_show_and_clear(self):
        code, _, err = self.cli("whoami", "set")
        self.assertEqual(code, 1)
        self.assertIn("--name", err)
        code, out, err = self.cli("whoami", "set", "--name", "محمد",
                                  "--email", "m@example.com")
        self.assertEqual(code, 0, err)
        me = json.loads(self.idfile.read_text(encoding="utf-8"))
        self.assertRegex(me["id"], r"^u-[0-9a-f]{8}$")
        self.assertEqual((me["name"], me["email"], me["record"]),
                         ("محمد", "m@example.com", "name"))
        # the email stays in the settings; records get the id and name
        self.assertEqual(core.who(), {"id": me["id"], "name": "محمد"})
        out = self.cli("هويتي", "--json")[1]
        self.assertEqual(json.loads(out)["records"]["name"], "محمد")
        # the id is kept when other fields change
        self.cli("whoami", "set", "--name", "Mohammed")
        self.assertEqual(json.loads(self.idfile.read_text(
            encoding="utf-8"))["id"], me["id"])
        self.cli("whoami", "clear")
        self.assertFalse(self.idfile.exists())
        # the project holds nothing of it
        self.assertFalse(list(self.root.glob("**/identity.json")))

    def test_checks(self):
        code, _, err = self.cli("whoami", "set", "--name", "x", "--email",
                                "not-an-email")
        self.assertEqual(code, 1)
        self.assertIn("email", err)
        code, _, err = self.cli("whoami", "set", "--name", "x", "--id",
                                "a b")
        self.assertEqual(code, 1)
        self.assertEqual(self.cli("whoami", "set", "--name", "x", "--id",
                                  "0000-0002-1825-0097")[0], 0)
        self.assertEqual(core.who()["id"], "0000-0002-1825-0097")

    def test_record_setting(self):
        self.cli("whoami", "set", "--name", "N", "--email", "n@example.com",
                 "--id", "u-1")
        cases = {"name": {"id": "u-1", "name": "N"},
                 "id": {"id": "u-1"},
                 "email": {"id": "u-1", "name": "N",
                           "email": "n@example.com"},
                 "none": None}
        for mode, want in cases.items():
            self.cli("whoami", "set", "--record", mode)
            self.assertEqual(core.who(), want, mode)
        self.assertEqual(core.who_label({"id": "u-1", "name": "N"}),
                         "N (u-1)")

    def test_first_use_makes_a_permanent_id(self):
        import contextlib
        import io
        self.assertFalse(self.idfile.exists())
        err = io.StringIO()
        with mock.patch.object(core, "_git_name", return_value="Git Name"), \
                contextlib.redirect_stderr(err):
            by = core.who()
        self.assertRegex(by["id"], r"^u-[0-9a-f]{8}$")
        self.assertEqual(by["name"], "Git Name")
        self.assertIn("your id is " + by["id"], err.getvalue())
        me = json.loads(self.idfile.read_text(encoding="utf-8"))
        self.assertEqual((me["from"], me["record"]), ("git", "name"))
        self.assertNotIn("email", me)
        # the same id from then on, said only once
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(core.who()["id"], by["id"])
        self.assertEqual(err.getvalue(), "")
        if os.name != "nt":
            self.assertEqual(self.idfile.stat().st_mode & 0o777, 0o600)
        # whoami shows it
        out = self.cli("whoami")[1]
        self.assertIn(by["id"], out)
        self.assertIn("made", out)

    def test_first_use_without_git(self):
        import contextlib
        import io
        with mock.patch.object(core, "_git_name", return_value=""), \
                contextlib.redirect_stderr(io.StringIO()):
            by = core.who()
        me = json.loads(self.idfile.read_text(encoding="utf-8"))
        self.assertEqual(me["from"], "system")
        self.assertIn("id", by)

    def test_cannot_keep_an_identity(self):
        self.env.stop()
        bad = self.root / "afile"
        bad.write_text("x")
        with mock.patch.dict(os.environ, {"TAWJIH_IDENTITY": str(
                bad / "identity.json")}), \
                mock.patch.object(core, "_git_name", return_value="G"):
            self.assertEqual(core.who(), {"name": "G", "from": "git"})
        self.env.start()

    def test_opt_out(self):
        self.assertEqual(self.cli("whoami", "set", "--record", "none")[0], 0)
        self.assertIsNone(core.who())

    def test_setup_questions(self):
        import io
        answers = iter(["", "not-an-email", "m@example.com", "bad id",
                        "0000-0002-1825-0097", "maybe", "email"])
        out = io.StringIO()
        from tawjihlib import governance
        with mock.patch.object(governance, "_git_name",
                               return_value="Git Name"):
            me = tawjih.setup_identity(ask=lambda q: next(answers), out=out)
        self.assertEqual((me["name"], me["email"], me["id"], me["record"]),
                         ("Git Name", "m@example.com", "0000-0002-1825-0097",
                          "email"))
        self.assertIn("does not look like an email", out.getvalue())
        self.assertIn("an id is letters", out.getvalue())
        self.assertIn("one of: name, id, email, none", out.getvalue())
        saved = json.loads(self.idfile.read_text(encoding="utf-8"))
        self.assertEqual(saved["id"], "0000-0002-1825-0097")
        # a name is required when git has none
        answers = iter(["", "Someone", "", "", ""])
        self.idfile.unlink()
        with mock.patch.object(governance, "_git_name", return_value=""):
            me = tawjih.setup_identity(ask=lambda q: next(answers),
                                       out=io.StringIO())
        self.assertEqual(me["name"], "Someone")
        self.assertRegex(me["id"], r"^u-[0-9a-f]{8}$")

    def test_first_use_asks_on_a_terminal_only(self):
        from tawjihlib import cli
        self.tree()
        asked = []

        def fake_setup():
            asked.append(1)
            return {}
        with mock.patch.object(cli, "setup_identity", fake_setup):
            ns = tawjih.build_parser().parse_args(["-p", self.dir, "trees"])
            with mock.patch.object(cli, "_interactive", return_value=False):
                cli.first_use(ns)
            self.assertEqual(asked, [])           # a script is not asked
            with mock.patch.object(cli, "_interactive", return_value=True):
                cli.first_use(ns)
            self.assertEqual(asked, [1])          # a person on a terminal is
            for argv in (["whoami"], ["mcp"]):    # never these
                ns = tawjih.build_parser().parse_args(["-p", self.dir] + argv)
                with mock.patch.object(cli, "_interactive",
                                       return_value=True):
                    cli.first_use(ns)
            self.assertEqual(asked, [1])
            self.cli("whoami", "set", "--name", "N")
            ns = tawjih.build_parser().parse_args(["-p", self.dir, "trees"])
            with mock.patch.object(cli, "_interactive", return_value=True):
                cli.first_use(ns)
            self.assertEqual(asked, [1])          # only before the first use

    def test_runs_and_approvals_note_who(self):
        self.cli("whoami", "set", "--name", "Reviewer", "--id", "u-r")
        self.tree()
        self.fill("1.1.1", instructions="answer")
        self.cli("contract", "1.1.1", "--approve", "after")
        self.replies = ['{"n": 1}']
        code, _, err = self.cli("produce", "p", "1.1.1", "--input", "q",
                                "--backend", "fake", "--approve", "ask",
                                stdin="y\n")
        self.assertEqual(code, 0, err)
        rec = self.last_record()
        self.assertEqual(rec["by"], {"id": "u-r", "name": "Reviewer"})
        step = rec["steps"][0]
        self.assertEqual(len(step["approvals"]), 1)
        a = step["approvals"][0]
        self.assertEqual((a["when"], a["mode"], a["approved"]),
                         ("after", "ask", True))
        self.assertEqual(a["by"], {"id": "u-r", "name": "Reviewer"})
        # --approve auto: noted as automatic, with no person
        self.replies = ['{"n": 1}']
        self.cli("produce", "p", "--input", "q", "--backend", "fake",
                 "--approve", "auto")
        a = self.last_record()["steps"][0]["approvals"][0]
        self.assertEqual(a["mode"], "auto")
        self.assertNotIn("by", a)
        # a refusal is noted too
        self.replies = ['{"n": 1}']
        self.cli("produce", "p", "--input", "q", "--backend", "fake",
                 "--approve", "ask", stdin="n\n")
        a = self.last_record()["steps"][0]["approvals"][0]
        self.assertFalse(a["approved"])

    def test_review_notes_who(self):
        self.cli("whoami", "set", "--name", "Reviewer", "--id", "u-r")
        self.tree()
        code, out, err = self.cli("review", "1.1.1", "--ok")
        self.assertEqual(code, 0, err)
        self.assertIn("by Reviewer (u-r)", out)
        st = self.store()
        n = st.nodes[st.resolve("1.1.1")]
        self.assertEqual(n.reviewed_by, "Reviewer (u-r)")
        head = json.loads((self.root / "files" / n.id / "head.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(head["reviewed_by"], "Reviewer (u-r)")
        self.fill("1.1.1", instructions="changed")
        self.assertIn("by Reviewer (u-r)", self.cli("review", "1.1.1")[1])

    def test_record_none_keeps_names_out(self):
        self.cli("whoami", "set", "--name", "Secret", "--record", "none")
        self.tree()
        self.fill("1.1.1", instructions="answer")
        self.cli("produce", "p", "1.1.1", "--input", "q", "--backend",
                 "fake")
        self.assertIsNone(self.last_record()["by"])
        text = "".join(f.read_text(encoding="utf-8") for f in
                       (self.root / "history").glob("*.json"))
        self.assertNotIn("Secret", text)

    def test_trace_user_is_never_an_email(self):
        from tawjihlib.tracing import Tracer
        rec = {"by": {"id": "u-1", "name": "N", "email": "n@example.com"}}
        t = Tracer(self.root, "p", [], rec)
        self.assertEqual(t.common()["langfuse.user.id"], "u-1")
        t = Tracer(self.root, "p", [], {"by": None})
        self.assertNotIn("langfuse.user.id", t.common())


if __name__ == "__main__":
    unittest.main()
