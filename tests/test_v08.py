"""Tests for 0.8: redaction, usage and stats, orphans, converters for
ingest, retries, fallback files and approval."""
import json
import os
import stat
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base, tawjih  # noqa: E402


class Redaction(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.cli("redact", "--init")
        f = self.root / "redact.txt"
        f.write_text(f.read_text(encoding="utf-8") + "Mohammed Khalid\n"
                     "re: Falcon-\\d+\n", encoding="utf-8")

    def test_mask_and_unmask(self):
        red = tawjih.Redactor.load(self.root)
        text = ("Mohammed Khalid (mohammed@example.com, +966 50 123 4567, "
                "0551234567) paid with 4111 1111 1111 1111, not 4111 1111 "
                "1111 1112; id 1012345678, IBAN SA03 8000 0000 6080 1016 "
                "7519, host 10.0.0.1, project Falcon-7. mohammed khalid again. "
                "The year 2026 and 271^321 mod 481 stay.")
        masked, mapping = red.mask(text)
        for secret in ("Mohammed Khalid", "mohammed@example.com", "123 4567",
                       "0551234567", "4111 1111 1111 1111", "1012345678",
                       "SA03", "10.0.0.1", "Falcon-7", "mohammed khalid"):
            self.assertNotIn(secret, masked)
        self.assertIn("4111 1111 1111 1112", masked)     # fails the Luhn check
        self.assertIn("2026", masked)
        self.assertIn("271^321 mod 481", masked)
        self.assertEqual(masked.count("[PRIVATE_1]"), 1)
        self.assertEqual(tawjih.Redactor.unmask(masked, mapping), text)

    def test_a_run_sends_placeholders_and_gets_the_names_back(self):
        self.fill("1.1.1", instructions="Write to the person.")

        def fake(prompt, model=""):
            self.prompts.append(prompt)
            return "Dear [PRIVATE_1], we wrote to [EMAIL_1]."
        tawjih.BACKENDS["fake"] = fake
        code, out, err = self.cli("produce", "p", "1.1.1", "--input",
                                  "Mohammed Khalid, mk@example.org",
                                  "--backend", "fake")
        self.assertEqual(code, 0, err)
        self.assertNotIn("Mohammed", self.prompts[0])
        self.assertNotIn("mk@example.org", self.prompts[0])
        self.assertEqual(out.strip(),
                         "Dear Mohammed Khalid, we wrote to mk@example.org.")
        rec = json.loads(next((self.root / "history").glob(
            "production-p-*.json")).read_text(encoding="utf-8"))
        self.assertEqual(rec["usage"]["redacted"], 2)

    def test_edit_is_redacted_too(self):
        self.fill("1.1.1", human="Ask Mohammed Khalid")
        self.cli("edit", "1.1.1", "--side", "instructions", "--ask", "x",
                 "--backend", "fake", "--yes")
        self.assertNotIn("Mohammed Khalid", self.prompts[0])

    def test_command(self):
        code, out, err = self.cli("redact", "write to a@b.co")
        self.assertEqual(out.strip(), "write to [EMAIL_1]")
        self.assertIn("[EMAIL_1] = a@b.co", err)
        self.assertIn("already exists", self.cli("redact", "--init")[2])
        (self.root / "redact.txt").write_text("@nothing\n")
        self.assertIn("unknown pattern", self.cli("redact", "x")[2])


class UsageAndStats(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.fill("1.1.1", instructions="answer", produces="text")

        def fake(prompt, model=""):
            tawjih._note_usage(1000, 200)
            return "ok"
        tawjih.BACKENDS["fake"] = fake

    def test_usage_is_recorded_and_stats_add_it_up(self):
        for _ in range(2):
            self.cli("produce", "p", "1.1.1", "--input", "q", "--backend",
                     "fake", "--model", "m1")
        rec = json.loads(sorted((self.root / "history").glob(
            "production-p-*.json"))[0].read_text(encoding="utf-8"))
        self.assertEqual((rec["usage"]["in"], rec["usage"]["out"],
                          rec["usage"]["calls"]), (1000, 200, 1))
        self.assertEqual(rec["steps"][0]["usage"]["in"], 1000)
        out = self.cli("stats")[1]
        self.assertIn("p: 2 run(s), 2 finished; 2 model call(s)", out)
        self.assertIn("tokens in 2000, out 400", out)
        self.assertIn("cost -", out)
        (self.root / "prices.json").write_text(json.dumps(
            {"m1": {"input": 3.0, "output": 15.0}}))
        self.assertIn("cost 0.0120", self.cli("stats")[1])
        self.assertIn("no runs recorded for path x",
                      self.cli("stats", "--path", "x")[1])

    def test_orphan_machine_sides_in_status(self):
        out = self.cli("status")[1]
        self.assertIn("machine side that no saved path uses: 1", out)
        self.cli("produce", "p", "1.1.1", "--plan")
        self.assertNotIn("no saved path uses", self.cli("status")[1])


class RetriesFallbackApproval(Base):
    def setUp(self):
        super().setUp()
        self.tree()
        self.cli("new", "spare", "--parent", "2")
        st = self.store()
        self.a, self.spare = st.resolve("1.1.1"), st.resolve("2.1")
        self.fill("1.1.1", instructions="give JSON", produces="json(n)")

    def test_retries_ask_again_with_the_reason(self):
        self.cli("contract", "1.1.1", "--retries", "2")
        self.replies = ["no json here", '{"n": 1}']
        code, out, err = self.cli("produce", "p", "1.1.1", "--input", "q",
                                  "--backend", "fake")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out), {"n": 1})
        self.assertIn("previous answer was rejected", self.prompts[1])
        self.assertIn("asking again (1 of 2)", err)
        rec = json.loads(next((self.root / "history").glob(
            "production-p-*.json")).read_text(encoding="utf-8"))
        self.assertEqual(rec["usage"]["retries"], 1)
        # the good answer is kept under the original prompt
        code, out, err = self.cli("produce", "p", "--input", "q", "--backend",
                                  "fake", "--resume")
        self.assertIn("from the cache", err)
        # --retries 0 overrides the file's line
        self.replies = ["still no json"]
        code, _, err = self.cli("produce", "p", "--input", "q2", "--backend",
                                "fake", "--retries", "0")
        self.assertEqual(code, 1)
        self.assertIn("failed in contract", err)

    def test_fallback_runs_in_place_of_a_failing_step(self):
        self.fill("2.1", code='output = \'{"n": 2}\'', produces="json(n)")
        self.assertIn("own fallback", self.cli("contract", "1.1.1",
                                               "--fallback", "1.1.1")[2])
        self.assertIn("no machine side", self.cli("contract", "1.1.1",
                                                  "--fallback", "1")[2])
        self.cli("contract", "1.1.1", "--fallback", "2.1")
        self.replies = ["not json"]
        code, out, err = self.cli("produce", "p", "1.1.1", "--input", "q",
                                  "--backend", "fake", "--allow-code")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out), {"n": 2})
        self.assertIn("falling back to spare", err)
        self.assertEqual(self.cli("check")[0], 0)
        self.fill("2.1", produces="text")
        self.assertIn("fallback %s produces text" % self.spare,
                      self.cli("check")[2] + self.cli("check")[1])

    def test_approval(self):
        self.cli("contract", "1.1.1", "--approve", "after")
        self.replies = ['{"n": 1}']
        code, _, err = self.cli("produce", "p", "1.1.1", "--input", "q",
                                "--backend", "fake")
        self.assertEqual(code, 1)
        self.assertIn("not approved", err)
        self.assertIn("--approve auto", err)
        self.replies = ['{"n": 1}']
        code, out, err = self.cli("produce", "p", "--input", "q", "--backend",
                                  "fake", "--approve", "ask", stdin="y\n")
        self.assertEqual(code, 0, err)
        self.replies = ['{"n": 1}']
        code, out, err = self.cli("produce", "p", "--input", "q", "--backend",
                                  "fake", "--approve", "ask",
                                  stdin='e\n{"n": 9}\n.\n')
        self.assertEqual(json.loads(out), {"n": 9})
        self.replies = ['{"n": 1}']
        code, out, err = self.cli("produce", "p", "--input", "q", "--backend",
                                  "fake", "--approve", "ask",
                                  stdin='e\nnot json\n.\n')
        self.assertEqual(code, 1)
        self.replies = ['{"n": 1}']
        self.assertEqual(self.cli("produce", "p", "--input", "q", "--backend",
                                  "fake", "--approve", "auto")[0], 0)
        self.cli("contract", "1.1.1", "--approve", "before")
        code, _, err = self.cli("produce", "p", "--input", "q", "--backend",
                                "fake", "--approve", "ask", stdin="n\n")
        self.assertIn("not approved before the step ran", err)
        self.assertEqual(len(self.prompts), 5)      # the model was not asked
        self.assertIn("approve is before", self.cli("contract", "1.1.1",
                                                    "--approve", "later")[2])


@unittest.skipIf(os.name == "nt", "the stand-in converters are shell scripts")
class Converters(Base):
    def setUp(self):
        super().setUp()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.old_path = os.environ["PATH"]
        os.environ["PATH"] = str(self.bin)

    def tearDown(self):
        os.environ["PATH"] = self.old_path
        super().tearDown()

    def tool(self, name, body):
        f = self.bin / name
        f.write_text("#!%s\nimport sys, pathlib\nargs = sys.argv[1:]\n%s\n"
                     % (sys.executable, body))
        f.chmod(f.stat().st_mode | stat.S_IEXEC)

    def test_auto_picks_the_best_installed(self):
        pdf = self.root / "b.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        self.tool("pdftotext", "pathlib.Path(args[-1]).write_text('plain')")
        code, out, err = self.cli("ingest", str(pdf))
        self.assertIn("with pdftotext", out)
        self.assertIn("damages formulas", out)
        self.tool("docling", "d = pathlib.Path(args[args.index('--output')+1])\n"
                  "(d / 'b.md').write_text('# Title\\n\\n$x^2$')")
        code, out, err = self.cli("ingest", str(pdf), "-o",
                                  str(self.root / "d.md"))
        self.assertEqual(code, 0, err)
        self.assertIn("with docling", out)
        self.assertEqual((self.root / "d.md").read_text(), "# Title\n\n$x^2$\n")
        self.assertIn("does not read .pdf", self.cli("ingest", str(pdf),
                                                     "--with", "pandoc")[2])

    def test_pandoc_for_word_and_a_failing_tool(self):
        import zipfile
        doc = self.root / "w.docx"
        with zipfile.ZipFile(doc, "w") as z:
            z.writestr("word/document.xml", '<w:document xmlns:w="x" '
                       'xmlns:m="y"><w:body><w:p><m:oMath/></w:p></w:body>'
                       '</w:document>')
        self.tool("pandoc", "pathlib.Path(args[args.index('-o')+1])"
                  ".write_text('# From pandoc\\n\\n$a^{-1}$')")
        code, out, err = self.cli("ingest", str(doc))
        self.assertEqual(code, 0, err)
        self.assertIn("with pandoc", out)
        self.tool("pandoc", "sys.stderr.write('bad file'); sys.exit(2)")
        code, _, err = self.cli("ingest", str(doc))
        self.assertIn("pandoc failed: bad file", err)


if __name__ == "__main__":
    unittest.main()
