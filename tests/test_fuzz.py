"""The fuzz targets of fuzz/targets.py, run with random and mutated inputs
from the standard library on every test run (ClusterFuzzLite runs them
coverage-guided on pull requests). A fixed seed keeps failures repeatable;
TAWJIH_FUZZ_RUNS sets how many inputs each target gets."""
import os
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fuzz"))
import targets  # noqa: E402

SEEDS = {
    "path_language": [b"1 TO 2", b"1 TO 2 IF yes\n2 TO 3",
                      b"ITERATE 3 UNTIL done\n1 TO 2", b"1 AND 2 TO 3",
                      b"1 AMONG 2 3 BY k", "1 إلى 2 إذا نعم".encode(),
                      b"EACH 1 TO APPROVE 2 RETRY 2 UNTIL NOT bad",
                      b"'1 TO 2", b"1 THROUGH 2 TO 3 # note"],
    "contract": [b"\x00{\"a\": 1, \"b\": 2}", b"\x02```json\n{\"a\": 1}\n```",
                 b"\x03plain text", b"\x81json(a) {}", b"\x04[1, 2]"],
    "examples": [b"Q: a\nA: b", b"Q: a\nA: b\nE: c\nBAD: d",
                 "س: سؤال\nج: جواب".encode(), b"A: no question"],
    "redaction": [b"mail a@b.co phone +966500000000",
                  b"card 4111 1111 1111 1111 iban SA0380000000608010167519",
                  b"ip 10.0.0.1 and 1234567890",
                  b"see [EMAIL_1] and a@b.co"],
}


def mutate(r, data):
    s = bytearray(data)
    for _ in range(r.randint(0, 8)):
        op = r.random()
        if op < 0.3 and s:
            s[r.randrange(len(s))] = r.randrange(256)
        elif op < 0.6:
            s.insert(r.randrange(len(s) + 1), r.randrange(256))
        elif op < 0.8 and s:
            del s[r.randrange(len(s))]
        else:                                # splice two seeds
            s += r.choice(SEEDS[r.choice(list(SEEDS))])
    return bytes(s)


class Fuzz(unittest.TestCase):
    RUNS = int(os.environ.get("TAWJIH_FUZZ_RUNS", "1500"))

    def run_target(self, name):
        r = random.Random(name)
        target = targets.TARGETS[name]
        for data in SEEDS[name] + [b"", b"\xff" * 40]:
            target(data)
        for _ in range(self.RUNS):
            data = mutate(r, r.choice(SEEDS[name]))
            try:
                target(data)
            except Exception as e:           # report the input that broke it
                self.fail("%s(%r): %s: %s" % (name, data,
                                               type(e).__name__, e))

    def test_path_language(self):
        self.run_target("path_language")

    def test_contract(self):
        self.run_target("contract")

    def test_examples(self):
        self.run_target("examples")

    def test_redaction(self):
        self.run_target("redaction")

    def test_placeholder_already_in_the_text(self):
        # found by writing the redaction property: a text that already held
        # "[EMAIL_1]" got the private address back in its place
        from tawjihlib.providers import Redactor
        sent, mapping = targets._REDACT.mask("see [EMAIL_1] and a@b.co")
        self.assertEqual(sent, "see [EMAIL_1] and [EMAIL_2]")
        self.assertEqual(Redactor.unmask(sent, mapping),
                         "see [EMAIL_1] and a@b.co")


if __name__ == "__main__":
    unittest.main()
