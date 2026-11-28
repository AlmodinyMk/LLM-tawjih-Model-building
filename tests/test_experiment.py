"""The path-or-direct measurement, end to end against a stand-in Ollama."""
import json
import re
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                       "experiments" / "path-vs-direct"))
import run  # noqa: E402


def coefficients(eq):
    """a, b, c of 'LHS = RHS' as the experiment writes equations."""
    def side(t):
        a = b = c = 0
        for sign, num, var in re.findall(r"([+−-]?)\s*(\d*)\s*(x²|x)?",
                                         t.replace(" ", "")):
            if not num and not var:
                continue
            v = int(num or 1) * (-1 if sign in "−-" and sign else 1)
            if var == "x²":
                a += v
            elif var == "x":
                b += v
            else:
                c += v
        return a, b, c
    left, right = eq.split("=")
    (a1, b1, c1), (a2, b2, c2) = side(left), side(right)
    return a1 - a2, b1 - b2, c1 - c2


def equation(text):
    """The 'LHS = RHS' around the first '=' of a text, read without a
    regular expression."""
    allowed = set("0123456789x²+−- ")
    i = text.index("=")
    lo, hi = i, i + 1
    while lo > 0 and text[lo - 1] in allowed:
        lo -= 1
    while hi < len(text) and text[hi] in allowed:
        hi += 1
    return text[lo:hi].strip()


class Ollama(BaseHTTPRequestHandler):
    """Extracts a, b, c correctly, explains from the computed roots, and
    answers a direct question wrongly: the path should win every case."""
    def log_message(self, *args):
        pass

    def reply(self, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.reply({"models": [{"name": "stand-in:1b", "digest": "abc",
                                "size": 1}]})

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        prompt = json.loads(self.rfile.read(n))["messages"][0]["content"]
        if '"roots"' in prompt:                  # the explanation step
            obj, _ = json.JSONDecoder().raw_decode(prompt,
                                                   prompt.index('{"a"'))
            text = " و ".join("x = %d" % round(complex(r).real)
                              for r in obj["roots"])
        elif '{"a"' in prompt:                   # the extraction step
            a, b, c = coefficients(equation(prompt.split("# Input")[-1]))
            text = json.dumps({"a": a, "b": b, "c": c})
        else:                                    # asked directly
            text = "x = 99"
        self.reply({"message": {"content": text}, "model": "stand-in:1b",
                    "prompt_eval_count": 10, "eval_count": 5})


class Experiment(unittest.TestCase):
    def test_cases_are_right(self):
        cases = run.make_cases(80)
        self.assertEqual(len(cases), 80)
        self.assertEqual(sum(c["source"] == "example" for c in cases), 20)
        for c in cases[20:]:
            eq = re.search(r"(?:المعادلة|Solve|of)\s+(.+?)(?:\s+واذكر| and|\.$)",
                           c["input"]).group(1)
            a, b, cc = coefficients(eq)
            for r in c["roots"]:
                self.assertEqual(a * r * r + b * r + cc, 0, c["input"])
            said = " ".join("x = %s" % (("−%d" % -r) if r < 0 else r)
                            for r in c["roots"])
            self.assertRegex(said, c["expect"])
            self.assertNotRegex("x = 999", c["expect"])
        steps = run.arm_cases(cases[:2], "steps")
        self.assertIn("x = ", steps[0]["input"])
        self.assertNotEqual(steps[0]["input"], cases[0]["input"])

    def test_statistics(self):
        self.assertEqual(run.mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(run.mcnemar(10, 0), 2 / 1024)
        self.assertAlmostEqual(run.mcnemar(3, 3), 1.0)
        lo, hi = run.wilson(50, 100)
        self.assertTrue(0.40 < lo < 0.41 and 0.59 < hi < 0.60)
        lo, hi = run.bootstrap([1.0] * 10)
        self.assertEqual((lo, hi), (1.0, 1.0))

    def test_end_to_end(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), Ollama)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        out = Path(tempfile.mkdtemp(prefix="tawjih-exp-"))
        self.addCleanup(shutil.rmtree, out, True)
        code = run.main(["--models", "stand-in:1b", "--cases", "24",
                         "--repeat", "2", "--host", "http://127.0.0.1:%d" %
                         srv.server_address[1], "--out", str(out)])
        self.assertEqual(code, 0)
        res = json.loads((out / "results.json").read_text(encoding="utf-8"))
        rows = res["results"]["stand-in:1b"]
        self.assertEqual(set(rows), {"path", "direct", "steps"})
        self.assertTrue(all(r["passes"] == 2 for r in rows["path"].values()),
                        {k: v["why"] for k, v in rows["path"].items()})
        self.assertTrue(all(r["passes"] == 0 for r in rows["direct"].values()))
        text = (out / "summary.md").read_text(encoding="utf-8")
        self.assertIn("| path | 48 / 48 (100%)", text)
        self.assertIn("path − direct: +100 points", text)
        self.assertIn("right only by the path in 24 cases", text)
        self.assertEqual(res["environment"]["models"]["stand-in:1b"]
                         ["digest"], "abc")


if __name__ == "__main__":
    unittest.main()
