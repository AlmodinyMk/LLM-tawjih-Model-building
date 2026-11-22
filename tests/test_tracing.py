"""Tests for traces of runs (run --trace, tawjih observe): OTLP/JSON as
Langfuse takes it, sent to a local stand-in server."""
import base64
import http.server
import json
import os
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tawjih import Base  # noqa: E402

ENV = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST",
       "LANGFUSE_BASE_URL", "TAWJIH_OTLP_ENDPOINT", "TAWJIH_OTLP_HEADERS",
       "TAWJIH_TRACE", "TAWJIH_TRACE_CONTENT")


class Server:
    def __init__(self):
        self.got, self.status = [], 200
        me = self

        class H(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                me.got.append((self.path, dict(self.headers),
                               json.loads(body)))
                self.send_response(me.status)
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *a):
                pass
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), H)
        self.url = "http://127.0.0.1:%d" % self.httpd.server_port
        threading.Thread(target=self.httpd.serve_forever,
                         daemon=True).start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def spans(body):
    return body["resourceSpans"][0]["scopeSpans"][0]["spans"]


def attrs(span):
    out = {}
    for a in span["attributes"]:
        v = a["value"]
        if "arrayValue" in v:
            out[a["key"]] = [x["stringValue"] for x in v["arrayValue"]
                             ["values"]]
        else:
            out[a["key"]] = list(v.values())[0]
    return out


def of_type(body, kind):
    return [s for s in spans(body)
            if attrs(s).get("langfuse.observation.type") == kind]


class Tracing(Base):
    def setUp(self):
        super().setUp()
        saved = {k: os.environ.pop(k, None) for k in ENV}

        def restore():
            for k in ENV:
                os.environ.pop(k, None)
            os.environ.update({k: v for k, v in saved.items() if v})
        self.addCleanup(restore)
        self.srv = Server()
        self.addCleanup(self.srv.stop)
        os.environ.update(LANGFUSE_PUBLIC_KEY="pk-lf-1",
                          LANGFUSE_SECRET_KEY="sk-lf-2",
                          LANGFUSE_HOST=self.srv.url)
        self.cli("new", "الجبر")
        self.cli("new", "Solve", "--parent", "1")
        self.fill("1", instructions="first")
        self.fill("1.1", instructions="second")
        self.cli("path", "p", "1 TO 1.1")
        (self.root / "redact.txt").write_text("@email\n", encoding="utf-8")

    def run_p(self, *extra, text="x from ali@example.com"):
        return self.cli("run", "p", "--input", text, "--backend", "fake",
                        "--no-cache", *extra)

    def test_a_run_is_one_trace(self):
        self.replies = ["one", "two"]
        code, out, err = self.run_p("--trace")
        self.assertEqual(code, 0, err)
        self.assertIn("trace: sent to langfuse", err)
        self.assertEqual(len(self.srv.got), 1)
        path, headers, body = self.srv.got[0]
        self.assertEqual(path, "/api/public/otel/v1/traces")
        h = {k.lower(): v for k, v in headers.items()}
        self.assertEqual(h["authorization"], "Basic " + base64.b64encode(
            b"pk-lf-1:sk-lf-2").decode())
        self.assertEqual(h["x-langfuse-ingestion-version"], "4")
        ss = spans(body)
        self.assertEqual(len({s["traceId"] for s in ss}), 1)
        self.assertTrue(all(len(s["traceId"]) == 32 and len(s["spanId"]) ==
                            16 for s in ss))
        (root,) = of_type(body, "chain")
        steps = of_type(body, "span")
        gens = of_type(body, "generation")
        self.assertEqual((len(steps), len(gens)), (2, 2))
        self.assertTrue(all(s["parentSpanId"] == root["spanId"]
                            for s in steps))
        self.assertEqual({g["parentSpanId"] for g in gens},
                         {s["spanId"] for s in steps})
        ra = attrs(root)
        self.assertEqual(ra["langfuse.trace.name"], "path p")
        self.assertIn("الجبر", ra["langfuse.trace.tags"])
        self.assertEqual(ra["langfuse.observation.output"], "two")
        ga = attrs(gens[0])
        self.assertIn("# Instructions", ga["langfuse.observation.input"])
        self.assertEqual(ga["langfuse.observation.output"], "one")
        for s in ss:
            self.assertLessEqual(int(s["startTimeUnixNano"]),
                                 int(s["endTimeUnixNano"]))
        blob = json.dumps(body)
        self.assertNotIn("ali@example.com", blob)
        self.assertIn("[EMAIL_1]", blob)

    def test_nothing_without_asking(self):
        self.replies = ["one", "two"]
        self.assertEqual(self.run_p()[0], 0)
        self.assertEqual(self.srv.got, [])
        os.environ["TAWJIH_TRACE"] = "1"
        self.replies = ["one", "two"]
        self.assertEqual(self.run_p()[0], 0)
        self.assertEqual(len(self.srv.got), 1)

    def test_a_failed_step_is_an_error(self):
        self.replies = ["one", ""]
        code, _, err = self.run_p("--trace")
        self.assertNotEqual(code, 0)
        self.assertIn("trace: sent", err)
        body = self.srv.got[0][2]
        (root,) = of_type(body, "chain")
        self.assertEqual(attrs(root)["langfuse.observation.level"], "ERROR")
        self.assertEqual(root["status"]["code"], 2)
        bad = [s for s in of_type(body, "span")
               if attrs(s)["langfuse.observation.metadata.status"] ==
               "failed"]
        self.assertEqual(len(bad), 1)

    def test_no_content(self):
        self.replies = ["one", "two"]
        self.assertEqual(self.run_p("--trace", "--trace-no-content")[0], 0)
        blob = json.dumps(self.srv.got[0][2])
        for word in ("one", "two", "Instructions", "EMAIL"):
            self.assertNotIn('"%s' % word, blob)
        self.assertIn("path p", blob)

    def test_kept_when_the_server_fails_then_flushed(self):
        self.srv.status = 503
        self.replies = ["one", "two"]
        code, _, err = self.run_p("--trace")
        self.assertEqual(code, 0, err)
        self.assertIn("kept in", err)
        self.assertIn("1 trace(s) waiting", self.cli("observe")[1])
        self.srv.status = 200
        code, out, _ = self.cli("observe", "flush")
        self.assertEqual(code, 0)
        self.assertIn("1 of 1", out)
        self.assertFalse(list((self.root / "history" /
                               "traces-pending").glob("*")))

    def test_policy_and_missing_settings(self):
        (self.root / "tawjih.policy.json").write_text(
            '{"allow_network": "local"}')
        os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"
        code, _, err = self.run_p("--trace")
        self.assertNotEqual(code, 0)
        self.assertIn("traces would go to", err)
        for k in ENV:
            os.environ.pop(k, None)
        code, _, err = self.run_p("--trace")
        self.assertIn("LANGFUSE_PUBLIC_KEY", err)
        self.assertIn("go nowhere", self.cli("langfuse", "status")[1])

    def test_any_otlp_collector_and_check(self):
        os.environ["TAWJIH_OTLP_ENDPOINT"] = self.srv.url + "/v1/traces"
        os.environ["TAWJIH_OTLP_HEADERS"] = "x-team=a, x-key = b"
        code, out, err = self.cli("observe", "check")
        self.assertEqual(code, 0, err)
        path, headers, _ = self.srv.got[0]
        h = {k.lower(): v for k, v in headers.items()}
        self.assertEqual(path, "/v1/traces")
        self.assertEqual((h["x-team"], h["x-key"]), ("a", "b"))

    def test_iterate_rounds_share_a_session(self):
        self.cli("path", "q", "1, ITERATE 2 UNTIL ok")
        self.replies = ["no", "no"]
        code, _, err = self.cli("run", "q", "--input", "x", "--backend",
                                "fake", "--no-cache", "--trace")
        sessions = {attrs(of_type(b, "chain")[0]).get("langfuse.session.id")
                    for _, _, b in self.srv.got}
        self.assertEqual(len(self.srv.got), 2)
        self.assertEqual(len(sessions), 1)
        self.assertTrue(sessions.pop().startswith("tawjih-q-"))


if __name__ == "__main__":
    unittest.main()
