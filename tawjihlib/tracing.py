"""Traces of runs for Langfuse, or for any OpenTelemetry collector: OTLP
over HTTP with JSON, standard library only.

A run of a path is one trace: the path is the root observation, each step a
span under it, and each model call a generation under its step, with the
model, the prompt as it was sent (after redaction), the answer, the tokens
and the times. Nothing is sent unless asked (run --trace, or TAWJIH_TRACE),
the project's policy is obeyed, and with TAWJIH_TRACE_CONTENT=0 (or
--trace-no-content) only names, times, models and counts are sent. A trace
that cannot be sent is kept in history/traces-pending/ for `observe flush`.

Where traces go, from the environment (never from the project):
  LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST (or
  LANGFUSE_BASE_URL; default https://cloud.langfuse.com): Langfuse, at
  /api/public/otel/v1/traces with x-langfuse-ingestion-version: 4.
  TAWJIH_OTLP_ENDPOINT (the full traces URL) and TAWJIH_OTLP_HEADERS
  ("name=value,name=value"): any other OTLP/HTTP collector.

Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
import base64
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from .core import __version__, die
from .providers import POLICY_FILE, Redactor, _is_local, load_policy

LANGFUSE_CLOUD = "https://cloud.langfuse.com"
OTEL_PATH = "/api/public/otel/v1/traces"
PENDING = "traces-pending"
MAX_TEXT = 100000          # characters of one input or output


def sink():
    """(kind, url, headers) of where traces go, or None."""
    url = os.environ.get("TAWJIH_OTLP_ENDPOINT", "").strip()
    if url:
        headers = {}
        for part in os.environ.get("TAWJIH_OTLP_HEADERS", "").split(","):
            k, eq, v = part.partition("=")
            if eq and k.strip():
                headers[k.strip()] = v.strip()
        return "otlp", url, headers
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    sk = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    if pk and sk:
        host = (os.environ.get("LANGFUSE_HOST") or os.environ.get(
            "LANGFUSE_BASE_URL") or LANGFUSE_CLOUD).strip().rstrip("/")
        auth = base64.b64encode(("%s:%s" % (pk, sk)).encode()).decode()
        return "langfuse", host + OTEL_PATH, {
            "Authorization": "Basic " + auth,
            "x-langfuse-ingestion-version": "4",
            "x-langfuse-public-key": pk}
    return None


def wanted(flag=False):
    return bool(flag) or os.environ.get("TAWJIH_TRACE", "").lower() in (
        "1", "yes", "true", "on", "langfuse", "otlp")


def check_allowed(root, url):
    """The project's policy decides whether traces may leave this
    computer."""
    allow = load_policy(root)["allow_network"]
    if allow is True or _is_local(url):
        return
    die("the project's policy (%s) %s; traces would go to %s" % (
        POLICY_FILE, "allows no network" if allow is False else
        "allows only this computer", url))


# -------------------------------------------------------------- OTLP/JSON
def _value(v):
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [_value(x) for x in v]}}
    return {"stringValue": str(v)}


def _attrs(d):
    return [{"key": k, "value": _value(v)} for k, v in d.items()
            if v is not None and v != "" and v != []]


def _nanos(t):
    return str(int(t * 1e9))


def _hex(n):
    return uuid.uuid4().hex[:n]


def _text(s):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    if len(s) > MAX_TEXT:
        s = s[:MAX_TEXT] + "\n[... %d more characters]" % (len(s) - MAX_TEXT)
    return s                    # as Langfuse's own SDK sends a string


class Tracer:
    """Collects a run while it goes; finish() sends it as one trace."""

    def __init__(self, root, name, plan, record, content=True, session=""):
        self.root = Path(root)
        self.name = name
        self.content = content and os.environ.get(
            "TAWJIH_TRACE_CONTENT", "1") not in ("0", "no", "false", "off")
        self.session = session
        self.trace_id = _hex(32)
        self.root_id = _hex(16)
        self.step_ids = {p["step"]: _hex(16) for p in plan}
        self.calls = []
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.red = Redactor.load(root)
        self.record = record

    def _mask(self, text):
        if not self.content:
            return None
        if self.red and text:
            text = self.red.mask(text)[0]
        return _text(text or "")

    def generation(self, step, model, usage, t_start, t_end, error=None,
                   attempt=1):
        """A model call of a step: usage is what call_model returned (its
        'sent' and 'answer' are the texts as they crossed the network)."""
        with self.lock:
            self.calls.append({"step": step, "model": model or "",
                               "usage": dict(usage or {}), "start": t_start,
                               "end": t_end, "error": error,
                               "attempt": attempt})

    def common(self):
        """The attributes Langfuse wants on every span."""
        rec = self.record
        tags = ["tawjih"] + [t.get("title", "") for t in rec.get("trees",
                                                                  [])]
        out = {"langfuse.trace.name": "path " + self.name,
               "langfuse.trace.tags": [t for t in tags if t],
               "langfuse.session.id": self.session,
               "langfuse.release": "tawjih " + __version__,
               "langfuse.trace.metadata.path": self.name,
               "langfuse.trace.metadata.backend": rec.get("backend", "")}
        by = rec.get("by") or {}
        if by.get("id") or by.get("name"):     # never an email
            out["langfuse.user.id"] = by.get("id") or by["name"]
        return out

    def spans(self, result=None, error=None):
        rec = self.record
        common = self.common()
        end = time.time()
        status = rec.get("status", "")
        failed = error is not None or status == "failed"
        root = {"traceId": self.trace_id, "spanId": self.root_id,
                "name": "path " + self.name, "kind": 1,
                "startTimeUnixNano": _nanos(self.t0),
                "endTimeUnixNano": _nanos(end),
                "attributes": _attrs(dict(common, **{
                    "langfuse.observation.type": "chain",
                    "langfuse.observation.input": self._mask(
                        rec.get("input", "")),
                    "langfuse.observation.output": self._mask(result)
                    if result is not None else None,
                    "langfuse.observation.level": "ERROR" if failed else (
                        "WARNING" if status == "partial" else "DEFAULT"),
                    "langfuse.observation.status_message": error or (
                        "steps left out: %s" % rec.get("failed_steps")
                        if status == "partial" else ""),
                    "langfuse.observation.metadata.status": status,
                    "langfuse.observation.metadata.steps": len(
                        rec.get("steps", [])),
                    "langfuse.observation.metadata.cross_tree": bool(
                        rec.get("cross_tree")),
                    "langfuse.observation.metadata.round": "%d/%d" % tuple(
                        rec["round"]) if rec.get("round") else ""})),
                "status": {"code": 2 if failed else 1,
                           "message": error or ""}}
        out = [root]
        for p in rec.get("steps", []):
            st = p.get("status", "waiting")
            if st == "waiting":
                continue
            s0 = p.get("started_at") or self.t0
            s1 = s0 + (p.get("seconds") or 0)
            bad = st == "failed"
            out.append({
                "traceId": self.trace_id, "spanId": self.step_ids[p["step"]],
                "parentSpanId": self.root_id,
                "name": "%s %s" % (p.get("number", ""), p.get("title", "")),
                "kind": 1, "startTimeUnixNano": _nanos(s0),
                "endTimeUnixNano": _nanos(s1),
                "attributes": _attrs(dict(common, **{
                    "langfuse.observation.type": "tool" if p.get("code")
                    else "span",
                    "langfuse.observation.level": "ERROR" if bad else (
                        "WARNING" if st in ("skipped", "stopped")
                        else "DEFAULT"),
                    "langfuse.observation.status_message": p.get("error") or
                    (st if st != "done" else ""),
                    "langfuse.observation.metadata.file_id": p.get("id"),
                    "langfuse.observation.metadata.number": p.get("number"),
                    "langfuse.observation.metadata.tree": p.get(
                        "tree_title"),
                    "langfuse.observation.metadata.machine_sha": p.get(
                        "machine_sha"),
                    "langfuse.observation.metadata.status": st,
                    "langfuse.observation.metadata.from_cache": bool(
                        p.get("cached")),
                    "langfuse.observation.metadata.fallback": p.get(
                        "fallback")})),
                "status": {"code": 2 if bad else 1,
                           "message": p.get("error", "")}})
        for c in self.calls:
            u = c["usage"]
            usage = {k2: u[k1] for k1, k2 in (("in", "input"),
                                              ("out", "output"))
                     if isinstance(u.get(k1), int) and u.get(k1)}
            out.append({
                "traceId": self.trace_id, "spanId": _hex(16),
                "parentSpanId": self.step_ids.get(c["step"], self.root_id),
                "name": "model call" + (" (try %d)" % c["attempt"]
                                        if c["attempt"] > 1 else ""),
                "kind": 1, "startTimeUnixNano": _nanos(c["start"]),
                "endTimeUnixNano": _nanos(c["end"]),
                "attributes": _attrs(dict(common, **{
                    "langfuse.observation.type": "generation",
                    "langfuse.observation.model.name": c["model"] or
                    u.get("provider", ""),
                    "gen_ai.request.model": c["model"] or None,
                    "gen_ai.system": u.get("provider"),
                    "langfuse.observation.input": self._mask(u["sent"])
                    if "sent" in u else None,
                    "langfuse.observation.output": self._mask(u["answer"])
                    if "answer" in u else None,
                    "langfuse.observation.usage_details": json.dumps(usage)
                    if usage else None,
                    "langfuse.observation.level": "ERROR" if c["error"]
                    else "DEFAULT",
                    "langfuse.observation.status_message": c["error"],
                    "langfuse.observation.metadata.provider": u.get(
                        "provider"),
                    "langfuse.observation.metadata.attempts": u.get(
                        "attempts"),
                    "langfuse.observation.metadata.redacted": u.get(
                        "redacted")})),
                "status": {"code": 2 if c["error"] else 1,
                           "message": c["error"] or ""}})
        return out

    def payload(self, result=None, error=None):
        return {"resourceSpans": [{
            "resource": {"attributes": _attrs({
                "service.name": "tawjih",
                "service.version": __version__})},
            "scopeSpans": [{"scope": {"name": "tawjih",
                                      "version": __version__},
                            "spans": self.spans(result, error)}]}]}

    def finish(self, result=None, error=None):
        """Send the trace; on failure keep it for `observe flush`. Never
        stops the run."""
        try:
            body = self.payload(result, error)
        except Exception as e:                    # a bug here: say, go on
            print("trace: not made (%s: %s)" % (type(e).__name__, e),
                  file=sys.stderr)
            return
        where = sink()
        ok, why = send(body, where) if where else (False, "nowhere to send")
        if ok:
            print("trace: sent to %s (trace id %s)" % (
                where[0], self.trace_id), file=sys.stderr)
            return
        f = keep(self.root, body, self.trace_id)
        print("trace: not sent (%s); kept in %s for: tawjih observe flush" % (
            why, f), file=sys.stderr)


def send(body, where, timeout=20):
    """POST an OTLP/JSON body. Returns (ok, reason)."""
    kind, url, headers = where
    req = urllib.request.Request(
        url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=dict({"Content-Type": "application/json"}, **headers),
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300, "HTTP %d" % r.status
    except urllib.error.HTTPError as e:
        return False, "HTTP %d: %s" % (e.code, e.read().decode(
            "utf-8", "replace")[:200])
    except (urllib.error.URLError, OSError) as e:
        return False, str(getattr(e, "reason", e))


def keep(root, body, trace_id):
    d = Path(root) / "history" / PENDING
    d.mkdir(parents=True, exist_ok=True)
    f = d / ("%s-%s.json" % (time.strftime("%Y%m%d-%H%M%S"), trace_id[:8]))
    f.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    return f


def cmd_observe(a):
    """Where traces go, a test trace, and sending kept traces again."""
    root = a.project
    where = sink()
    if a.action == "status":
        if not where:
            print("traces go nowhere: set LANGFUSE_PUBLIC_KEY and "
                  "LANGFUSE_SECRET_KEY (and LANGFUSE_HOST for your own "
                  "server), or TAWJIH_OTLP_ENDPOINT for another collector")
        else:
            print("traces go to %s: %s" % (where[0], where[1]))
            if where[0] == "langfuse":
                pk = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
                print("public key: %s…  (the secret key is set)" % pk[:10])
        print("sent with a run: %s" % (
            "always (TAWJIH_TRACE)" if wanted() else "with run --trace"))
        print("texts: %s" % ("sent (after redaction)" if os.environ.get(
            "TAWJIH_TRACE_CONTENT", "1") not in ("0", "no", "false", "off")
            else "not sent (TAWJIH_TRACE_CONTENT=0)"))
        pend = sorted((Path(root) / "history" / PENDING).glob("*.json"))
        if pend:
            print("%d trace(s) waiting: tawjih observe flush" % len(pend))
        return 0
    if not where:
        die("set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY, or "
            "TAWJIH_OTLP_ENDPOINT (see: tawjih observe status)")
    check_allowed(root, where[1])
    if a.action == "check":
        rec = {"input": "a test from tawjih trace check", "status": "done",
               "steps": [], "trees": [], "backend": ""}
        t = Tracer(root, "tawjih-check", [], rec)
        ok, why = send(t.payload("ok"), where)
        print("%s: %s (trace id %s)" % ("sent" if ok else "not sent", why,
                                        t.trace_id))
        return 0 if ok else 1
    if a.action == "flush":
        d = Path(root) / "history" / PENDING
        files = sorted(d.glob("*.json")) if d.is_dir() else []
        sent = 0
        for f in files:
            ok, why = send(json.loads(f.read_text(encoding="utf-8")), where)
            if not ok:
                print("stopped at %s: %s" % (f.name, why), file=sys.stderr)
                break
            f.unlink()
            sent += 1
        print("%d of %d kept trace(s) sent" % (sent, len(files)))
        return 0 if sent == len(files) else 1


__all__ = ['LANGFUSE_CLOUD', 'OTEL_PATH', 'Tracer', 'check_allowed',
           'cmd_observe', 'keep', 'send', 'sink', 'wanted']
