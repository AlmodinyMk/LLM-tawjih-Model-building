"""Running code sides (also sandboxed) and paths (flows).

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping

from .core import (
    Die, MARK, __version__, _json_in, _validate_contract, condition_holds,
    conform, die,
)
from .providers import (
    call_model,
)


# ----------------------------------------------------------------- execution
def build_prompt(store, n, prev, code_out, origin=""):
    """Run-time prompt: the machine side only. The human and reading sides
    are build-time material (see 'edit') and are never sent when running."""
    parts = ["# Instructions\n" + n.instructions]
    if code_out:
        parts.append("# Code result\n" + code_out)
    if origin and origin != prev:
        parts.append("# Original request\n" + origin)
    if prev:
        parts.append("# Input\n" + prev)
    return "\n\n".join(parts)


class StepOutputs(Mapping):
    """The 'steps' a code step can read: outputs of the files upstream of
    it in the path, by id or number. Asking for any other file is an error,
    because its output might not exist yet and the result would then depend
    on the order of execution. An upstream file that was skipped has no
    output: steps.get() gives None and steps[...] raises KeyError."""

    def __init__(self, outputs, upstream, store=None):
        self._out, self._up, self._st = outputs, set(upstream), store

    def _key(self, key):
        if self._st is not None and key not in self._st.nodes \
                and re.fullmatch(r"\d+(\.\d+)*", str(key)):
            try:
                key = self._st.resolve(key)
            except SystemExit:
                pass
        if key not in self._up:
            raise LookupError(
                "steps[%r]: that file is not upstream of this step in the "
                "path, so its output may not exist yet. Connect it to this "
                "step (TO) to read its output." % key)
        return key

    def __getitem__(self, key):
        return self._out[self._key(key)]

    def get(self, key, default=None):
        return self._out.get(self._key(key), default)

    def __contains__(self, key):
        try:
            return self._key(key) in self._out
        except LookupError:
            return False

    def __iter__(self):
        return iter([k for k in self._out if k in self._up])

    def __len__(self):
        return sum(1 for k in self._out if k in self._up)


CODE_TIMEOUT = 60.0


class CodeTimeout(BaseException):
    """Raised inside step code that runs too long. A BaseException, so the
    code's own 'except Exception' cannot swallow it."""


def _exec_limited(code_obj, ns, seconds):
    """exec with a time limit: a trace function checks the clock on every
    line and call. It stops Python loops; a single long call into C code
    (a huge pow, a blocking read) ends only when that call returns."""
    if not seconds or seconds <= 0:
        exec(code_obj, ns)
        return
    deadline = time.monotonic() + seconds

    def tracer(frame, event, arg):
        if time.monotonic() > deadline:
            raise CodeTimeout("the code ran longer than %g seconds (change "
                              "the limit with --code-timeout; 0 for none)"
                              % seconds)
        return tracer

    before = sys.gettrace()
    sys.settrace(tracer)
    try:
        exec(code_obj, ns)
    finally:
        sys.settrace(before)


SANDBOX_BOOT = r"""
import contextlib, io, json, os, sys
d = json.loads(sys.stdin.read())
try:
    import resource
    mem = d["memory_mb"] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    cpu = int(d["timeout"]) + 1
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    size = d["file_mb"] * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_FSIZE, (size, size))
except Exception:
    pass
links = d["linked"]
def linked(rel_type=""):
    return [x for x in links if not rel_type or x["type"] == rel_type]
class Steps(dict):
    def __missing__(self, key):
        raise LookupError("%s is not upstream of this step" % key)
ns = {"input": d["input"], "origin": d["origin"], "vars": d["vars"],
      "steps": Steps(d["steps"]), "meta": d["meta"], "linked": linked,
      "output": None}
buf = io.StringIO()
res = {}
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(d["code"], d["name"], "exec"), ns)
    out = ns.get("output")
    res["output"] = None if out is None else str(out)
except BaseException as e:
    res["error"] = "%s: %s" % (type(e).__name__, e)
res["printed"] = buf.getvalue()
try:
    json.dumps(ns["vars"])
    res["vars"] = ns["vars"]
except (TypeError, ValueError):
    res["vars"] = None
sys.stdout.write(json.dumps(res))
"""
SANDBOX_MEMORY_MB = 512
SANDBOX_FILE_MB = 16
SANDBOX_LIMITS = {"memory_mb": SANDBOX_MEMORY_MB, "file_mb": SANDBOX_FILE_MB}


class SandboxError(Exception):
    pass


def run_code_sandboxed(n, prev, shared, origin, outputs, upstream, meta,
                       links, timeout):
    """Run a code side in a separate Python process: no access to Tawjih's
    memory, a temporary working folder, an empty environment, and on Linux
    and macOS limits on memory, processor time and file size. It does not
    cut off the network or the file system outside the working folder."""
    import tempfile
    data = {"code": n.code, "name": n.id + ".code", "input": prev,
            "origin": origin, "vars": shared,
            "steps": {k: v for k, v in outputs.items() if k in upstream},
            "meta": meta, "linked": links, "timeout": timeout or 60,
            "memory_mb": SANDBOX_LIMITS["memory_mb"],
            "file_mb": SANDBOX_LIMITS["file_mb"]}
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
    if os.name == "nt":
        env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", "")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            r = subprocess.run([sys.executable, "-I", "-c", SANDBOX_BOOT],
                               input=json.dumps(data), capture_output=True,
                               text=True, encoding="utf-8", cwd=tmp, env=env,
                               timeout=(timeout or None))
        except subprocess.TimeoutExpired:
            raise CodeTimeout("the code ran longer than %g seconds in the "
                              "sandbox and was stopped" % timeout)
    try:
        res = json.loads(r.stdout)
    except ValueError:
        raise SandboxError("the sandboxed code ended abnormally (exit %s): %s"
                           % (r.returncode, (r.stderr or "").strip()[-300:]
                              or "it may have hit a memory or time limit"))
    if "error" in res:
        raise SandboxError(res["error"])
    if isinstance(res.get("vars"), dict):
        shared.clear()
        shared.update(res["vars"])
    out = res.get("output")
    if out is not None:
        return out
    return res.get("printed", "").strip() or None


def run_code(n, prev, shared, origin="", steps=None, store=None, run_dir=None,
             upstream=None, timeout=CODE_TIMEOUT, sandbox=False):
    """Execute a node's Python code. It sees input, origin, vars, steps
    (outputs upstream of it), meta and linked(), and sets output."""
    outputs = steps or {}
    if upstream is None:
        upstream = set(outputs)
    steps_view = StepOutputs(outputs, upstream, store)

    def linked_info(rel_type=""):
        if not store:
            return []
        res = []
        for tid, ty in n.links:
            if rel_type and ty != rel_type:
                continue
            tn = store.nodes.get(tid)
            if tn:
                res.append({
                    "id": tn.id,
                    "number": store.number(tn.id),
                    "title": tn.title,
                    "type": ty,
                    # only an upstream output is certain to be there
                    "output": outputs.get(tn.id) if tn.id in upstream else None,
                    "upstream": tn.id in upstream,
                    "expects": tn.expects,
                    "produces": tn.produces,
                })
        return res

    meta = {
        "id": n.id,
        "title": n.title,
        "number": store.number(n.id) if store else "",
        "tree": store.ancestors(n.id)[0].title if store else "",
        "run_dir": str(run_dir) if run_dir else "",
    }
    if sandbox:
        return run_code_sandboxed(n, prev, shared, origin, outputs, upstream,
                                  meta, linked_info(), timeout)
    ns = {
        "input": prev,
        "origin": origin,
        "vars": shared,
        "steps": steps_view,
        "meta": meta,
        "linked": linked_info,
        "output": None,
    }
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _exec_limited(compile(n.code, n.id + ".code", "exec"), ns, timeout)
    out = ns.get("output")
    if out is not None:
        return str(out)
    return buf.getvalue().strip() or None    # None: nothing was produced


def machine_side(n):
    """Extract the machine side of a file: (instructions, code, digest).
    The digest identifies the exact machine-side version used in a run."""
    ins, code = n.instructions.strip("\n"), n.code.strip("\n")
    blob = ins + "\n" + MARK.format("code") + "\n" + code
    return ins, code, hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def split_statements(tokens):
    """Statements on a command line are separated by ',' or '/'."""
    out, cur = [], []
    for t in tokens:
        end = t in (",", "/", "؛", "،") or t.endswith((",", "؛", "،"))
        t = "" if t in (",", "/", "؛", "،") else t.rstrip(",؛،")
        if t:
            cur.append(t)
        if end and cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def flow_text(store, flow):
    lines = []
    for norm in flow.lines:
        words = [store.nodes[t].title if t in store.nodes else t for t in norm]
        lines.append("%s  # %s" % (" ".join(norm), " | ".join(words)))
    return "\n".join(lines) + "\n"


def write_flow(store, name, flow):
    store.pdir.mkdir(exist_ok=True)
    f = store.pdir / (name + ".path")
    f.write_text(flow_text(store, flow), encoding="utf-8")
    return f


def edge_text(store, e):
    t = "%s %s" % (store.number(e["src"]), store.nodes[e["src"]].title)
    if e.get("order"):
        t += " (LINK: after it, without its output)"
    if e["among"]:
        t += " (if chosen)"
    if e["cond"]:
        t += " (IF %s%s)" % ("NOT " if e["cond"][0] else "", e["cond"][1])
    return t


def plan_steps(store, flow):
    """Describe a flow as production steps in execution order."""
    plan = []
    for i, nid in enumerate(flow.order(), 1):
        n = store.nodes[nid]
        root = store.ancestors(nid)[0]
        ins, code, digest = machine_side(n)
        plan.append({"step": i, "id": nid, "number": store.number(nid),
                     "title": n.title, "tree": root.id,
                     "tree_title": root.title,
                     "model": n.model,
                     "expects": n.expects,
                     "produces": n.produces,
                     "stale": n.is_stale(),
                     "instructions": bool(ins.strip()),
                     "code": bool(code.strip()), "machine_sha": digest,
                     "instructions_chars": len(ins), "code_chars": len(code),
                     "path": dict(flow.steps.get(nid, {})),
                     "from": [edge_text(store, e) for e in flow.incoming(nid)]})
    return plan


def print_plan(store, name, plan, out):
    trees = []
    for p in plan:
        if p["tree_title"] not in trees:
            trees.append(p["tree_title"])
    print("'%s': %d step(s) across %d tree(s): %s" % (
        name, len(plan), len(trees), " | ".join(trees)), file=out)
    for p in plan:
        kind = "+".join(k for k in ("instructions", "code") if p[k]) or "EMPTY"
        flags = []
        ps = p.get("path", {})
        if ps.get("using") or p["model"]:
            flags.append("model=" + (ps.get("using") or p["model"]))
        for k in ("each", "approve", "retry", "pick"):
            if ps.get(k):
                flags.append("%s=%s" % (k.upper(), ps[k]))
        if ps.get("until"):
            flags.append("UNTIL %s%s" % ("NOT " if ps["until"][0] else "",
                                         ps["until"][1]))
        if p["stale"]:
            flags.append("STALE")
        extra = ("  [%s]" % ", ".join(flags)) if flags else ""
        print("%3d  %-14s %-10s [%s]  %-18s %s%s" % (
            p["step"], p["tree_title"], p["number"], p["id"], kind,
            p["title"], extra), file=out)
        size = ["%s %d" % (k, p[k + "_chars"])
                for k in ("instructions", "code") if p[k]]
        if size:
            print("       size: %s characters" % ", ".join(size), file=out)
        for src in p["from"]:
            print("       <- %s" % src, file=out)
        if not p["from"]:
            print("       <- (the original request)", file=out)
    sent = sum(p["instructions_chars"] for p in plan)
    print("sent to the model: %d characters of instructions in %d step(s), "
          "plus the request and each step's input at run time" % (
              sent, sum(1 for p in plan if p["instructions"])), file=out)


def choose_among(store, output, candidates):
    """The candidate named first in the output (by id, number or title),
    falling back to contract compatibility if a single candidate matches."""
    low, best = output.casefold(), None
    for nid in candidates:
        n = store.nodes[nid]
        hits = [m.start() for m in re.finditer(
            r"(?<![\d.])%s(?![\d.])" % re.escape(store.number(nid)), output)]
        for key in (nid, n.title.strip()):
            if key:
                m = re.search(
                    r"(?:^|[\s\W])%s(?:$|[\s\W])" % re.escape(key.casefold()),
                    low)
                if m:
                    hits.append(m.start())
        if hits and (best is None or min(hits) < best[0]):
            best = (min(hits), nid)
    if best:
        return best[1]
    # Contract-aware branch selection fallback
    contract_hits = [
        nid for nid in candidates
        if store.nodes[nid].expects
        and _validate_contract(output, store.nodes[nid].expects, "") is None
    ]
    if len(contract_hits) == 1:
        return contract_hits[0]
    return None


def _stamped(folder, name, suffix):
    folder.mkdir(exist_ok=True)
    stamp, k = time.strftime("%Y%m%d-%H%M%S"), 1
    f = folder / ("%s-%s%s" % (name, stamp, suffix))
    while f.exists():
        k += 1
        f = folder / ("%s-%s-%d%s" % (name, stamp, k, suffix))
    return f


def cache_file(store, backend, model, prompt):
    """Where the answer to this exact prompt, from this provider and model,
    is kept. Any change to the instructions, the input or the model gives
    another file."""
    key = hashlib.sha256("\0".join((
        backend, model, prompt)).encode("utf-8")).hexdigest()
    return store.root / "cache" / (key[:32] + ".json")


class StepFailure(Exception):
    def __init__(self, what, err):
        super().__init__("%s: %s" % (what, err))
        self.what, self.err = what, err


def _ask_approval(n, when, text, mode):
    """'auto' approves, 'deny' stops, 'ask' asks on the terminal. After a
    step the person may also type a replacement output ('e')."""
    if mode == "auto":
        return True, None
    if mode == "deny":
        return False, None
    bar = "-" * 60
    if when == "before":
        print("\n%s\napproval: run step %s [%s]? input:\n%s\n%s\n[y]es / "
              "[n]o: " % (bar, n.title, n.id, text[:2000], bar), end="",
              file=sys.stderr, flush=True)
    else:
        print("\n%s\napproval: pass on the output of %s [%s]?\n%s\n%s\n"
              "[y]es / [n]o / [e]dit: " % (bar, n.title, n.id, text, bar),
              end="", file=sys.stderr, flush=True)
    ans = sys.stdin.readline().strip().lower()
    if ans in ("y", "yes", "نعم", "ن"):
        return True, None
    if ans in ("e", "edit") and when == "after":
        print("new output (finish with a line containing only '.'):",
              file=sys.stderr)
        lines = []
        while True:
            line = sys.stdin.readline()
            if not line or line.strip() == ".":
                break
            lines.append(line.rstrip("\n"))
        return True, "\n".join(lines).strip()
    return False, None


def run_flow(store, name, text, backend, allow_code, flow, resume=False,
             use_cache=True, redo=(), model_override="",
             code_timeout=CODE_TIMEOUT, retries=None, approve=None,
             sandbox=False):
    """Execute the machine sides of the files of a flow. A file runs after
    the files that feed it and receives their outputs; a file none of whose
    connections is active is skipped. Writes the log, state snapshot and
    result to runs/ and a record of the operation to history/."""
    use_cache = use_cache and backend != "dry"
    if approve is None:
        approve = "ask" if sys.stdin is not None and sys.stdin.isatty() \
            else "deny"
    plan = plan_steps(store, flow)
    trees = []
    for p in plan:
        if p["tree"] not in [t["id"] for t in trees]:
            trees.append({"id": p["tree"], "title": p["tree_title"]})
    tree_of = {p["id"]: p["tree"] for p in plan}
    log = ["# Run of path '%s'" % name]
    logfile = _stamped(store.rdir, name, ".md")
    resfile = logfile.with_name(logfile.stem + ".result.txt")
    statefile = logfile.with_name(logfile.stem + ".state.json")
    default_model = os.environ.get("TAWJIH_MODEL", "")
    record = {"tawjih": __version__, "name": name,
              "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "backend": backend, "allow_code": bool(allow_code),
              "input_chars": len(text), "input": text,
              "input_sha": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
              "trees": trees, "cross_tree": len(trees) > 1, "handoffs": 0,
              "flow": [" ".join(l) for l in flow.lines],
              "connections": [{"from": e["src"], "to": e["dst"],
                               "if": e["cond"], "among": e["among"],
                               "order_only": bool(e.get("order"))}
                              for e in flow.edges],
              "calls": list(flow.calls),
              "steps": plan, "status": "running", "log": str(logfile),
              "state": str(statefile),
              "result": "", "model": default_model,
              "resume": bool(resume), "from_cache": 0, "redo": list(redo),
              "usage": {"in": 0, "out": 0, "seconds": 0.0, "calls": 0,
                        "retries": 0, "fallbacks": 0, "redacted": 0}}

    def save_state(outputs_map):
        statefile.write_text(json.dumps({
            "path": name,
            "origin": text,
            "steps": outputs_map,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def document():
        record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        record["usage"]["seconds"] = round(record["usage"]["seconds"], 3)
        rec = _stamped(store.root / "history", "production-" + name, ".json")
        rec.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        return rec

    def fail(p, n, what, e, outputs_map):
        log.append("### FAILED (%s)\n%s: %s" % (what, type(e).__name__, e))
        logfile.write_text("\n".join(log) + "\n", encoding="utf-8")
        save_state(outputs_map)
        p["status"] = "failed"
        record.update(status="failed", failed_step=p["step"],
                      error="%s: %s: %s" % (what, type(e).__name__, e))
        rec = document()
        suspects = [n.id] if what in ("AMONG", "contract") \
            else p.get("received_from", [])
        suspects = [x for x in suspects if use_cache
                    and store.nodes[x].instructions.strip()]
        hint = ""
        if suspects and what in ("code", "AMONG", "output", "contract",
                                 "input contract"):
            hint = ("\nIf a model answer caused this, ask for it again and "
                    "keep the rest: --resume " + " ".join(
                        "--redo " + store.number(x) for x in suspects))
        die("step %d [%s] %s failed in %s: %s: %s (log: %s, record: %s)%s"
            % (p["step"], n.id, n.title, what, type(e).__name__, e, logfile,
               rec, hint))

    outputs, chosen, shared = {}, {}, {}

    def active(e):
        if e["src"] not in outputs:
            return False
        if e["among"] is not None:
            return chosen.get((e["src"], e["among"])) == e["dst"]
        if e["cond"]:
            neg, word = e["cond"]
            return condition_holds(outputs[e["src"]], word) != neg
        return True

    def settings(p, n):
        """This path's settings for the step; none for a fallback file."""
        return flow.steps.get(n.id, {}) if n.id == p["id"] else {}

    def execute(p, n, prev, note=""):
        """The step on its input, or with 'each' once per item of a JSON
        list (the input itself, or the list under that key); the outputs
        are passed on as {"items": [...]}."""
        each = settings(p, n).get("each") or n.each
        if not each:
            return execute_one(p, n, prev, note)
        obj, _ = _json_in(prev)
        if obj is None:
            try:
                obj = json.loads(prev)
            except ValueError:
                obj = None
        items = obj if isinstance(obj, list) else (
            obj.get(each) if isinstance(obj, dict) and each != "yes"
            else None)
        if not isinstance(items, list):
            raise StepFailure("each", ValueError(
                "the input is not a JSON list%s" % (
                    "" if each == "yes" else
                    " and has no list under '%s'" % each)))
        results = []
        for k, item in enumerate(items, 1):
            text = item if isinstance(item, str) else json.dumps(
                item, ensure_ascii=False)
            print("  item %d of %d" % (k, len(items)), file=sys.stderr)
            log.append("#### Item %d of %d" % (k, len(items)))
            out = execute_one(p, n, text, note)
            parsed, _ = _json_in(out)
            results.append(parsed if parsed is not None else out)
        return json.dumps({"items": results}, ensure_ascii=False)

    def execute_one(p, n, prev, note=""):
        """Run the machine side of n on prev and return its checked output,
        or raise StepFailure. Model answers that break the contract are
        asked again, up to the retries allowed."""
        ins, code, _ = machine_side(n)
        ps = settings(p, n)
        # --model > the path's USING > the file's model: > TAWJIH_MODEL
        step_model = model_override or ps.get("using") or n.model or \
            default_model
        p["model"] = step_model
        if n.expects:
            prev, cerr = conform(prev, n.expects, "input")
            if cerr:
                raise StepFailure("input contract", ValueError(cerr))
        code_out, out, emptied = "", prev, False
        if code.strip():
            if allow_code:
                try:
                    produced = run_code(n, prev, shared, text, steps=outputs,
                                        store=store, run_dir=store.rdir,
                                        upstream=flow.upstream(p["id"]),
                                        timeout=code_timeout, sandbox=sandbox)
                except (Exception, CodeTimeout) as e:
                    raise StepFailure("code", e)
                emptied = produced is not None and not produced.strip()
                code_out = produced or ""
            else:
                print("  code skipped (pass --allow-code to execute it)",
                      file=sys.stderr)
        if ins.strip():
            prompt = build_prompt(store, n, prev, code_out, text) + note
            print("  prompt: %d characters" % len(prompt), file=sys.stderr)
            log.append("### Prompt\n" + prompt)
            cfile = (cache_file(store, backend, step_model, prompt)
                     if use_cache else None)
            kept = None
            if resume and cfile is not None and cfile.is_file() \
                    and p["id"] not in redo:
                try:
                    kept = json.loads(cfile.read_text(encoding="utf-8"))["answer"]
                except (ValueError, KeyError, TypeError):
                    kept = None
            if isinstance(kept, str) and kept.strip():
                out = kept
                p["cached"] = True
                record["from_cache"] += 1
                print("  answer taken from the cache", file=sys.stderr)
                log.append("(answer taken from the cache: %s)" % cfile.name)
            else:
                allowed = retries if retries is not None else \
                    ps.get("retry", int(n.retries or 0))
                ask, problem = prompt, None
                for attempt in range(allowed + 1):
                    if attempt:
                        record["usage"]["retries"] += 1
                        print("  asking again (%d of %d): %s" % (
                            attempt, allowed, problem), file=sys.stderr)
                        log.append("(answer rejected: %s; asked again)"
                                   % problem)
                        ask = (prompt + "\n\n## Your previous answer was "
                               "rejected\n" + problem + "\nAnswer again, "
                               "following the instructions exactly.")
                    try:
                        out, usage = call_model(store.root, backend, ask,
                                                step_model)
                    except Die:
                        raise
                    except Exception as e:
                        raise StepFailure("model call", e)
                    u = record["usage"]
                    u["calls"] += 1
                    for k in ("in", "out", "seconds", "redacted"):
                        u[k] += usage.get(k, 0)
                    pu = p.setdefault("usage", {"in": 0, "out": 0,
                                                "seconds": 0.0, "calls": 0})
                    pu["calls"] += 1
                    for k in ("in", "out", "seconds"):
                        pu[k] = round(pu[k] + usage.get(k, 0), 3)
                    if usage.get("redacted"):
                        log.append("(%d private item(s) were replaced by "
                                   "placeholders before sending)"
                                   % usage["redacted"])
                    problem = None
                    if not out.strip():
                        problem = "the model returned an empty answer"
                    elif n.produces:   # never keep an answer the contract rejects
                        problem = conform(out, n.produces, "output")[1]
                    if not problem:
                        break
                if problem:
                    raise StepFailure("contract" if out.strip() else
                                      "model call", ValueError(problem))
                if cfile is not None:
                    cfile.parent.mkdir(exist_ok=True)
                    cfile.write_text(json.dumps({
                        "answer": out, "prompt": prompt, "backend": backend,
                        "model": step_model,
                        "step": n.id, "machine_sha": p["machine_sha"],
                        "saved": time.strftime("%Y-%m-%d %H:%M:%S")},
                        ensure_ascii=False, indent=2), encoding="utf-8")
        elif code_out:
            out = code_out
        elif emptied:
            out = ""
        if not out.strip():
            raise StepFailure("output", ValueError(
                "the step produced no output, so nothing can be passed on"))
        if n.produces:
            out, cerr = conform(out, n.produces, "output")
            if cerr:
                raise StepFailure("contract", ValueError(cerr))
        return out

    for p in plan:
        n = store.nodes[p["id"]]
        head = "step %d/%d  %s [%s] %s" % (p["step"], len(plan), p["number"],
                                           n.id, n.title)
        if n.is_stale():
            print("  warning: %s [%s] machine side is stale" % (n.title, n.id),
                  file=sys.stderr)
        log.append("\n## Step %d: %s [%s] (tree: %s, machine %s)" % (
            p["step"], n.title, n.id, p["tree_title"], p["machine_sha"]))
        incoming = flow.incoming(n.id)
        if not incoming:
            prev, feeders = text, []
        else:
            live = [e for e in incoming if active(e)]
            if not live:
                p["status"] = "skipped"
                print(head + "  (skipped: no active connection)",
                      file=sys.stderr)
                log.append("skipped: no active connection")
                continue
            feeders = [e["src"] for e in live if not e.get("order")]
            p["after"] = [e["src"] for e in live if e.get("order")]
            if not feeders:              # only LINKs: the original request
                prev = text
            elif len(feeders) == 1:
                prev = outputs[feeders[0]]
            else:
                prev = "\n\n".join("## from: %s [%s]\n%s" % (
                    store.nodes[f].title, f, outputs[f]) for f in feeders)
        print(head, file=sys.stderr)
        p["received_from"] = feeders
        for f in feeders:
            log.append("input from: %s [%s]" % (store.nodes[f].title, f))
            if tree_of[f] != p["tree"]:
                record["handoffs"] += 1
                print("  handoff from tree: %s" % store.ancestors(f)[0].title,
                      file=sys.stderr)
        ps = flow.steps.get(n.id, {})
        approve_when = n.approve
        if ps.get("approve") and approve_when not in ("before", "both"):
            approve_when = "both" if approve_when == "after" else "before"
        if approve_when in ("before", "both"):
            ok, _ = _ask_approval(n, "before", prev, approve)
            if not ok:
                fail(p, n, "approval", ValueError(
                    "not approved before the step ran" + (
                        " (run with --approve ask on a terminal, or "
                        "--approve auto)" if approve == "deny" else "")),
                    outputs)
        try:
            out = execute(p, n, prev)
            until = ps.get("until")
            for attempt in range(ps.get("retry", 0) if until else 0):
                if condition_holds(out, until[1]) != until[0]:
                    break
                record["usage"]["retries"] += 1
                print("  UNTIL %s%s not met; trying again (%d of %d)" % (
                    "NOT " if until[0] else "", until[1], attempt + 1,
                    ps["retry"]), file=sys.stderr)
                log.append("(UNTIL %s%s not met; tried again)" % (
                    "NOT " if until[0] else "", until[1]))
                out = execute(p, n, prev, note=(
                    "\n\n## Your previous answer was not accepted\nIt must "
                    "%s'%s'. Answer again.\n\nPrevious answer:\n%s" % (
                        "not satisfy " if until[0] else "satisfy ",
                        until[1], out)))
            if until and condition_holds(out, until[1]) == until[0]:
                raise StepFailure("UNTIL", ValueError(
                    "the output still does not meet UNTIL %s%s after %d "
                    "tries" % ("NOT " if until[0] else "", until[1],
                               ps["retry"] + 1)))
        except StepFailure as sf:
            fb = store.nodes.get(n.fallback) if n.fallback else None
            if fb is None or not (fb.instructions.strip() or fb.code.strip()):
                fail(p, n, sf.what, sf.err, outputs)
            print("  failed (%s: %s); falling back to %s [%s]" % (
                sf.what, sf.err, fb.title, fb.id), file=sys.stderr)
            log.append("### FAILED (%s), falling back to %s [%s]\n%s" % (
                sf.what, fb.title, fb.id, sf.err))
            record["usage"]["fallbacks"] += 1
            p["fallback"] = fb.id
            try:
                out = execute(p, fb, prev)
            except StepFailure as sf2:
                fail(p, n, sf2.what + " (in the fallback %s)" % fb.title,
                     sf2.err, outputs)
        if approve_when in ("after", "both"):
            ok, edited = _ask_approval(n, "after", out, approve)
            if not ok:
                fail(p, n, "approval", ValueError(
                    "the output was not approved" + (
                        " (run with --approve ask on a terminal, or "
                        "--approve auto)" if approve == "deny" else "")),
                    outputs)
            if edited is not None:
                log.append("(output replaced by hand at approval)")
                if n.produces:
                    edited, cerr = conform(edited, n.produces, "output")
                    if cerr:
                        fail(p, n, "contract", ValueError(cerr), outputs)
                out = edited
        if ps.get("pick"):
            obj, _ = _json_in(out)
            if not isinstance(obj, dict) or ps["pick"] not in obj:
                fail(p, n, "PICK", ValueError(
                    "the output is not a JSON object with the key '%s'"
                    % ps["pick"]), outputs)
            v = obj[ps["pick"]]
            out = v if isinstance(v, str) else json.dumps(v,
                                                          ensure_ascii=False)
            log.append("(PICK: only '%s' is passed on)" % ps["pick"])
        outputs[n.id] = out
        log.append("### Output\n" + out)
        groups = {}
        for e in flow.edges:
            if e["src"] == n.id and e["among"] is not None:
                groups.setdefault(e["among"], []).append(e["dst"])
        for gid, cands in groups.items():
            pick = choose_among(store, out, cands)
            if pick is None:
                fail(p, n, "AMONG", ValueError(
                    "the output names none of: " + ", ".join(
                        "%s %s" % (store.number(c), store.nodes[c].title)
                        for c in cands)), outputs)
            chosen[(n.id, gid)] = pick
            print("  chose: %s %s" % (store.number(pick),
                                      store.nodes[pick].title), file=sys.stderr)
            log.append("chose: %s [%s]" % (store.nodes[pick].title, pick))
        p.update(status="done", output_chars=len(out))
    ends = [t for t in flow.terminals() if t in outputs]
    if len(ends) == 1:
        result = outputs[ends[0]]
    else:
        result = "\n\n".join("## %s [%s]\n%s" % (store.nodes[t].title, t,
                                                  outputs[t]) for t in ends)
    if not ends:
        print("warning: no final step ran, the result is empty",
              file=sys.stderr)
    logfile.write_text("\n".join(log) + "\n", encoding="utf-8")
    resfile.write_text(result + "\n", encoding="utf-8")
    save_state(outputs)
    record.update(status="done", result=str(resfile), final_steps=ends)
    rec = document()
    print("log: %s\nstate: %s\nresult: %s\nrecord: %s" % (
        logfile, statefile, resfile, rec), file=sys.stderr)
    return result


__all__ = [
    'CODE_TIMEOUT', 'CodeTimeout', 'SANDBOX_BOOT', 'SANDBOX_FILE_MB',
    'SANDBOX_LIMITS', 'SANDBOX_MEMORY_MB', 'SandboxError', 'StepFailure',
    'StepOutputs', '_ask_approval', '_exec_limited', '_stamped',
    'build_prompt', 'cache_file', 'choose_among', 'edge_text',
    'flow_text', 'machine_side', 'plan_steps', 'print_plan', 'run_code',
    'run_code_sandboxed', 'run_flow', 'split_statements', 'write_flow',
]
