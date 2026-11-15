#!/usr/bin/env python3
"""tawjih - personal LLM command-directing (توجيه أوامر الآلة).

A project is a folder of three-sided files arranged as trees:
    human side   : the expressive human concept (context)
    machine side : instructions for the model + optional Python code
    reading side : filtered passages of deposited books and research
A path is an ordered list of files; running it executes their machine
sides as steps, each step's output feeding the next (a linear workflow).
The human and reading sides are used only when building a file ('edit').

Standard library only. Python 3.11+.

Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
import argparse
import contextlib
import difflib
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

__version__ = "0.4.0"

LEVELS = ["general", "section", "chapter", "topic", "branch", "part"]
LEVELS_AR = {"عام": 1, "قسم": 2, "فصل": 3, "موضوع": 4, "فرع": 5, "جزء": 6}
SECTIONS = ["human", "instructions", "code", "reading"]
MARK = "=== {} ==="


class Die(SystemExit):
    """Exit with status 1; keeps the message for callers that catch it."""

    def __init__(self, msg):
        super().__init__(1)
        self.msg = msg


def die(msg):
    print("error: " + msg, file=sys.stderr)
    raise Die(msg)


def parse_level(text):
    text = text.strip()
    if text.isdigit() and 1 <= int(text) <= len(LEVELS):
        return int(text)
    if text.lower() in LEVELS:
        return LEVELS.index(text.lower()) + 1
    if text in LEVELS_AR:
        return LEVELS_AR[text]
    die("unknown level '%s' (use 1-6, %s)" % (text, ", ".join(LEVELS)))


# ---------------------------------------------------------------- file model
class Node:
    def __init__(self, id="", title="", level=1, parent="", order=0,
                 links=None, human="", instructions="", code="", reading=""):
        self.id, self.title, self.level = id, title, int(level)
        self.parent, self.order = parent, int(order)
        self.links = links or []          # [(target_id, type)]
        self.human, self.instructions, self.code = human, instructions, code
        self.reading = reading

    def dump(self):
        links = ", ".join("%s:%s" % (t, ty) for t, ty in self.links)
        out = ["---", "id: " + self.id, "title: " + self.title,
               "level: %d" % self.level, "parent: " + self.parent,
               "order: %d" % self.order, "links: " + links, "---"]
        for s in SECTIONS:
            out += [MARK.format(s), getattr(self, s).strip("\n")]
        return "\n".join(out) + "\n"

    @staticmethod
    def parse(text):
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            raise ValueError("missing '---' header")
        end = next((i for i in range(1, len(lines))
                    if lines[i].strip() == "---"), None)
        if end is None:
            raise ValueError("header is not closed with '---'")
        meta = {}
        for line in lines[1:end]:
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
        if not meta.get("id"):
            raise ValueError("header has no id")
        sec, cur = {s: [] for s in SECTIONS}, None
        for line in lines[end + 1:]:
            hit = [s for s in SECTIONS if line.strip() == MARK.format(s)]
            if hit:
                cur = hit[0]
            elif cur:
                sec[cur].append(line)
        links = []
        for part in meta.get("links", "").split(","):
            if part.strip():
                t, _, ty = part.strip().partition(":")
                links.append((t.strip(), ty.strip() or "related"))
        return Node(id=meta["id"], title=meta.get("title", ""),
                    level=meta.get("level", "1") or 1,
                    parent=meta.get("parent", ""),
                    order=meta.get("order", "0") or 0, links=links,
                    **{s: "\n".join(sec[s]).strip("\n") for s in SECTIONS})


# --------------------------------------------------------------------- store
class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.fdir = self.root / "files"
        self.pdir = self.root / "paths"
        self.rdir = self.root / "runs"
        if not self.fdir.is_dir():
            die("not a tawjih project here (run: tawjih.py init)")
        self.nodes = {}
        for f in sorted(self.fdir.glob("*.md")):
            try:
                n = Node.parse(f.read_text(encoding="utf-8"))
            except Exception as e:
                die("%s: %s" % (f.name, e))
            if n.id != f.stem:
                die("%s: the id in the header (%s) does not match the file "
                    "name" % (f.name, n.id))
            self.nodes[n.id] = n

    def file_of(self, nid):
        return self.fdir / (nid + ".md")

    def save(self, n):
        self.file_of(n.id).write_text(n.dump(), encoding="utf-8")
        self.nodes[n.id] = n

    def children(self, pid):
        return sorted((n for n in self.nodes.values() if n.parent == pid),
                      key=lambda n: (n.order, n.id))

    def number(self, nid):
        """Positional number such as 1.2.3 (changes when files move)."""
        parts, seen = [], set()
        while nid in self.nodes and nid not in seen:
            seen.add(nid)
            n = self.nodes[nid]
            sibs = [s.id for s in self.children(n.parent)]
            parts.append(str(sibs.index(nid) + 1))
            nid = n.parent
        return ".".join(reversed(parts))

    def resolve(self, ref):
        """Accept a stable id or a positional number."""
        if ref in self.nodes:
            return ref
        if re.fullmatch(r"\d+(\.\d+)*", ref):
            cur = ""
            for idx in ref.split("."):
                kids = self.children(cur)
                if not 1 <= int(idx) <= len(kids):
                    die("no file numbered " + ref)
                cur = kids[int(idx) - 1].id
            return cur
        die("unknown file '%s'" % ref)

    def ancestors(self, nid):
        """Root-first chain ending at nid."""
        chain, seen = [], set()
        while nid in self.nodes and nid not in seen:
            seen.add(nid)
            chain.append(self.nodes[nid])
            nid = self.nodes[nid].parent
        return list(reversed(chain))

    def new_node(self, title, level, parent):
        nid = uuid.uuid4().hex[:6]
        while nid in self.nodes:
            nid = uuid.uuid4().hex[:6]
        order = max([s.order for s in self.children(parent)] + [0]) + 1
        title = " ".join(title.split())
        if not title:
            die("the title is empty")
        n = Node(id=nid, title=title, level=level, parent=parent, order=order)
        self.save(n)
        return n

    def load_path(self, name, whole_lines=False):
        if not re.fullmatch(r"[\w\-]+", name):
            die("path names may contain only letters, digits, '_' and '-'")
        f = self.pdir / (name + ".path")
        if not f.is_file():
            die("no path named '%s'" % name)
        lines = [l.split("#", 1)[0].strip()
                 for l in f.read_text(encoding="utf-8").splitlines()]
        lines = [l for l in lines if l]
        return lines if whole_lines else [l.split()[0] for l in lines]

    def load_flow(self, name):
        """Read paths/NAME.path as a Flow (see the flow section)."""
        def known(ref):
            if ref not in self.nodes:
                raise FlowError("file %s does not exist" % ref)
            return ref
        try:
            stmts = [shlex.split(line) for line in self.load_path(name, True)]
        except ValueError as e:
            raise FlowError("unreadable line (%s)" % e)
        return build_flow(stmts, known)

    def check(self):
        errors, warns = [], []
        for n in self.nodes.values():
            tag = "%s (%s)" % (n.id, n.title)
            if n.parent:
                p = self.nodes.get(n.parent)
                if p is None:
                    errors.append(tag + ": parent %s does not exist" % n.parent)
                elif n.level <= p.level:
                    errors.append(tag + ": level %d is not deeper than parent "
                                  "level %d" % (n.level, p.level))
            seen, cur = set(), n.id
            while cur in self.nodes:
                if cur in seen:
                    errors.append(tag + ": parent chain contains a cycle")
                    break
                seen.add(cur)
                cur = self.nodes[cur].parent
            for t, _ in n.links:
                if t not in self.nodes:
                    errors.append(tag + ": link target %s does not exist" % t)
            if not n.human.strip():
                warns.append(tag + ": human side is empty")
            if n.code.strip():
                try:
                    compile(n.code, n.id + ".code", "exec")
                except SyntaxError as e:
                    errors.append(tag + ": code has a syntax error (%s)" % e)
        if self.pdir.is_dir():
            for f in sorted(self.pdir.glob("*.path")):
                try:
                    flow = self.load_flow(f.stem)
                    flow.order()
                except FlowError as e:
                    errors.append("path %s: %s" % (f.stem, e))
                    continue
                for nid in flow.nodes:
                    n = self.nodes[nid]
                    if not n.instructions.strip() and not n.code.strip():
                        warns.append("path %s: step %s has an empty machine "
                                     "side" % (f.stem, nid))
        return errors, warns


# ------------------------------------------------------------------ backends
def _post(url, body, headers):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers=dict({"Content-Type": "application/json"}, **headers))
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError("HTTP %d from %s: %s" % (e.code, url, detail))


def _model():
    m = os.environ.get("TAWJIH_MODEL")
    if not m:
        die("set the TAWJIH_MODEL environment variable to the model name")
    return m


def backend_dry(prompt):
    return "[dry run: no model was called; prompt was %d characters]" % len(prompt)


def backend_ollama(prompt):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    data = _post(host + "/api/chat",
                 {"model": _model(), "stream": False,
                  "messages": [{"role": "user", "content": prompt}]}, {})
    return data["message"]["content"]


def backend_anthropic(prompt):
    key = os.environ.get("ANTHROPIC_API_KEY") or die("set ANTHROPIC_API_KEY")
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    data = _post(base.rstrip("/") + "/v1/messages",
                 {"model": _model(),
                  "max_tokens": int(os.environ.get("TAWJIH_MAX_TOKENS", "4096")),
                  "messages": [{"role": "user", "content": prompt}]},
                 {"x-api-key": key, "anthropic-version": "2023-06-01"})
    return "".join(b.get("text", "") for b in data.get("content", []))


def backend_openai(prompt):
    """Any provider exposing the OpenAI-style /chat/completions endpoint."""
    base = os.environ.get("TAWJIH_BASE_URL", "https://api.openai.com/v1")
    key = os.environ.get("TAWJIH_API_KEY", "")
    headers = {"Authorization": "Bearer " + key} if key else {}
    data = _post(base.rstrip("/") + "/chat/completions",
                 {"model": _model(),
                  "messages": [{"role": "user", "content": prompt}]}, headers)
    return data["choices"][0]["message"]["content"]


def backend_manual(prompt):
    """A person (or any chat model) plays the model: paste the answer."""
    bar = "=" * 60
    print("\n%s\n%s\n%s" % (bar, prompt, bar), file=sys.stderr)
    print("answer (finish with a line containing only '.'):", file=sys.stderr)
    lines = []
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == ".":
            break
        lines.append(line.rstrip("\n"))
    return "\n".join(lines).strip()


BACKENDS = {"dry": backend_dry, "manual": backend_manual,
            "openai": backend_openai, "ollama": backend_ollama,
            "anthropic": backend_anthropic}


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


def run_code(n, prev, shared, origin=""):
    ns = {"input": prev, "origin": origin, "vars": shared, "output": None}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(n.code, n.id + ".code", "exec"), ns)
    out = ns.get("output")
    return str(out) if out is not None else buf.getvalue().strip()


def machine_side(n):
    """Extract the machine side of a file: (instructions, code, digest).
    The digest identifies the exact machine-side version used in a run."""
    ins, code = n.instructions.strip("\n"), n.code.strip("\n")
    blob = ins + "\n" + MARK.format("code") + "\n" + code
    return ins, code, hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------- flow
# A path is written with connecting words between files:
#   A TO B                 A's output goes to B
#   A TO B AND C           ... to both B and C (branching)
#   B AND C TO D           D receives both outputs (joining)
#   A TO B IF word         only when A's output contains the word
#   A TO B IF NOT word     only when it does not
#   A AMONG B C D          A's output names one of them; only that one runs
#   A TO D THROUGH B C     A -> B -> C -> D
class FlowError(Exception):
    pass


KEYWORDS = {"TO": "TO", "IF": "IF", "AND": "AND", "AMONG": "AMONG",
            "THROUGH": "THROUGH", "NOT": "NOT",
            "إلى": "TO", "الى": "TO", "إذا": "IF", "اذا": "IF", "و": "AND",
            "بين": "AMONG", "عبر": "THROUGH", "ليس": "NOT"}


def _kw(tok):
    return KEYWORDS.get(tok.upper())


class Flow:
    def __init__(self):
        self.nodes, self.edges, self.lines, self._groups = [], [], [], 0

    def add_node(self, nid):
        if nid not in self.nodes:
            self.nodes.append(nid)

    def add_edge(self, src, dst, cond=None, among=None):
        if src == dst:
            raise FlowError("file %s cannot feed itself" % src)
        for e in self.edges:
            if e["src"] == src and e["dst"] == dst:
                if e["cond"] == cond and among is None and e["among"] is None:
                    return
                raise FlowError("%s is connected to %s twice in different "
                                "ways" % (src, dst))
        self.edges.append({"src": src, "dst": dst, "cond": cond,
                           "among": among})

    def statement(self, tokens, resolve):
        pos, norm = 0, []
        text = " ".join(tokens)

        def peek():
            return _kw(tokens[pos]) if pos < len(tokens) else None

        def take(k):
            nonlocal pos
            pos += 1
            norm.append(k)

        def ref():
            nonlocal pos
            if pos >= len(tokens):
                raise FlowError("a file is missing at the end of: " + text)
            if _kw(tokens[pos]):
                raise FlowError("expected a file but found '%s' in: %s"
                                % (tokens[pos], text))
            nid = resolve(tokens[pos])
            pos += 1
            self.add_node(nid)
            norm.append(nid)
            return nid

        def group():                       # A AND B AND C
            g = [ref()]
            while peek() == "AND":
                take("AND")
                g.append(ref())
            return g

        def listing():                     # A B C   (AND optional)
            g = [ref()]
            while pos < len(tokens) and peek() in (None, "AND"):
                if peek() == "AND":
                    take("AND")
                g.append(ref())
            return g

        cur = group()
        while pos < len(tokens):
            k = peek()
            if (k == "TO" and pos + 1 < len(tokens)
                    and _kw(tokens[pos + 1]) == "AMONG"):
                pos += 1
                k = "AMONG"
            if k == "AMONG":
                take("AMONG")
                if len(cur) != 1:
                    raise FlowError("AMONG needs exactly one deciding file "
                                    "before it in: " + text)
                cands = listing()
                if len(cands) < 2:
                    raise FlowError("AMONG needs at least two files to "
                                    "choose from in: " + text)
                self._groups += 1
                for d in cands:
                    self.add_edge(cur[0], d, None, self._groups)
                cur = cands
            elif k == "TO":
                take("TO")
                tgt, via, cond = group(), [], None
                while peek() in ("THROUGH", "IF"):
                    if peek() == "THROUGH":
                        take("THROUGH")
                        via = listing()
                    else:
                        take("IF")
                        neg = peek() == "NOT"
                        if neg:
                            take("NOT")
                        if pos >= len(tokens):
                            raise FlowError("IF needs a word in: " + text)
                        if "#" in tokens[pos]:
                            raise FlowError("the IF word cannot contain '#'")
                        cond = [neg, tokens[pos]]
                        norm.append(shlex.quote(tokens[pos]))
                        pos += 1
                if via:
                    for src in cur:
                        self.add_edge(src, via[0], cond)
                    for x, y in zip(via, via[1:]):
                        self.add_edge(x, y)
                    for dst in tgt:
                        self.add_edge(via[-1], dst)
                else:
                    for src in cur:
                        for dst in tgt:
                            self.add_edge(src, dst, cond)
                cur = tgt
            else:
                raise FlowError("expected TO or AMONG but found '%s' in: %s"
                                % (tokens[pos], text))
        self.lines.append(norm)

    def order(self):
        """Execution order: every file after all the files that feed it."""
        indeg = {n: 0 for n in self.nodes}
        for e in self.edges:
            indeg[e["dst"]] += 1
        out, left = [], list(self.nodes)
        while left:
            ready = [n for n in left if indeg[n] == 0]
            if not ready:
                raise FlowError("the connections form a loop through: "
                                + ", ".join(left))
            n = ready[0]
            left.remove(n)
            out.append(n)
            for e in self.edges:
                if e["src"] == n:
                    indeg[e["dst"]] -= 1
        return out

    def incoming(self, nid):
        return [e for e in self.edges if e["dst"] == nid]

    def terminals(self):
        return [n for n in self.nodes
                if not any(e["src"] == n for e in self.edges)]


def split_statements(tokens):
    """Statements on a command line are separated by ',' or '/'."""
    out, cur = [], []
    for t in tokens:
        end = t in (",", "/") or t.endswith(",")
        t = "" if t in (",", "/") else t.rstrip(",")
        if t:
            cur.append(t)
        if end and cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def build_flow(statements, resolve):
    """Statements without any connecting word are a plain sequence."""
    flow = Flow()
    if not any(_kw(t) for st in statements for t in st):
        ids = [resolve(t) for st in statements for t in st]
        for nid in ids:
            flow.add_node(nid)
            flow.lines.append([nid])
        if len(set(ids)) != len(ids):
            raise FlowError("a file appears twice in the sequence")
        for x, y in zip(ids, ids[1:]):
            flow.add_edge(x, y)
    else:
        for st in statements:
            flow.statement(st, resolve)
    if not flow.nodes:
        raise FlowError("no files given")
    return flow


def write_flow(store, name, flow):
    store.pdir.mkdir(exist_ok=True)
    lines = []
    for norm in flow.lines:
        titles = [store.nodes[t].title for t in norm if t in store.nodes]
        lines.append("%s  # %s" % (" ".join(norm), ", ".join(titles)))
    f = store.pdir / (name + ".path")
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


def edge_text(store, e):
    t = "%s %s" % (store.number(e["src"]), store.nodes[e["src"]].title)
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
                     "instructions": bool(ins.strip()),
                     "code": bool(code.strip()), "machine_sha": digest,
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
        print("%3d  %-14s %-10s [%s]  %-18s %s" % (
            p["step"], p["tree_title"], p["number"], p["id"], kind,
            p["title"]), file=out)
        for src in p["from"]:
            print("       <- %s" % src, file=out)
        if not p["from"]:
            print("       <- (the original request)", file=out)


def choose_among(store, output, candidates):
    """The candidate named first in the output (by id, number or title)."""
    low, best = output.casefold(), None
    for nid in candidates:
        n = store.nodes[nid]
        hits = [m.start() for m in re.finditer(
            r"(?<![\d.])%s(?![\d.])" % re.escape(store.number(nid)), output)]
        for key in (nid, n.title.strip()):
            if key:
                m = re.search(r"(?<!\w)%s(?!\w)" % re.escape(key.casefold()),
                              low)
                if m:
                    hits.append(m.start())
        if hits and (best is None or min(hits) < best[0]):
            best = (min(hits), nid)
    return best[1] if best else None


def _stamped(folder, name, suffix):
    folder.mkdir(exist_ok=True)
    stamp, k = time.strftime("%Y%m%d-%H%M%S"), 1
    f = folder / ("%s-%s%s" % (name, stamp, suffix))
    while f.exists():
        k += 1
        f = folder / ("%s-%s-%d%s" % (name, stamp, k, suffix))
    return f


def run_flow(store, name, text, backend, allow_code, flow):
    """Execute the machine sides of the files of a flow. A file runs after
    the files that feed it and receives their outputs; a file none of whose
    connections is active is skipped. Writes the log and the result to runs/
    and a record of the operation to history/."""
    plan = plan_steps(store, flow)
    trees = []
    for p in plan:
        if p["tree"] not in [t["id"] for t in trees]:
            trees.append({"id": p["tree"], "title": p["tree_title"]})
    tree_of = {p["id"]: p["tree"] for p in plan}
    log = ["# Run of path '%s'" % name]
    logfile = _stamped(store.rdir, name, ".md")
    resfile = logfile.with_name(logfile.stem + ".result.txt")
    record = {"tawjih": __version__, "name": name,
              "started": time.strftime("%Y-%m-%d %H:%M:%S"),
              "backend": backend, "allow_code": bool(allow_code),
              "input_chars": len(text),
              "input_sha": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
              "trees": trees, "cross_tree": len(trees) > 1, "handoffs": 0,
              "flow": [" ".join(l) for l in flow.lines],
              "connections": [{"from": e["src"], "to": e["dst"],
                               "if": e["cond"], "among": e["among"]}
                              for e in flow.edges],
              "steps": plan, "status": "running", "log": str(logfile),
              "result": ""}

    def document():
        record["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        rec = _stamped(store.root / "history", "production-" + name, ".json")
        rec.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        return rec

    def fail(p, n, what, e):
        log.append("### FAILED (%s)\n%s: %s" % (what, type(e).__name__, e))
        logfile.write_text("\n".join(log) + "\n", encoding="utf-8")
        p["status"] = "failed"
        record.update(status="failed", failed_step=p["step"],
                      error="%s: %s: %s" % (what, type(e).__name__, e))
        rec = document()
        die("step %d [%s] %s failed in %s: %s: %s (log: %s, record: %s)"
            % (p["step"], n.id, n.title, what, type(e).__name__, e, logfile,
               rec))

    outputs, chosen, shared = {}, {}, {}

    def active(e):
        if e["src"] not in outputs:
            return False
        if e["among"] is not None:
            return chosen.get((e["src"], e["among"])) == e["dst"]
        if e["cond"]:
            neg, word = e["cond"]
            return (word.casefold() in outputs[e["src"]].casefold()) != neg
        return True

    for p in plan:
        n = store.nodes[p["id"]]
        ins, code, _ = machine_side(n)
        head = "step %d/%d  %s [%s] %s" % (p["step"], len(plan), p["number"],
                                           n.id, n.title)
        log.append("\n## Step %d: %s [%s] (tree: %s, machine %s)" % (
            p["step"], n.title, n.id, p["tree_title"], p["machine_sha"]))
        incoming = flow.incoming(n.id)
        if not incoming:
            prev, feeders = text, []
        else:
            feeders = [e["src"] for e in incoming if active(e)]
            if not feeders:
                p["status"] = "skipped"
                print(head + "  (skipped: no active connection)",
                      file=sys.stderr)
                log.append("skipped: no active connection")
                continue
            if len(feeders) == 1:
                prev = outputs[feeders[0]]
            else:
                prev = "\n\n".join("## from: %s [%s]\n%s" % (
                    store.nodes[f].title, f, outputs[f]) for f in feeders)
        print(head, file=sys.stderr)
        for f in feeders:
            log.append("input from: %s [%s]" % (store.nodes[f].title, f))
            if tree_of[f] != p["tree"]:
                record["handoffs"] += 1
                print("  handoff from tree: %s" % store.ancestors(f)[0].title,
                      file=sys.stderr)
        p["received_from"] = feeders
        code_out, out = "", prev
        if code.strip():
            if allow_code:
                try:
                    code_out = run_code(n, prev, shared, text)
                except Exception as e:
                    fail(p, n, "code", e)
            else:
                print("  code skipped (pass --allow-code to execute it)",
                      file=sys.stderr)
        if ins.strip():
            prompt = build_prompt(store, n, prev, code_out, text)
            print("  prompt: %d characters" % len(prompt), file=sys.stderr)
            log.append("### Prompt\n" + prompt)
            try:
                out = BACKENDS[backend](prompt)
            except Exception as e:
                fail(p, n, "model call", e)
        elif code_out:
            out = code_out
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
                        for c in cands)))
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
    record.update(status="done", result=str(resfile), final_steps=ends)
    rec = document()
    print("log: %s\nresult: %s\nrecord: %s" % (logfile, resfile, rec),
          file=sys.stderr)
    return result


# ------------------------------------------------------------------ commands
def cmd_init(a):
    root = Path(a.project)
    for d in ("files", "paths", "runs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    print("project ready in %s" % root.resolve())


def cmd_new(a):
    st = Store(a.project)
    parent = st.resolve(a.parent) if a.parent else ""
    if a.level:
        level = parse_level(a.level)
    else:
        level = st.nodes[parent].level + 1 if parent else 1
    if level > len(LEVELS):
        die("the parent is already at the deepest level (part)")
    if parent and level <= st.nodes[parent].level:
        die("level must be deeper than the parent's level")
    n = st.new_node(a.title, level, parent)
    print("%s  [%s]  %s" % (st.number(n.id), n.id, st.file_of(n.id)))
    if a.open:
        open_editors(st, n.id)


def cmd_tree(a):
    st = Store(a.project)

    def walk(pid, depth):
        for n in st.children(pid):
            flags = "".join(c if getattr(n, s).strip() else "-"
                            for c, s in zip("HICR", SECTIONS))
            print("%s%-8s [%s] %-8s %s  %s" % ("  " * depth, st.number(n.id),
                  n.id, LEVELS[n.level - 1], flags, n.title))
            walk(n.id, depth + 1)
    walk("", 0)


def cmd_show(a):
    st = Store(a.project)
    nid = st.resolve(a.ref)
    print("# %s  number %s" % (st.file_of(nid), st.number(nid)))
    print(st.nodes[nid].dump(), end="")


def cmd_link(a):
    st = Store(a.project)
    src, dst = st.resolve(a.src), st.resolve(a.dst)
    if src == dst:
        die("a file cannot link to itself")
    n = st.nodes[src]
    n.links = [(t, ty) for t, ty in n.links if t != dst] + [(dst, a.type)]
    st.save(n)
    print("linked %s -> %s (%s)" % (src, dst, a.type))


def cmd_check(a):
    errors, warns = Store(a.project).check()
    for w in warns:
        print("warning: " + w)
    for e in errors:
        print("ERROR:   " + e)
    print("%d error(s), %d warning(s)" % (len(errors), len(warns)))
    sys.exit(1 if errors else 0)


def _flow_from_args(st, name, refs):
    """Build and save a flow from command-line words, or load the saved one."""
    if not re.fullmatch(r"[\w\-]+", name):
        die("path names may contain only letters, digits, '_' and '-'")
    try:
        if refs:
            flow = build_flow(split_statements(refs), st.resolve)
            flow.order()
            write_flow(st, name, flow)
        else:
            flow = st.load_flow(name)
            flow.order()
    except FlowError as e:
        die("path %s: %s" % (name, e))
    return flow


def cmd_path(a):
    st = Store(a.project)
    flow = _flow_from_args(st, a.name, a.refs)
    print_plan(st, a.name, plan_steps(st, flow), sys.stdout)


def cmd_run(a):
    st = Store(a.project)
    flow = _flow_from_args(st, a.name, [])
    if not a.allow_code:
        coded = [st.nodes[n].title for n in flow.nodes
                 if st.nodes[n].code.strip()]
        if coded:
            die("not started: these steps have code, pass --allow-code after "
                "reading it: " + ", ".join(coded))
    text = a.input
    if a.input_file:
        text = Path(a.input_file).read_text(encoding="utf-8")
    print(run_flow(st, a.name, text, a.backend, a.allow_code, flow))


def cmd_trees(a):
    """List the available trees (one per root file)."""
    st = Store(a.project)
    roots = st.children("")
    if not roots:
        print("no trees yet (create one with: new TITLE)")
    for r in roots:
        files, machine, depth = [], 0, 0

        def walk(n, d):
            nonlocal machine, depth
            files.append(n)
            depth = max(depth, d)
            if n.instructions.strip() or n.code.strip():
                machine += 1
            for k in st.children(n.id):
                walk(k, d + 1)
        walk(r, 1)
        print("%-4s [%s]  %-24s %3d file(s), %d with a machine side, "
              "%d level(s)" % (st.number(r.id), r.id, r.title, len(files),
                               machine, depth))


def cmd_produce(a):
    """Cross-tree production: select files from any trees, connect them,
    extract their machine sides, run them, save and document."""
    st = Store(a.project)
    flow = _flow_from_args(st, a.name, a.refs)
    plan = plan_steps(st, flow)
    print_plan(st, a.name, plan, sys.stdout if a.plan else sys.stderr)
    problems = []
    for p in plan:
        if not p["instructions"] and not p["code"]:
            problems.append("step %d [%s] %s has an empty machine side"
                            % (p["step"], p["id"], p["title"]))
        elif p["code"] and not a.allow_code and not a.plan:
            problems.append("step %d [%s] %s has code; pass --allow-code "
                            "after reading it" % (p["step"], p["id"],
                                                  p["title"]))
    if problems:
        die("production not started:\n  " + "\n  ".join(problems))
    if a.plan:
        return
    text = a.input
    if a.input_file:
        text = Path(a.input_file).read_text(encoding="utf-8")
    print(run_flow(st, a.name, text, a.backend, a.allow_code, flow))


def cmd_import(a):
    """Split a text file on '#' headings into child files (human side)."""
    st = Store(a.project)
    parent = st.resolve(a.parent)
    level = st.nodes[parent].level + 1
    if level > len(LEVELS):
        die("the parent is already at the deepest level (part)")
    src = Path(a.file)
    chunks, title, body = [], src.stem, []
    for line in src.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            if "\n".join(body).strip():
                chunks.append((title, body))
            title, body = line.lstrip("#").strip() or src.stem, []
        else:
            body.append(line)
    if "\n".join(body).strip():
        chunks.append((title, body))
    for title, body in chunks:
        n = st.new_node(title, level, parent)
        n.human = "\n".join(body).strip("\n")
        st.save(n)
        print("%s  [%s]  %s" % (st.number(n.id), n.id, title))
    print("%d file(s) created" % len(chunks))


SIDE_NOTES = {
    "human": "The human side is prose: the expressive concept of the subject, "
             "written for a person. It is the basis the machine side is "
             "built from, and is not sent to the model when a path runs.",
    "instructions": "The instructions side is addressed to a language model "
                    "that, when the path runs, receives ONLY these "
                    "instructions, the original request and the previous "
                    "step's output. It will not see the human or reading "
                    "sides, so the instructions must be self-contained: put "
                    "every rule, formula or fact the model needs in them.",
    "code": "The code side is Python run as one step of a path. Available "
            "names: input (str, previous step's output), origin (str, the "
            "original request), vars (dict shared by all steps). The code "
            "cannot see the human or reading sides. "
            "Assign the step's result to a variable named output.",
    "reading": "The reading side holds passages from books and research for "
               "a person to read, each under a '## title' line followed by "
               "its '[source: ...]' line. Keep the source lines intact.",
}


def strip_fence(text):
    t = text.strip()
    m = re.fullmatch(r"```[^\n]*\n(.*?)\n?```", t, re.S)
    return m.group(1) if m else t


def cmd_edit(a):
    """Ask an AI provider to rewrite one side of a file; review before saving."""
    st = Store(a.project)
    nid = st.resolve(a.ref)
    n = st.nodes[nid]
    parts = ["You are editing one side of a three-sided knowledge file. Reply "
             "with the complete new content of the '%s' side only: no preface, "
             "no explanation, no code fences.\n%s" % (a.side, SIDE_NOTES[a.side])]
    above = [x for x in st.ancestors(nid)[:-1] if x.human.strip()]
    if above:
        parts.append("# Tree context")
        for x in above:
            parts.append("## [%s] %s\n%s" % (LEVELS[x.level - 1], x.title, x.human))
    parts.append("# File: %s [%s]" % (n.title, LEVELS[n.level - 1]))
    for sec in SECTIONS:
        parts.append("## Current %s side\n%s" % (sec, getattr(n, sec) or "(empty)"))
    if a.source:
        parts.append("# Source material\n"
                     + Path(a.source).read_text(encoding="utf-8"))
    parts.append("# Request\n" + a.ask)
    prompt = "\n\n".join(parts)
    if a.backend == "dry":
        print(prompt)
        print("\n[dry run: %d characters; choose --backend to send it]"
              % len(prompt), file=sys.stderr)
        return
    try:
        new = strip_fence(BACKENDS[a.backend](prompt)).strip("\n")
    except Exception as e:
        die("model call failed: %s: %s" % (type(e).__name__, e))
    old = getattr(n, a.side).strip("\n")
    if not new.strip():
        die("the model returned nothing; file unchanged")
    if any(l.strip() == MARK.format(sec) for l in new.splitlines() for sec in SECTIONS):
        die("the answer contains a section marker line; file unchanged")
    if new == old:
        print("no change")
        return
    for line in difflib.unified_diff(old.splitlines(), new.splitlines(),
                                     "before", "after", lineterm=""):
        print(line)
    if a.side == "code":
        try:
            compile(new, nid + ".code", "exec")
        except SyntaxError as e:
            die("the proposed code has a syntax error (%s); file unchanged" % e)
    if not a.yes:
        print("save this change? [y/N] ", end="", flush=True)
        if sys.stdin.readline().strip().lower() not in ("y", "yes", "نعم"):
            print("not saved")
            return
    bak = _backup(st, n)
    setattr(n, a.side, new)
    st.save(n)
    print("saved %s (previous version: %s)" % (st.file_of(nid), bak))


# ------------------------------------------------------------ side editors
GUI_PARALLEL = {"notepad", "notepad++", "gedit", "kate", "mousepad", "xed",
                "pluma"}                       # one window per side
SPLIT_FLAGS = {"vim": ["-O"], "nvim": ["-O"]}  # sides shown side by side
NEEDS_WAIT = {"code", "subl"}


def editor_command(arg):
    spec = (arg or os.environ.get("TAWJIH_EDITOR") or os.environ.get("VISUAL")
            or os.environ.get("EDITOR"))
    if spec:
        cmd = [c.strip('"') for c in shlex.split(spec, posix=(os.name != "nt"))]
    elif os.name == "nt":
        cmd = ["notepad"]
    else:
        found = next((e for e in ("nano", "vim", "vi") if shutil.which(e)), None)
        if not found:
            die("no editor found; set TAWJIH_EDITOR or pass --editor")
        cmd = [found]
    name = Path(cmd[0]).name.lower()
    if name.endswith(".exe"):
        name = name[:-4]
    if name in NEEDS_WAIT and "--wait" not in cmd and "-w" not in cmd:
        cmd.append("--wait")
    return cmd, name


def _backup(st, n):
    hdir = st.root / "history"
    hdir.mkdir(exist_ok=True)
    stamp, k = time.strftime("%Y%m%d-%H%M%S"), 1
    bak = hdir / ("%s-%s.md" % (n.id, stamp))
    while bak.exists():
        k += 1
        bak = hdir / ("%s-%s-%d.md" % (n.id, stamp, k))
    bak.write_text(n.dump(), encoding="utf-8")
    return bak


def open_editors(st, nid, editor=None, force_reading=False):
    """Open one editor per side: human, machine, and reading when the file
    has deposited-book passages. Changes are saved when the editors close."""
    n = st.nodes[nid]
    m_ins, m_code = MARK.format("instructions"), MARK.format("code")
    sides = [("human", n.human),
             ("machine", "%s\n%s\n%s\n%s" % (m_ins, n.instructions, m_code,
                                             n.code))]
    if n.reading.strip() or force_reading:
        sides.append(("reading", n.reading))
    wdir = st.root / "work" / nid
    wdir.mkdir(parents=True, exist_ok=True)
    enc = "utf-8-sig" if os.name == "nt" else "utf-8"   # BOM helps Notepad
    files = []
    for name, text in sides:
        f = wdir / (name + ".txt")
        f.write_text(text.strip("\n") + "\n", encoding=enc)
        files.append(f)
    cmd, ename = editor_command(editor)
    print("%s [%s] %s: opening %s with '%s'" % (
        st.number(nid), nid, n.title, ", ".join(x[0] for x in sides),
        " ".join(cmd)))
    started = time.time()
    try:
        if ename in GUI_PARALLEL:
            procs = [subprocess.Popen(cmd + [str(f)]) for f in files]
            for p in procs:
                p.wait()
        else:
            subprocess.call(cmd + SPLIT_FLAGS.get(ename, [])
                            + [str(f) for f in files])
    except OSError as e:
        shutil.rmtree(str(wdir), ignore_errors=True)
        die("cannot start editor '%s': %s" % (cmd[0], e))
    if time.time() - started < 1.0:
        print("The editor returned at once. Press Enter here after you have "
              "saved and closed it. ", end="", flush=True)
        sys.stdin.readline()
    new = {}
    for (name, _), f in zip(sides, files):
        try:
            new[name] = f.read_text(encoding="utf-8-sig").strip("\n")
        except UnicodeDecodeError:
            die("%s was not saved as UTF-8; nothing changed (your text is "
                "still in %s)" % (f.name, f))
    marks = {MARK.format(sec) for sec in SECTIONS}
    for name in ("human", "reading"):
        if any(l.strip() in marks for l in new.get(name, "").splitlines()):
            die("the %s side contains a '=== ... ===' marker line; nothing "
                "changed (your text is still in %s)" % (name, wdir))
    ins, code, cur = [], [], None
    for line in new["machine"].splitlines():
        if line.strip() == m_ins:
            cur = ins
        elif line.strip() == m_code:
            cur = code
        else:
            (ins if cur is None else cur).append(line)
    result = {"human": new["human"],
              "instructions": "\n".join(ins).strip("\n"),
              "code": "\n".join(code).strip("\n"),
              "reading": new.get("reading", n.reading.strip("\n"))}
    changed = [k for k in SECTIONS if result[k] != getattr(n, k).strip("\n")]
    if changed:
        if "code" in changed and result["code"].strip():
            try:
                compile(result["code"], nid + ".code", "exec")
            except SyntaxError as e:
                print("warning: the code has a syntax error (%s); saved "
                      "anyway, fix it before running a path" % e)
        bak = _backup(st, n)
        for k in changed:
            setattr(n, k, result[k])
        st.save(n)
        print("saved: %s (previous version: %s)" % (", ".join(changed), bak))
    else:
        print("no changes")
    shutil.rmtree(str(wdir), ignore_errors=True)
    try:
        (st.root / "work").rmdir()
    except OSError:
        pass


def cmd_open(a):
    st = Store(a.project)
    open_editors(st, st.resolve(a.ref), a.editor, a.reading)


# ------------------------------------------------ library: deposit + filter
def _library(st):
    ldir = st.root / "library"
    ldir.mkdir(exist_ok=True)
    idx = ldir / "index.json"
    books = json.loads(idx.read_text(encoding="utf-8")) if idx.is_file() else {}
    return ldir, idx, books


def _book(st, ref):
    ldir, _, books = _library(st)
    hits = [b for b in books if b.startswith(ref)]
    if len(hits) != 1:
        die("no single deposited book matches '%s' (see: library)" % ref)
    return ldir, hits[0], books[hits[0]]


def cmd_deposit(a):
    """Store a text book in the library; identical content is refused."""
    st = Store(a.project)
    src = Path(a.file)
    try:
        text = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        die("not a UTF-8 text file (convert PDF or Word to plain text first)")
    if not text.strip():
        die("the file is empty")
    sha = hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()
    ldir, idx, books = _library(st)
    for b, info in books.items():
        if info["sha256"] == sha:
            die("same content already deposited as %s (%s)" % (b, info["name"]))
    bid = sha[:8]
    (ldir / (bid + ".txt")).write_text(text, encoding="utf-8")
    books[bid] = {"name": src.name, "sha256": sha, "chars": len(text),
                  "date": time.strftime("%Y-%m-%d")}
    idx.write_text(json.dumps(books, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("deposited %s  %s  (%d characters)" % (bid, src.name, len(text)))


def cmd_library(a):
    st = Store(a.project)
    ldir, _, books = _library(st)
    for b, info in sorted(books.items(), key=lambda x: x[1]["date"]):
        rep = ldir / (b + ".filter.json")
        state = "deposited"
        if rep.is_file():
            done = json.loads(rep.read_text(encoding="utf-8")).get("applied")
            state = "accepted" if done else "filtered, awaiting accept"
        print("%s  %s  %8d chars  %s  [%s]" % (b, info["date"], info["chars"],
                                               info["name"], state))


def split_chunks(text, name, size):
    lines = text.splitlines()
    chunks = []
    if any(l.startswith("#") for l in lines):
        title, body = name, []
        for line in lines:
            if line.startswith("#"):
                if "\n".join(body).strip():
                    chunks.append((title, "\n".join(body).strip()))
                title, body = line.lstrip("#").strip() or name, []
            else:
                body.append(line)
        if "\n".join(body).strip():
            chunks.append((title, "\n".join(body).strip()))
        return chunks
    cur = ""
    for para in re.split(r"\n\s*\n", text):
        if not para.strip():
            continue
        if cur and len(cur) + len(para) > size:
            chunks.append(("%s (%d)" % (name, len(chunks) + 1), cur))
            cur = ""
        cur = (cur + "\n\n" + para.strip()).strip()
    if cur:
        chunks.append(("%s (%d)" % (name, len(chunks) + 1), cur))
    return chunks


FILTER_HEAD = (
    "You are filtering a deposited book for a personal knowledge tree. Decide "
    "whether the passage below belongs on the reading side of one of the "
    "listed files. Keep it only if it carries real subject content about "
    "that file's subject, and choose the most specific file. Reject "
    "publisher matter, acknowledgements, advertisements, tables of contents "
    "and anything off-topic.%s\nReply with JSON only: {\"keep\": true or "
    "false, \"target\": \"id of the file it belongs to, or empty\", "
    "\"title\": \"short title for the passage\", \"reason\": \"one "
    "sentence\"}")


def cmd_filter(a):
    """Ask the model, passage by passage, what to keep and where it belongs.
    Writes a review report only; the tree is not touched until 'accept'."""
    st = Store(a.project)
    ldir, bid, info = _book(st, a.book)
    text = (ldir / (bid + ".txt")).read_text(encoding="utf-8")
    parent = st.resolve(a.parent) if a.parent else ""
    rows = []

    def listing(pid):
        for n in st.children(pid):
            rows.append(n)
            listing(n.id)
    if parent:
        rows.append(st.nodes[parent])
    listing(parent)
    if not rows:
        die("the tree is empty; create files first")
    ok_targets = {n.id for n in rows}
    table = "\n".join("%s | %s | %s | %s" % (n.id, st.number(n.id),
                       LEVELS[n.level - 1], n.title) for n in rows)
    head = FILTER_HEAD % ((" Extra criteria: " + a.ask) if a.ask else "")
    chunks = split_chunks(text, Path(info["name"]).stem, a.size)

    def prompt_for(title, body):
        return ("%s\n\n# Tree files (id | number | level | title)\n%s\n\n"
                "# Passage: %s\n%s" % (head, table, title, body))
    if a.backend == "dry":
        print(prompt_for(*chunks[0]))
        print("\n[dry run: %d passages = %d model calls, %d characters in the "
              "book]" % (len(chunks), len(chunks), len(text)), file=sys.stderr)
        return
    items = []
    for i, (title, body) in enumerate(chunks, 1):
        try:
            ans = BACKENDS[a.backend](prompt_for(title, body))
        except Exception as e:
            die("model call failed at passage %d: %s: %s (nothing was written)"
                % (i, type(e).__name__, e))
        item = {"n": i, "status": "ok", "keep": False, "target": "",
                "title": title, "reason": "", "chars": len(body), "text": body}
        try:
            d = json.loads(re.search(r"\{.*\}", ans, re.S).group(0))
            item["keep"] = d.get("keep") is True
            item["target"] = str(d.get("target") or "")
            item["title"] = str(d.get("title") or title)
            item["reason"] = str(d.get("reason") or "")
            if item["keep"] and item["target"] not in ok_targets:
                item.update(status="review", keep=False,
                            reason="model chose an invalid target '%s'. %s"
                                   % (item["target"], item["reason"]))
        except Exception:
            item.update(status="review", reason="unreadable model reply: "
                                                + ans.strip()[:200])
        items.append(item)
        if item["status"] == "review":
            verdict = "REVIEW"
        elif item["keep"]:
            verdict = "KEEP -> " + st.number(item["target"])
        else:
            verdict = "DROP"
        print("%3d  %-16s %s  | %s" % (i, verdict, item["title"], item["reason"]))
    rep = ldir / (bid + ".filter.json")
    rep.write_text(json.dumps({"book": bid, "name": info["name"],
                               "applied": False, "items": items},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    kept = sum(1 for x in items if x["keep"])
    review = sum(1 for x in items if x["status"] == "review")
    print("%d kept, %d dropped, %d need review. Report: %s"
          % (kept, len(items) - kept - review, review, rep))
    print("Edit the report if you disagree, then run: accept %s" % bid)


def cmd_accept(a):
    """Append the kept passages of a filter report to the reading side of
    their target files."""
    st = Store(a.project)
    ldir, bid, _ = _book(st, a.book)
    rep = ldir / (bid + ".filter.json")
    if not rep.is_file():
        die("this book has not been filtered yet")
    data = json.loads(rep.read_text(encoding="utf-8"))
    if data.get("applied"):
        die("this report was already accepted")
    made = skipped = 0
    backed_up = set()
    for item in data["items"]:
        if item.get("keep") is not True:
            continue
        t = st.nodes.get(item.get("target", ""))
        if t is None:
            print("skipped passage %d: target '%s' does not exist"
                  % (item["n"], item.get("target", "")))
            skipped += 1
            continue
        marks = {MARK.format(sec) for sec in SECTIONS}
        body = "\n".join(("> " + l) if l.strip() in marks else l
                         for l in item["text"].splitlines())
        block = "## %s\n[source: %s, book %s, passage %d]\n%s" % (
            item["title"], data["name"], bid, item["n"], body)
        if t.id not in backed_up:
            _backup(st, t)
            backed_up.add(t.id)
        t.reading = (t.reading.strip("\n") + "\n\n" + block).strip("\n")
        st.save(t)
        made += 1
        print("%s  [%s]  %s  <-  %s" % (st.number(t.id), t.id, t.title,
                                        item["title"]))
    data["applied"] = True
    rep.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("%d passage(s) added to reading sides, %d skipped" % (made, skipped))


def cmd_read(a):
    """Print the reading side of a file, or of a file and its whole subtree."""
    st = Store(a.project)
    nid = st.resolve(a.ref)

    def emit(n):
        if n.reading.strip():
            print("# %s  %s [%s]\n\n%s\n" % (st.number(n.id), n.title, n.id,
                                              n.reading))
            return 1
        return 0

    def walk(pid):
        return sum(emit(k) + walk(k.id) for k in st.children(pid))
    count = emit(st.nodes[nid]) + (walk(nid) if a.all else 0)
    if not count:
        print("nothing to read here yet (deposit a book, then filter and accept)")


def main():
    for s in (sys.stdout, sys.stderr, sys.stdin):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    p = argparse.ArgumentParser(prog="tawjih.py", description=__doc__.split("\n")[0])
    p.add_argument("-p", "--project", default=".", help="project folder")
    p.add_argument("--version", action="version",
                   version="tawjih " + __version__)
    sub = p.add_subparsers(dest="cmd")
    sub.required = True

    s = sub.add_parser("init", help="create a project")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("new", help="create a two-sided file")
    s.add_argument("title")
    s.add_argument("--parent", help="id or number of the parent file")
    s.add_argument("--level", help="1-6 or name; default: parent level + 1")
    s.add_argument("--open", action="store_true",
                   help="open the editors on the new file")
    s.set_defaults(fn=cmd_new)

    s = sub.add_parser("open", help="open an editor for each side of a file")
    s.add_argument("ref")
    s.add_argument("--editor", help="editor command (default: TAWJIH_EDITOR, "
                   "VISUAL, EDITOR, then notepad on Windows or nano/vim)")
    s.add_argument("--reading", action="store_true",
                   help="open the reading side even when it is empty")
    s.set_defaults(fn=cmd_open)

    s = sub.add_parser("tree", help="show all trees (H/I/C/R = sides filled)")
    s.set_defaults(fn=cmd_tree)

    s = sub.add_parser("show", help="print one file and its location")
    s.add_argument("ref")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("link", help="link a file to any file, any tree")
    s.add_argument("src")
    s.add_argument("dst")
    s.add_argument("--type", default="related")
    s.set_defaults(fn=cmd_link)

    s = sub.add_parser("check", help="validate trees, links and paths")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("path", help="define (with refs) or show a path")
    s.add_argument("name")
    s.add_argument("refs", nargs="*", help="files (ids or numbers) joined by "
                   "TO, AND, IF, AMONG, THROUGH; ',' or '/' between statements")
    s.set_defaults(fn=cmd_path)

    s = sub.add_parser("run", help="execute a path step by step")
    s.add_argument("name")
    s.add_argument("--input", default="")
    s.add_argument("--input-file")
    s.add_argument("--backend", choices=sorted(BACKENDS), default="dry")
    s.add_argument("--allow-code", action="store_true",
                   help="execute Python in the code sections (not sandboxed)")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("trees", help="list the available trees")
    s.set_defaults(fn=cmd_trees)

    s = sub.add_parser("produce", help="cross-tree production: select, "
                       "order, extract machine sides, run, save, document")
    s.add_argument("name")
    s.add_argument("refs", nargs="*", help="files from any tree (ids or "
                   "numbers) joined by TO, AND, IF, AMONG, THROUGH; ',' or "
                   "'/' between statements; omit to reuse NAME")
    s.add_argument("--plan", action="store_true",
                   help="show and validate the plan without running it")
    s.add_argument("--input", default="")
    s.add_argument("--input-file")
    s.add_argument("--backend", choices=sorted(BACKENDS), default="dry")
    s.add_argument("--allow-code", action="store_true",
                   help="execute Python in the code sections (not sandboxed)")
    s.set_defaults(fn=cmd_produce)

    s = sub.add_parser("import", help="split a text file on '#' headings into child "
                       "files (human side), without a model")
    s.add_argument("file")
    s.add_argument("--parent", required=True)
    s.set_defaults(fn=cmd_import)

    s = sub.add_parser("edit", help="have an AI provider rewrite one side")
    s.add_argument("ref")
    s.add_argument("--side", choices=SECTIONS, required=True)
    s.add_argument("--ask", required=True, help="what the model should do")
    s.add_argument("--source", help="text file given to the model as material")
    s.add_argument("--backend", choices=sorted(BACKENDS), default="dry")
    s.add_argument("--yes", action="store_true", help="save without asking")
    s.set_defaults(fn=cmd_edit)

    s = sub.add_parser("deposit", help="add a text book to the library")
    s.add_argument("file")
    s.set_defaults(fn=cmd_deposit)

    s = sub.add_parser("library", help="list deposited books and their state")
    s.set_defaults(fn=cmd_library)

    s = sub.add_parser("filter", help="have an AI provider screen a book")
    s.add_argument("book", help="book id (or its first characters)")
    s.add_argument("--parent", help="limit targets to this file's subtree")
    s.add_argument("--ask", help="extra filtering criteria")
    s.add_argument("--size", type=int, default=2000,
                   help="passage size when the book has no '#' headings")
    s.add_argument("--backend", choices=sorted(BACKENDS), default="dry")
    s.set_defaults(fn=cmd_filter)

    s = sub.add_parser("accept", help="add the kept passages to reading sides")
    s.add_argument("book")
    s.set_defaults(fn=cmd_accept)

    s = sub.add_parser("read", help="read the reading side of a file")
    s.add_argument("ref")
    s.add_argument("--all", action="store_true", help="include the subtree")
    s.set_defaults(fn=cmd_read)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
