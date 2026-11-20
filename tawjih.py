#!/usr/bin/env python3
"""tawjih - personal LLM command-directing (توجيه أوامر الآلة).

A project is a folder of three-sided files arranged as trees:
    human side   : the expressive human concept (context)
    machine side : instructions for the model + optional Python code
    reading side : filtered passages of deposited books and research
A path connects files across trees into a directed acyclic workflow (DAG);
running it executes their machine sides step by step.
The human and reading sides are build-time material and never reach the model
at run time.

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
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from pathlib import Path

__version__ = "0.8.0"

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


CONTRACT_HELP = "use json, json(key, ...) or text"


def _parse_contract(spec):
    """Parse a contract: 'json', 'json(a, b)' or 'text'. Anything else is
    returned as ('unknown', [spec]) so that a typo is reported, not ignored."""
    spec = (spec or "").strip()
    if not spec:
        return None
    m = re.fullmatch(r"json\s*(?:\(([^)]*)\))?", spec, re.I)
    if m:
        keys = [k.strip() for k in (m.group(1) or "").split(",") if k.strip()]
        return ("json", keys)
    if spec.lower() == "text":
        return ("text", [])
    return ("unknown", [spec])


def _json_in(text):
    """The JSON object in an answer: the whole text, the inside of a code
    fence, or the span from the first '{' to the last '}' (models often add
    a sentence before or after). Returns (object, its JSON text)."""
    t = text.strip()
    cands = [t]
    m = re.search(r"```[^\n]*\n(.*?)\n?```", t, re.S)
    if m:
        cands.append(m.group(1).strip())
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        cands.append(t[i:j + 1])
    for c in cands:
        try:
            return json.loads(c), c
        except ValueError:
            continue
    return None, None


FALSE_WORDS = {"", "false", "no", "0", "none", "null", "لا", "خطأ"}


def condition_holds(text, word):
    """Whether 'IF word' holds for a step's output. When the output is a JSON
    object with 'word' as a top-level key, the key's value decides: true,
    non-zero and non-empty hold; false, 0, null, empty, and the strings
    "false", "no", "0", "none", "null", "لا" and "خطأ" do not. Otherwise
    the word must appear in the text (case-insensitive)."""
    obj, _ = _json_in(text)
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.casefold() == word.casefold():
                if isinstance(v, str):      # models write "false" as text too
                    return v.strip().casefold() not in FALSE_WORDS
                return bool(v)
    return word.casefold() in text.casefold()


def conform(text, spec, label):
    """Check text against a contract. Returns (text to pass on, error).
    For a JSON contract the text passed on is the bare JSON, without the
    fence or the sentences around it."""
    parsed = _parse_contract(spec)
    if not parsed:
        return text, None
    kind, keys = parsed
    if kind == "unknown":
        return text, "%s has an unknown contract '%s' (%s)" % (
            label, keys[0], CONTRACT_HELP)
    if kind == "text":
        return text, None
    obj, raw = _json_in(text)
    if raw is None:
        return text, "%s expected JSON and none was found" % label
    if not isinstance(obj, dict):
        return text, "%s expected a JSON object" % label
    missing = [k for k in keys if k not in obj]
    if missing:
        return text, "%s is missing JSON key(s): %s" % (label, ", ".join(missing))
    return raw, None


def _validate_contract(text, spec, label):
    """The error of conform(), or None."""
    return conform(text, spec, label)[1]


# ---------------------------------------------------------------- file model
class Node:
    def __init__(self, id="", title="", level=1, parent="", order=0,
                 links=None, model="", expects="", produces="", built_sha="",
                 human="", instructions="", code="", reading="", tags=None,
                 retries="", fallback="", approve=""):
        self.id, self.title, self.level = id, title, int(level)
        self.parent, self.order = parent, int(order)
        self.links = links or []          # [(target_id, type)]
        self.model = model.strip()
        self.expects = expects.strip()
        self.produces = produces.strip()
        self.built_sha = built_sha.strip()
        self.human, self.instructions, self.code = human, instructions, code
        self.reading = reading
        self.tags = list(tags or [])          # cross-tree labels
        self.retries = str(retries).strip()   # model answers asked again
        self.fallback = fallback.strip()      # file run instead on failure
        self.approve = approve.strip()        # before | after | both

    def source_digest(self):
        """SHA-256 prefix of build-time material (human + reading)."""
        blob = (self.human.strip("\n") + "\n" + MARK.format("reading") + "\n"
                + self.reading.strip("\n"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def is_stale(self):
        """True if human/reading changed after machine side was built."""
        if not self.built_sha:
            return False
        if not (self.instructions.strip() or self.code.strip()):
            return False
        return self.source_digest() != self.built_sha

    def dump(self):
        links = ", ".join("%s:%s" % (t, ty) for t, ty in self.links)
        out = ["---", "id: " + self.id, "title: " + self.title,
               "level: %d" % self.level, "parent: " + self.parent,
               "order: %d" % self.order, "links: " + links]
        if self.model:
            out.append("model: " + self.model)
        if self.expects:
            out.append("expects: " + self.expects)
        if self.produces:
            out.append("produces: " + self.produces)
        if self.built_sha:
            out.append("built_sha: " + self.built_sha)
        if self.tags:
            out.append("tags: " + ", ".join(self.tags))
        for k in ("retries", "fallback", "approve"):
            if getattr(self, k):
                out.append("%s: %s" % (k, getattr(self, k)))
        out.append("---")
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
                    model=meta.get("model", ""),
                    expects=meta.get("expects", ""),
                    produces=meta.get("produces", ""),
                    built_sha=meta.get("built_sha", ""),
                    tags=[t.strip() for t in meta.get("tags", "").split(",")
                          if t.strip()],
                    retries=meta.get("retries", ""),
                    fallback=meta.get("fallback", ""),
                    approve=meta.get("approve", ""),
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

    # The digests of each file's sides as Tawjih last wrote them, so that
    # edits made outside the program can be found (sync, status).
    @property
    def state(self):
        if not hasattr(self, "_state"):
            f = self.root / "state.json"
            try:
                self._state = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._state = {}
        return self._state

    @staticmethod
    def side_digests(n):
        def d(t):
            return hashlib.sha256(t.strip("\n").encode("utf-8")).hexdigest()[:16]
        return {"human": d(n.human), "machine": d(n.instructions + "\0" + n.code),
                "reading": d(n.reading)}

    def _write_state(self):
        (self.root / "state.json").write_text(
            json.dumps(self.state, indent=0, sort_keys=True), encoding="utf-8")

    def record(self, n):
        if not (self.root / "state.json").exists():
            # the first write in a project made before 0.7: every file as
            # it is now is the starting point
            self._state = {i: self.side_digests(m)
                           for i, m in self.nodes.items()}
        self.state[n.id] = self.side_digests(n)
        self._write_state()

    def record_all(self):
        self._state = {i: self.side_digests(n) for i, n in self.nodes.items()}
        self._write_state()

    def forget(self, nid):
        if self.state.pop(nid, None) is not None:
            self._write_state()

    def outside_changes(self, nid):
        """Sides changed since Tawjih last wrote the file ([] if unknown)."""
        old = self.state.get(nid)
        if not old:
            return []
        new = self.side_digests(self.nodes[nid])
        return [k for k in ("human", "machine", "reading") if old.get(k) != new[k]]

    def save(self, n):
        self.file_of(n.id).write_text(n.dump(), encoding="utf-8")
        self.nodes[n.id] = n
        self.record(n)

    def children(self, pid):
        return sorted((n for n in self.nodes.values() if n.parent == pid),
                      key=lambda n: (n.order, n.id))

    def subtree_ids(self, root_id=""):
        """All node ids in preorder under root_id (inclusive if root_id given)."""
        out = []

        def walk(pid):
            for k in self.children(pid):
                out.append(k.id)
                walk(k.id)

        if root_id:
            out.append(root_id)
        walk(root_id)
        return out

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

    def check(self, branch_id=""):
        errors, warns = [], []
        scope = set(self.subtree_ids(branch_id)) if branch_id else set(self.nodes)
        for nid in scope:
            n = self.nodes[nid]
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
            if n.is_stale():
                warns.append(tag + ": machine side is stale (human/reading "
                             "changed since last build)")
            if n.retries and not re.fullmatch(r"\d", n.retries):
                errors.append(tag + ": retries must be a number from 0 to 9")
            if n.approve and n.approve not in ("before", "after", "both"):
                errors.append(tag + ": approve must be before, after or both")
            if n.fallback:
                fb = self.nodes.get(n.fallback)
                if fb is None:
                    errors.append(tag + ": fallback %s does not exist"
                                  % n.fallback)
                elif not (fb.instructions.strip() or fb.code.strip()):
                    errors.append(tag + ": fallback %s has no machine side"
                                  % n.fallback)
                elif n.produces and fb.produces and contract_fits(
                        fb.produces, n.produces):
                    errors.append(tag + ": fallback %s produces %s, but this "
                                  "file promises %s" % (n.fallback,
                                                        fb.produces,
                                                        n.produces))
            for field in ("expects", "produces"):
                pc = _parse_contract(getattr(n, field))
                if pc and pc[0] == "unknown":
                    errors.append(tag + ": unknown %s contract '%s' (%s)" % (
                        field, pc[1][0], CONTRACT_HELP))
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
                if branch_id and not (set(flow.nodes) & scope):
                    continue
                for nid in flow.nodes:
                    n = self.nodes[nid]
                    if not n.instructions.strip() and not n.code.strip():
                        warns.append("path %s: step %s has an empty machine "
                                     "side" % (f.stem, nid))
                errors += ["path %s: %s" % (f.stem, m)
                           for m in flow_contract_errors(self, flow)]
                warns += ["path %s: %s" % (f.stem, m)
                          for m in flow_condition_notes(self, flow)]
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
    except urllib.error.URLError as e:
        raise RuntimeError("Connection failed to %s: %s" % (url, e.reason))


def _model(override=""):
    m = (override or "").strip() or os.environ.get("TAWJIH_MODEL")
    if not m:
        die("name the model: pass --model NAME, set model in node header, "
            "or set the TAWJIH_MODEL environment variable")
    return m


def use_model(a):
    """--model on the command line wins over the environment variable."""
    if getattr(a, "model", None):
        os.environ["TAWJIH_MODEL"] = a.model


_usage = threading.local()     # token counts of the last answer, per thread


def _note_usage(tokens_in, tokens_out):
    _usage.last = {"in": int(tokens_in or 0), "out": int(tokens_out or 0)}


def backend_dry(prompt, model=""):
    m_tag = (" [model: %s]" % model) if model else ""
    return "[dry run%s: no model was called; prompt was %d characters]" % (
        m_tag, len(prompt))


def backend_ollama(prompt, model=""):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    data = _post(host + "/api/chat",
                 {"model": _model(model), "stream": False,
                  "messages": [{"role": "user", "content": prompt}]}, {})
    _note_usage(data.get("prompt_eval_count"), data.get("eval_count"))
    return data["message"]["content"]


def backend_anthropic(prompt, model=""):
    key = os.environ.get("ANTHROPIC_API_KEY") or die("set ANTHROPIC_API_KEY")
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    data = _post(base.rstrip("/") + "/v1/messages",
                 {"model": _model(model),
                  "max_tokens": int(os.environ.get("TAWJIH_MAX_TOKENS", "4096")),
                  "messages": [{"role": "user", "content": prompt}]},
                 {"x-api-key": key, "anthropic-version": "2023-06-01"})
    u = data.get("usage") or {}
    _note_usage(u.get("input_tokens"), u.get("output_tokens"))
    return "".join(b.get("text", "") for b in data.get("content", []))


def backend_openai(prompt, model=""):
    """Any provider exposing the OpenAI-style /chat/completions endpoint."""
    base = os.environ.get("TAWJIH_BASE_URL", "https://api.openai.com/v1")
    key = os.environ.get("TAWJIH_API_KEY", "")
    headers = {"Authorization": "Bearer " + key} if key else {}
    data = _post(base.rstrip("/") + "/chat/completions",
                 {"model": _model(model),
                  "messages": [{"role": "user", "content": prompt}]}, headers)
    u = data.get("usage") or {}
    _note_usage(u.get("prompt_tokens"), u.get("completion_tokens"))
    return data["choices"][0]["message"]["content"]


def backend_manual(prompt, model=""):
    """A person (or any chat model) plays the model: paste the answer."""
    bar = "=" * 60
    if model:
        print("\n[suggested model: %s]" % model, file=sys.stderr)
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


# ---------------------------------------------------------------- redaction
REDACT_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    "phone": r"(?<![\w+])(?:(?:\+|00)\d{1,3}[\s.-]?\d(?:[\s.-]?\d){6,12}"
             r"|05\d(?:[\s-]?\d){7})(?!\w)",
    "card": r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)",
    "iban": r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b",
    "national_id": r"(?<!\d)[12]\d{9}(?!\d)",
    "ip": r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
          r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])",
}
REDACT_TEMPLATE = r"""# Tawjih redaction: what is replaced by a placeholder such as [EMAIL_1]
# before any text is sent to a model, and put back in the answer.
# Delete this file to turn redaction off.
#
# Built-in patterns (delete a line to leave that kind alone):
@email
@phone
@card
@iban
@national_id
@ip
#
# Your own words and names, one per line (matched without regard to case),
# or a regular expression after 're:'.
# Mohammed Khalid
# re: Project\s+Falcon\w*
"""


def _luhn(digits):
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


class Redactor:
    """Replaces private text with placeholders before a model sees it, and
    puts it back in the answer. Patterns cannot know what is private: what
    they miss is sent as it is."""
    def __init__(self, rules):
        self.rules = rules            # [(label, compiled pattern)]

    @staticmethod
    def load(root):
        f = Path(root) / "redact.txt"
        if not f.is_file():
            return None
        rules, words = [], []
        for k, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                name = line[1:].strip()
                if name not in REDACT_PATTERNS:
                    die("redact.txt line %d: unknown pattern @%s (%s)" % (
                        k, name, ", ".join(sorted(REDACT_PATTERNS))))
                rules.append((name.upper(), re.compile(REDACT_PATTERNS[name])))
            elif line.startswith("re:"):
                try:
                    rules.append(("PRIVATE", re.compile(line[3:].strip())))
                except re.error as e:
                    die("redact.txt line %d: %s" % (k, e))
            else:
                words.append(line)
        # long, specific patterns first, so a phone number cannot take a
        # piece of an IBAN or a card
        order = ["IBAN", "CARD", "EMAIL", "IP", "NATIONAL_ID", "PHONE"]
        rules.sort(key=lambda r: order.index(r[0]) if r[0] in order else -1)
        if words:   # longest first, so a full name wins over a part of it
            rules.insert(0, ("PRIVATE", re.compile("|".join(
                re.escape(w) for w in sorted(words, key=len, reverse=True)),
                re.I)))
        return Redactor(rules)

    def mask(self, text):
        found = {}                    # original -> placeholder
        counts = {}

        def sub(label):
            def f(m):
                val = m.group(0)
                if label == "CARD" and not _luhn(re.sub(r"\D", "", val)):
                    return val
                if val not in found:
                    counts[label] = counts.get(label, 0) + 1
                    found[val] = "[%s_%d]" % (label, counts[label])
                return found[val]
            return f
        for label, pat in self.rules:
            text = pat.sub(sub(label), text)
        return text, {v: k for k, v in found.items()}

    @staticmethod
    def unmask(text, mapping):
        for ph in sorted(mapping, key=len, reverse=True):
            text = text.replace(ph, mapping[ph])
        return text


def call_model(store_root, backend, prompt, model=""):
    """Every model call goes through here: redaction (when the project has
    redact.txt), the call, the placeholders put back, and what it cost.
    Returns (answer, usage)."""
    red = Redactor.load(store_root) if store_root else None
    sent, mapping = red.mask(prompt) if red else (prompt, {})
    _usage.last = None
    t0 = time.monotonic()
    answer = BACKENDS[backend](sent, model=model) or ""
    usage = dict(getattr(_usage, "last", None) or {},
                 seconds=round(time.monotonic() - t0, 3),
                 chars_in=len(sent), chars_out=len(answer),
                 redacted=len(mapping))
    if mapping:
        answer = Redactor.unmask(answer, mapping)
    return answer, usage


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


def run_code(n, prev, shared, origin="", steps=None, store=None, run_dir=None,
             upstream=None, timeout=CODE_TIMEOUT):
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


# ---------------------------------------------------------------------- flow
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

    def upstream(self, nid):
        """Every file whose output can reach nid, directly or through
        others: exactly the files certain to have run (or been skipped)
        before it."""
        seen, todo = set(), [nid]
        while todo:
            cur = todo.pop()
            for e in self.edges:
                if e["dst"] == cur and e["src"] not in seen:
                    seen.add(e["src"])
                    todo.append(e["src"])
        return seen

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


def contract_fits(produces, expects):
    """None if an output under contract `produces` may go to a step that
    expects `expects`, else the reason it may not. Undeclared or unknown
    contracts give no evidence either way (None)."""
    sp, de = _parse_contract(produces), _parse_contract(expects)
    if not (sp and de) or "unknown" in (sp[0], de[0]) or de[0] == "text":
        return None                         # any output is text
    if sp[0] != de[0]:
        return "produces '%s' but the next step expects '%s'" % (produces,
                                                                expects)
    if sp[0] == "json" and de[1] and sp[1]:
        missing = [k for k in de[1] if k not in sp[1]]
        if missing:
            return "produces %s but the next step expects keys %s" % (
                produces, ", ".join(missing))
    return None


def flow_contract_errors(store, flow):
    out = []
    for e in flow.edges:
        src, dst = store.nodes[e["src"]], store.nodes[e["dst"]]
        why = contract_fits(src.produces, dst.expects)
        if why:
            out.append("contract mismatch %s -> %s: %s %s" % (
                src.id, dst.id, src.id, why))
    return out


def flow_condition_notes(store, flow):
    """IF words that will be looked for in the text of a JSON output
    because they are not among the keys its file promises: often a slip,
    since the word may also occur inside a key or a value."""
    out = []
    for e in flow.edges:
        if not e["cond"]:
            continue
        kind, keys = _parse_contract(store.nodes[e["src"]].produces) or ("", [])
        word = e["cond"][1]
        if kind == "json" and keys and word.casefold() not in (
                k.casefold() for k in keys):
            out.append("IF %s on %s: not one of its JSON keys (%s), so it is "
                       "looked for anywhere in the output text" % (
                           word, e["src"], ", ".join(keys)))
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
        if p["model"]:
            flags.append("model=" + p["model"])
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
             code_timeout=CODE_TIMEOUT, retries=None, approve=None):
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
                               "if": e["cond"], "among": e["among"]}
                              for e in flow.edges],
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

    def execute(p, n, prev):
        """Run the machine side of n on prev and return its checked output,
        or raise StepFailure. Model answers that break the contract are
        asked again, up to the retries allowed."""
        ins, code, _ = machine_side(n)
        # --model on the command line > the file's model: header > TAWJIH_MODEL
        step_model = model_override or n.model or default_model
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
                                        timeout=code_timeout)
                except (Exception, CodeTimeout) as e:
                    raise StepFailure("code", e)
                emptied = produced is not None and not produced.strip()
                code_out = produced or ""
            else:
                print("  code skipped (pass --allow-code to execute it)",
                      file=sys.stderr)
        if ins.strip():
            prompt = build_prompt(store, n, prev, code_out, text)
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
                    int(n.retries or 0)
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
        p["received_from"] = feeders
        for f in feeders:
            log.append("input from: %s [%s]" % (store.nodes[f].title, f))
            if tree_of[f] != p["tree"]:
                record["handoffs"] += 1
                print("  handoff from tree: %s" % store.ancestors(f)[0].title,
                      file=sys.stderr)
        if n.approve in ("before", "both"):
            ok, _ = _ask_approval(n, "before", prev, approve)
            if not ok:
                fail(p, n, "approval", ValueError(
                    "not approved before the step ran" + (
                        " (run with --approve ask on a terminal, or "
                        "--approve auto)" if approve == "deny" else "")),
                    outputs)
        try:
            out = execute(p, n, prev)
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
        if n.approve in ("after", "both"):
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
    if getattr(a, "model", None):
        n.model = a.model.strip()
        st.save(n)
    print("%s  [%s]  %s" % (st.number(n.id), n.id, st.file_of(n.id)))
    if a.open:
        open_editors(st, n.id)


def cmd_tree(a):
    st = Store(a.project)
    root = st.resolve(a.branch) if getattr(a, "branch", None) else ""

    def show_node(n, depth):
        flags = "".join(c if getattr(n, s).strip() else "-"
                        for c, s in zip("HICR", SECTIONS))
        stale = "!" if n.is_stale() else " "
        m_tag = ("  (%s)" % n.model) if n.model else ""
        print("%s%-8s [%s] %-8s %s%s %s%s" % (
            "  " * depth, st.number(n.id), n.id, LEVELS[n.level - 1],
            flags, stale, n.title, m_tag))

    def walk(pid, depth):
        for n in st.children(pid):
            show_node(n, depth)
            walk(n.id, depth + 1)

    if root:
        show_node(st.nodes[root], 0)
        walk(root, 1)
    else:
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


def cmd_contract(a):
    """Show or set a file's contracts (expects, produces) and model."""
    st = Store(a.project)
    n = st.nodes[st.resolve(a.ref)]
    changes = {k: v for k, v in (("expects", a.expects),
                                 ("produces", a.produces),
                                 ("model", a.model),
                                 ("retries", a.retries),
                                 ("fallback", a.fallback),
                                 ("approve", a.approve)) if v is not None}
    if changes.get("retries", "none").strip().lower() not in ("", "none") \
            and not re.fullmatch(r"\d", changes["retries"].strip()):
        die("retries is a number from 0 to 9")
    if changes.get("approve", "none").strip().lower() not in (
            "", "none", "before", "after", "both"):
        die("approve is before, after or both")
    fb = changes.get("fallback", "none").strip()
    if fb.lower() not in ("", "none"):
        fid = st.resolve(fb)
        if fid == n.id:
            die("a file cannot be its own fallback")
        f = st.nodes[fid]
        if not (f.instructions.strip() or f.code.strip()):
            die("the fallback %s has no machine side" % f.title)
        changes["fallback"] = fid
    for k in ("expects", "produces"):
        v = changes.get(k)
        if v is None or v.strip().lower() in ("", "none"):
            continue
        kind, keys = _parse_contract(v)
        if kind == "unknown":
            die("%s: unknown contract '%s' (%s)" % (k, v, CONTRACT_HELP))
        changes[k] = "json(%s)" % ", ".join(keys) if kind == "json" and keys \
            else kind
    if changes:
        _backup(st, n)
        for k, v in changes.items():
            setattr(n, k, "" if v.strip().lower() in ("", "none") else v)
        st.save(n)
    print("%s [%s] %s" % (st.number(n.id), n.id, n.title))
    for k in ("expects", "produces", "model", "retries", "fallback",
              "approve"):
        v = getattr(n, k)
        if k == "fallback" and v:
            v = "%s [%s] %s" % (st.number(v), v, st.nodes[v].title) \
                if v in st.nodes else "[%s] (missing)" % v
        print("  %-9s %s" % (k + ":", v or "-"))
    if changes:                 # what the change does to the saved paths
        for pf in sorted((st.root / "paths").glob("*.path")):
            try:
                flow = st.load_flow(pf.stem)
            except FlowError:
                continue
            if n.id in flow.nodes:
                for err in flow_contract_errors(st, flow):
                    print("warning: path %s: %s" % (pf.stem, err))


def cmd_confirm(a):
    """Record that a machine side was reviewed against the current human
    and reading sides and still fits them: it is no longer stale."""
    st = Store(a.project)
    for ref in a.refs:
        n = st.nodes[st.resolve(ref)]
        if not (n.instructions.strip() or n.code.strip()):
            die("%s %s has no machine side to confirm" % (st.number(n.id),
                                                          n.title))
        was = n.is_stale()
        n.built_sha = n.source_digest()
        st.save(n)
        print("%s [%s] %s: %s" % (st.number(n.id), n.id, n.title,
                                  "confirmed, no longer stale" if was else
                                  "confirmed"))


def cmd_check(a):
    st = Store(a.project)
    branch_id = st.resolve(a.branch) if getattr(a, "branch", None) else ""
    errors, warns = st.check(branch_id=branch_id)
    for w in warns:
        print("warning: " + w)
    for e in errors:
        print("ERROR:   " + e)
    print("%d error(s), %d warning(s)" % (len(errors), len(warns)))
    sys.exit(1 if errors else 0)


TANGLE_HEAD = '''"""Generated by tawjih tangle from %s.

Each code side is a function step_<id>(input, origin="", vars=None,
steps=None, meta=None, linked=None) that returns the step's output, as
Tawjih would pass it on (None if the code sets no output). STEPS lists them
in %s order. Instructions sides are not included: they need a model. These
functions do not run the path; call them yourself, for example to test one
step on its own.
"""
'''


def tangle_source(st, ids, label, order_word):
    """The Python module text for the code sides of ids, and its count."""
    parts, names = [TANGLE_HEAD % (label, order_word)], []
    for nid in ids:
        n = st.nodes[nid]
        code = n.code.strip("\n")
        if not code.strip():
            continue
        fname = "step_" + nid
        doc = "%s (%s)" % (n.title, st.number(nid))
        if n.human.strip():
            doc += "\n\n" + n.human.strip()
        doc = doc.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
        parts.append("")
        parts.append("def %s(input, origin=\"\", vars=None, steps=None, "
                     "meta=None, linked=None):" % fname)
        parts.append(textwrap.indent('"""%s\n"""' % doc, "    "))
        parts.append("    vars = {} if vars is None else vars\n"
                     "    steps = {} if steps is None else steps\n"
                     "    meta = {} if meta is None else meta\n"
                     "    linked = linked or (lambda rel_type=\"\": [])\n"
                     "    output = None")
        if '"""' in code or "'''" in code:
            # indenting would change the text inside multi-line strings
            parts.append("    _ns = dict(input=input, origin=origin, vars=vars,"
                         " steps=steps, meta=meta, linked=linked, output=None)")
            parts.append("    exec(%r, _ns)" % code)
            parts.append("    output = _ns[\"output\"]")
        else:
            parts.append(textwrap.indent(code, "    "))
        parts.append("    return None if output is None else str(output)")
        parts.append("")
        names.append((nid, n.title, fname))
    parts.append("")
    parts.append("STEPS = [")
    for nid, title, fname in names:
        parts.append("    (%r, %r, %s)," % (nid, title, fname))
    parts.append("]")
    return "\n".join(parts) + "\n", len(names)


def cmd_tangle(a):
    """Export the code sides of a branch or a path as a Python module of
    functions, one per step."""
    st = Store(a.project)
    if a.path:
        try:
            ids = st.load_flow(a.path).order()
        except FlowError as e:
            die("path %s: %s" % (a.path, e))
        label, order_word = "path '%s'" % a.path, "execution"
    else:
        root = st.resolve(a.branch) if a.branch else ""
        ids = st.subtree_ids(root)
        label = ("branch %s (%s)" % (st.number(root), st.nodes[root].title)
                 if root else "all trees")
        order_word = "tree"
    rendered, count = tangle_source(st, ids, label, order_word)
    if not count:
        die("no code sections found in " + label)
    try:
        compile(rendered, a.out or "tangled", "exec")
    except SyntaxError as e:
        die("the exported module does not compile (%s); check the code "
            "sides with: check" % e)
    if a.out:
        out_path = Path(a.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
        print("tangled %d code step(s) into %s" % (count, out_path))
    else:
        print(rendered, end="")


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
    print(_run_from_args(st, a, flow))


def _run_from_args(st, a, flow):
    if a.resume and a.no_cache:
        die("--resume reads the cache and --no-cache turns it off; "
            "use one of them")
    use_model(a)
    text = a.input
    if a.input_file:
        text = Path(a.input_file).read_text(encoding="utf-8")
    redo = [st.resolve(r) for r in a.redo or []]
    stray = [st.number(r) for r in redo if r not in flow.nodes]
    if stray:
        die("--redo names a file that is not in this path: " + ", ".join(stray))
    if redo and not a.resume:
        die("--redo goes with --resume (without --resume every step is "
            "asked again anyway)")
    if a.allow_code:
        audit_warnings(st, flow)
    return run_flow(st, a.name, text, a.backend, a.allow_code, flow,
                    resume=a.resume, use_cache=not a.no_cache, redo=redo,
                    model_override=getattr(a, "model", None) or "",
                    code_timeout=a.code_timeout,
                    retries=a.retries, approve=a.approve)


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
    print(_run_from_args(st, a, flow))


# --------------------------------------------------------------- the network
GRAPH_FORMATS = ("mermaid", "md", "dot")


def _mm(text):
    """Text for a quoted Mermaid label."""
    return (text.replace("&", "&amp;").replace('"', "#quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def _dot(text):
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _node_lines(st, n, with_kind=False):
    lines = ["%s %s" % (st.number(n.id), n.title)]
    if with_kind:
        kind = "+".join(k for k in ("instructions", "code")
                        if getattr(n, k).strip()) or "no machine side"
        lines.append(kind)
    for field in ("expects", "produces"):
        if getattr(n, field):
            lines.append("%s %s" % (field, getattr(n, field)))
    if n.is_stale():
        lines.append("STALE")
    return lines


def graph_source(st, fmt, path=None, branch="", name=""):
    """The network as Mermaid or Graphviz DOT text.
    Without a path: the trees (parent to child). With a path: its steps,
    grouped by tree, with every connection and its condition."""
    mermaid = fmt in ("mermaid", "md")
    if path is not None:
        flow = path
        ids = flow.order()
        groups = {}
        for nid in ids:
            groups.setdefault(st.ancestors(nid)[0].id, []).append(nid)
        edges = []
        for e in flow.edges:
            if e["among"] is not None:
                label, style = "AMONG", "among"
            elif e["cond"]:
                label = "IF %s%s" % ("NOT " if e["cond"][0] else "",
                                     e["cond"][1])
                style = "if"
            else:
                label, style = "", "to"
            cross = st.ancestors(e["src"])[0].id != st.ancestors(e["dst"])[0].id
            edges.append((e["src"], e["dst"], label, style, cross))
        with_kind = True
    else:
        root = st.resolve(branch) if branch else ""
        ids = st.subtree_ids(root)
        groups = {"": ids}
        edges = [(st.nodes[i].parent, i, "", "tree", False) for i in ids
                 if st.nodes[i].parent and st.nodes[i].parent in ids]
        with_kind = False
    out = []
    if mermaid:
        out.append("flowchart %s" % ("LR" if path is not None else "TD"))
        for gid, members in groups.items():
            if gid:
                out.append('  subgraph t_%s["%s"]' % (gid, _mm(st.nodes[gid].title)))
            for nid in members:
                n = st.nodes[nid]
                label = "<br/>".join(_mm(x) for x in _node_lines(st, n, with_kind))
                out.append('  %sn_%s["%s"]' % ("  " if gid else "", nid, label))
            if gid:
                out.append("  end")
        for src, dst, label, style, cross in edges:
            arrow = "-.->" if style == "among" else ("==>" if cross else "-->")
            lab = '|"%s"|' % _mm(label) if label else ""
            out.append("  n_%s %s%s n_%s" % (src, arrow, lab, dst))
        machine = [i for i in ids if st.nodes[i].instructions.strip()
                   or st.nodes[i].code.strip()]
        stale = [i for i in ids if st.nodes[i].is_stale()]
        out.append("  classDef machine stroke-width:3px")
        out.append("  classDef stale fill:#fde2e2,stroke:#c0392b")
        if machine:
            out.append("  class %s machine" % ",".join("n_" + i for i in machine))
        if stale:
            out.append("  class %s stale" % ",".join("n_" + i for i in stale))
    else:
        out.append("digraph tawjih {")
        out.append('  rankdir=%s; node [shape=box, style=rounded, '
                   'fontname="DejaVu Sans"];' % ("LR" if path is not None else "TB"))
        for k, (gid, members) in enumerate(groups.items()):
            if gid:
                out.append('  subgraph cluster_%d { label="%s";' % (
                    k, _dot(st.nodes[gid].title)))
            for nid in members:
                n = st.nodes[nid]
                extra = []
                if n.instructions.strip() or n.code.strip():
                    extra.append("penwidth=2")
                if n.is_stale():
                    extra.append('color="#c0392b", fillcolor="#fde2e2", '
                                 'style="rounded,filled"')
                out.append('  n_%s [label="%s"%s];' % (
                    nid, _dot("\n".join(_node_lines(st, n, with_kind))),
                    "".join(", " + x for x in extra)))
            if gid:
                out.append("  }")
        for src, dst, label, style, cross in edges:
            attrs = []
            if label:
                attrs.append('label="%s"' % _dot(label))
            if style == "among":
                attrs.append("style=dashed")
            if cross:
                attrs.append("penwidth=2.5")
            out.append("  n_%s -> n_%s%s;" % (
                src, dst, " [%s]" % ", ".join(attrs) if attrs else ""))
        out.append("}")
    text = "\n".join(out) + "\n"
    if fmt == "md":
        what = ("path '%s'" % name) if path is not None else "the trees"
        text = ("# Tawjih network: %s\n\nThick arrows cross from one tree to "
                "another; dashed arrows are AMONG choices; STALE marks a "
                "machine side older than its human or reading side.\n\n"
                "```mermaid\n%s```\n" % (what, text))
    return text


def cmd_graph(a):
    """Draw the trees, or one path, as Mermaid or Graphviz DOT."""
    st = Store(a.project)
    flow = None
    if a.path:
        try:
            flow = st.load_flow(a.path)
            flow.order()
        except FlowError as e:
            die("path %s: %s" % (a.path, e))
    if not st.nodes:
        die("the project has no files yet")
    fmt = a.format or {".mmd": "mermaid", ".mermaid": "mermaid", ".dot": "dot",
                       ".gv": "dot"}.get(Path(a.out or "").suffix.lower(), "md")
    a.format = fmt
    text = graph_source(st, fmt, flow, a.branch or "", a.path or "")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text, encoding="utf-8")
        print("wrote %s (%s)" % (a.out, a.format))
    else:
        print(text, end="")


# ----------------------------------------------------------- path generation
GENERATE_HEAD = (
    "You are designing a production path in Tawjih. A path connects files "
    "with these statements, one per line, files named by id:\n"
    "  A TO B            A's output goes to B\n"
    "  A TO B AND C      to both (branching)\n"
    "  B AND C TO D      D receives both outputs (joining)\n"
    "  A TO B IF word    only when the word holds for A's output: for a JSON "
    "output with that key, when its value is true or non-empty; otherwise when "
    "the output contains the word (IF NOT word for the opposite)\n"
    "  A AMONG B C       A's output names one of B, C; only that one runs\n"
    "  A TO D THROUGH B C   A, then B, then C, then D\n"
    "Use only files from the list. A file's 'expects' must fit what the "
    "files before it 'produce'. No loops; each file at most once.\n"
    "Reply with the statements only: no explanation, no code fence.")


def _proposal_file(st, name):
    return st.pdir / (name + ".proposed")


def _scope_ids(st, scope):
    root = st.resolve(scope) if scope else ""
    return [i for i in st.subtree_ids(root)
            if st.nodes[i].instructions.strip() or st.nodes[i].code.strip()]


def contract_route(st, start, end, allowed):
    """Shortest chain start -> ... -> end in which each step's declared
    'produces' fits the next step's declared 'expects'. None if none."""
    def fits(u, v):
        pu, ev = st.nodes[u].produces, st.nodes[v].expects
        if not (_parse_contract(pu) and _parse_contract(ev)):
            return False                    # needs both declared
        if "unknown" in (_parse_contract(pu)[0], _parse_contract(ev)[0]):
            return False
        sp, de = _parse_contract(pu), _parse_contract(ev)
        if de[0] == "json" and de[1] and not sp[1]:
            return False                    # keys required but not promised
        return contract_fits(pu, ev) is None
    prev, todo = {start: None}, [start]
    while todo:
        nxt = []
        for u in todo:
            for v in allowed:
                if v not in prev and fits(u, v):
                    prev[v] = u
                    if v == end:
                        chain = [v]
                        while prev[chain[-1]] is not None:
                            chain.append(prev[chain[-1]])
                        return list(reversed(chain))
                    nxt.append(v)
        todo = nxt
    return None


def check_proposal(st, statements):
    """Build a flow from statements and list everything that would stop it
    from running. Returns (flow or None, problems)."""
    def quiet(ref):                     # report once, in the problem list
        with contextlib.redirect_stderr(io.StringIO()):
            return st.resolve(ref)
    try:
        flow = build_flow(statements, quiet)
        flow.order()
    except FlowError as e:
        return None, [str(e)]
    except Die as e:
        return None, [e.msg]
    problems = ["step %s %s has no machine side" % (st.number(i), st.nodes[i].title)
                for i in flow.nodes if not (st.nodes[i].instructions.strip()
                                            or st.nodes[i].code.strip())]
    problems += flow_contract_errors(st, flow)
    return flow, problems


def cmd_generate(a):
    """Propose a path: from contracts (no model) or from a model given a
    goal. The proposal is written next to the paths and runs nothing;
    'generate NAME --accept' turns it into a path."""
    st = Store(a.project)
    if not re.fullmatch(r"[\w\-]+", a.name):
        die("path names may contain only letters, digits, '_' and '-'")
    prop = _proposal_file(st, a.name)
    target = st.pdir / (a.name + ".path")
    if a.accept:
        if not prop.is_file():
            die("there is no proposal named '%s' (make one with generate)"
                % a.name)
        stmts = []
        for line in prop.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                try:
                    stmts.append(shlex.split(line))
                except ValueError as e:
                    die("proposal line cannot be read (%s): %s" % (e, line))
        flow, problems = check_proposal(st, stmts)
        if flow is None or problems:
            die("the proposal cannot be accepted:\n  " + "\n  ".join(
                problems) + "\nEdit %s and try again." % prop)
        if target.is_file() and not a.replace:
            die("a path named '%s' already exists; add --replace to "
                "overwrite it" % a.name)
        write_flow(st, a.name, flow)
        prop.unlink()
        print("accepted: paths/%s.path" % a.name)
        print_plan(st, a.name, plan_steps(st, flow), sys.stdout)
        return
    allowed = _scope_ids(st, a.scope)
    if not allowed:
        die("no file with a machine side in this scope")
    if a.goal is None:
        if not (a.start and a.end):
            die("give --start and --end to link files by their contracts, "
                "or --goal to ask a model")
        start, end = st.resolve(a.start), st.resolve(a.end)
        for ref, nid in ((a.start, start), (a.end, end)):
            if nid not in allowed:
                die("%s has no machine side or is outside the scope" % ref)
        chain = contract_route(st, start, end, allowed)
        if chain is None:
            die("no chain of files links %s to %s by their contracts. Each "
                "step needs 'produces' and the next one a fitting 'expects' "
                "(see: graph). %s produces '%s'; %s expects '%s'." % (
                    st.number(start), st.number(end), st.number(start),
                    st.nodes[start].produces or "-", st.number(end),
                    st.nodes[end].expects or "-"))
        statements = [[chain[0], "TO", chain[-1]] + (
            ["THROUGH"] + chain[1:-1] if len(chain) > 2 else [])]
        how = "contracts, from %s to %s" % (st.number(start), st.number(end))
    else:
        rows = []
        for nid in allowed:
            n = st.nodes[nid]
            kind = "+".join(k for k in ("instructions", "code")
                            if getattr(n, k).strip())
            rows.append("%s | %s | %s | %s | %s | expects %s | produces %s" % (
                nid, st.number(nid), st.ancestors(nid)[0].title, n.title,
                kind, n.expects or "-", n.produces or "-"))
        prompt = ("%s\n\n# Files (id | number | tree | title | machine side |"
                  " expects | produces)\n%s\n\n# Goal\n%s" % (
                      GENERATE_HEAD, "\n".join(rows), a.goal))
        use_model(a)
        if a.backend == "dry":
            print(prompt)
            print("\n[dry run: nothing was sent; choose --backend]",
                  file=sys.stderr)
            return
        try:
            answer = call_model(st.root, a.backend, prompt, a.model or "")[0]
        except Exception as e:
            die("model call failed: %s: %s" % (type(e).__name__, e))
        statements, bad = [], []
        for line in strip_fence(answer).splitlines():
            line = line.split("#", 1)[0].strip().strip("`").strip()
            if not line:
                continue
            try:
                statements.append(shlex.split(line))
            except ValueError:
                bad.append(line)
        if bad or not statements:
            statements = []
        how = "model %s/%s, goal: %s" % (a.backend,
                                         os.environ.get("TAWJIH_MODEL", ""),
                                         " ".join(a.goal.split()))
        if not statements:
            st.pdir.mkdir(exist_ok=True)
            prop.write_text("# proposed by %s\n# INVALID: the answer holds no "
                            "readable statement. The answer was:\n%s\n" % (
                                how, "\n".join("# " + l for l in
                                               answer.splitlines())),
                            encoding="utf-8")
            die("the model's answer holds no path statement; it is kept in "
                "%s" % prop)
    flow, problems = check_proposal(st, statements)
    st.pdir.mkdir(exist_ok=True)
    head = ["# proposed by %s on %s" % (how, time.strftime("%Y-%m-%d %H:%M")),
            "# review it, edit it if needed, then: generate %s --accept"
            % a.name]
    if flow is None:
        body = "\n".join(" ".join(x) for x in statements) + "\n"
    else:
        body = flow_text(st, flow)
    head += ["# PROBLEM: " + p for p in problems]
    notes = flow_condition_notes(st, flow) if flow is not None else []
    head += ["# NOTE: " + m for m in notes]
    prop.write_text("\n".join(head) + "\n" + body, encoding="utf-8")
    if flow is not None:
        print_plan(st, a.name + " (proposed)", plan_steps(st, flow),
                   sys.stdout)
    if problems:
        die("the proposal has problems (kept in %s for you to edit):\n  %s"
            % (prop, "\n  ".join(problems)))
    for m in notes:
        print("note: " + m)
    print("proposal written to %s; nothing was run. Accept it with: "
          "generate %s --accept" % (prop, a.name))


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
            "original request), vars (dict shared by all steps), steps (dict "
            "mapping previous node ids to their outputs), meta (dict), and "
            "linked(type='') helper. The code cannot see the human or reading "
            "sides. Assign the step's result to a variable named output.",
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
    # Include linked nodes' interfaces/concepts when building machine sides
    if a.side in ("instructions", "code") and n.links:
        linked_ctx = []
        for tid, ty in n.links:
            tn = st.nodes.get(tid)
            if tn:
                info = ["## Linked [%s] %s (%s)" % (tn.id, tn.title, ty)]
                if tn.expects:
                    info.append("expects: " + tn.expects)
                if tn.produces:
                    info.append("produces: " + tn.produces)
                if tn.human.strip():
                    info.append(tn.human.strip())
                linked_ctx.append("\n".join(info))
        if linked_ctx:
            parts.append("# Linked branches\n" + "\n\n".join(linked_ctx))
    parts.append("# File: %s [%s]" % (n.title, LEVELS[n.level - 1]))
    if n.expects:
        parts.append("Contract expects: " + n.expects)
    if n.produces:
        parts.append("Contract produces: " + n.produces)
    for sec in SECTIONS:
        parts.append("## Current %s side\n%s" % (sec, getattr(n, sec) or "(empty)"))
    if a.source:
        parts.append("# Source material\n"
                     + Path(a.source).read_text(encoding="utf-8"))
    parts.append("# Request\n" + a.ask)
    prompt = "\n\n".join(parts)
    use_model(a)
    step_model = (getattr(a, "model", None) or n.model
                  or os.environ.get("TAWJIH_MODEL", ""))
    if a.backend == "dry":
        print(prompt)
        print("\n[dry run: %d characters; choose --backend to send it]"
              % len(prompt), file=sys.stderr)
        return
    try:
        new = strip_fence(call_model(st.root, a.backend, prompt,
                                     step_model)[0]).strip("\n")
    except Exception as e:
        die("model call failed: %s: %s" % (type(e).__name__, e))
    old = getattr(n, a.side).strip("\n")
    if not new.strip():
        die("the model returned nothing; file unchanged")
    if any(l.strip() == MARK.format(sec) for l in new.splitlines() for sec in SECTIONS):
        die("the answer contains a section marker line; file unchanged")
    if new == old:
        if a.side in ("instructions", "code") and n.is_stale():
            # asked to rebuild from the current sides, the model kept it:
            # that is a rebuild, so the file is no longer stale
            n.built_sha = n.source_digest()
            st.save(n)
            print("no change; the machine side is now recorded as built "
                  "from the current human and reading sides")
        else:
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
    if a.side in ("instructions", "code"):
        n.built_sha = n.source_digest()
    st.save(n)
    print("saved %s (previous version: %s)" % (st.file_of(nid), bak))


# ------------------------------------------------------------ side editors
GUI_PARALLEL = {"notepad", "notepad++", "gedit", "kate", "mousepad", "xed",
                "pluma"}                       # one window per side
SPLIT_FLAGS = {"vim": ["-O"], "nvim": ["-O"]}  # sides shown side by side
NEEDS_WAIT = {"code", "code-insiders", "codium", "subl"}
VSCODE = {"code", "code-insiders", "codium"}    # one file per part, by type
EDITOR_ALIASES = {"vscode": "code", "vs-code": "code", "vscodium": "codium"}


def editor_command(arg):
    spec = (arg or os.environ.get("TAWJIH_EDITOR") or os.environ.get("VISUAL")
            or os.environ.get("EDITOR"))
    if spec:
        cmd = [c.strip('"') for c in shlex.split(spec, posix=(os.name != "nt"))]
        cmd[0] = EDITOR_ALIASES.get(cmd[0].lower(), cmd[0])
    elif os.name == "nt":
        cmd = ["notepad"]
    else:
        found = next((e for e in ("nano", "vim", "vi") if shutil.which(e)), None)
        if not found:
            die("no editor found; set TAWJIH_EDITOR or pass --editor")
        cmd = [found]
    name = re.split(r"[\\/]", cmd[0])[-1].lower()
    for ext in (".exe", ".cmd", ".bat"):
        if name.endswith(ext):
            name = name[:-len(ext)]
    cmd[0] = shutil.which(cmd[0]) or cmd[0]
    if name in VSCODE and "--new-window" not in cmd and "-n" not in cmd:
        cmd.append("--new-window")
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


def open_editors(st, nid, editor=None, force_reading=False, split=False,
                 discard=False):
    """Open one editor per side: human, machine, and reading when the file
    has deposited-book passages. Changes are saved when the editors close."""
    n = st.nodes[nid]
    cmd, ename = editor_command(editor)
    split = split or ename in VSCODE
    m_ins, m_code = MARK.format("instructions"), MARK.format("code")
    if split:
        sides = [("human", "human.md", n.human),
                 ("instructions", "instructions.md", n.instructions),
                 ("code", "code.py", n.code)]
    else:
        sides = [("human", "human.txt", n.human),
                 ("machine", "machine.txt", "%s\n%s\n%s\n%s" % (
                     m_ins, n.instructions, m_code, n.code))]
    if n.reading.strip() or force_reading:
        sides.append(("reading", "reading.md" if split else "reading.txt",
                      n.reading))
    wdir = st.root / "work" / nid
    if wdir.is_dir() and any(wdir.iterdir()):
        if not discard:
            die("an earlier edit of this file did not finish, or it is open "
                "in another terminal. Its text is in %s. Copy what you need "
                "from there, then run again with --discard" % wdir)
        shutil.rmtree(str(wdir))
    wdir.mkdir(parents=True, exist_ok=True)
    enc = "utf-8-sig" if os.name == "nt" and not split else "utf-8"
    files = []
    for name, fname, text in sides:
        f = wdir / fname
        f.write_text(text.strip("\n") + "\n", encoding=enc)
        files.append(f)
    sides = [(name, text) for name, _, text in sides]
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
    for name in ("human", "reading") + (("instructions", "code") if split
                                        else ()):
        if any(l.strip() in marks for l in new.get(name, "").splitlines()):
            die("the %s side contains a '=== ... ===' marker line; nothing "
                "changed (your text is still in %s)" % (name, wdir))
    ins, code, cur = [], [], None
    if split:
        ins, code = new["instructions"].splitlines(), new["code"].splitlines()
    for line in new.get("machine", "").splitlines():
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
        if any(k in changed for k in ("instructions", "code")):
            n.built_sha = n.source_digest()
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
    open_editors(st, st.resolve(a.ref), a.editor, a.reading, a.split,
                 a.discard)


def cmd_cache(a):
    """Show how much is kept in cache/, or empty it."""
    st = Store(a.project)
    files = sorted((st.root / "cache").glob("*.json")) \
        if (st.root / "cache").is_dir() else []
    size = sum(f.stat().st_size for f in files)
    if a.clear:
        for f in files:
            f.unlink()
        print("removed %d kept answer(s), %d bytes" % (len(files), size))
    else:
        print("%d kept answer(s), %d bytes, in %s" % (len(files), size,
                                                      st.root / "cache"))


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


# ------------------------------------------------------------ source formats
# deposit turns a source file into plain text with '#' headings (one '#' per
# level), which is what filter splits on. Mathematics is kept as LaTeX
# ($...$ inline, $$...$$ displayed). Standard library only.

def _tidy(text):
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _heading(level, title):
    title = " ".join(title.split())
    return "\n\n%s %s\n\n" % ("#" * max(1, min(level, 6)), title)


def from_markdown(text, path=None):
    """Markdown already uses '#' headings. Setext headings become '#', and
    '#' lines inside code fences are protected."""
    out, fence = [], None
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"\s*(```+|~~~+)", line)
        if m:
            mark = m.group(1)[0] * 3
            fence = None if fence and mark == fence else (fence or mark)
            out.append(line)
        elif fence:
            out.append(" " + line if line.startswith("#") else line)
        elif (i + 1 < len(lines) and line.strip()
              and re.fullmatch(r"\s*(=+|-+)\s*", lines[i + 1])
              and not line.lstrip().startswith(("-", "*", "#", ">"))):
            out.append("%s %s" % ("#" if "=" in lines[i + 1] else "##",
                                  line.strip()))
            i += 1
        else:
            m = re.match(r"(#{1,6})\s+(.*?)\s*#*\s*$", line)
            out.append("%s %s" % (m.group(1), m.group(2)) if m else line)
        i += 1
    return _tidy("\n".join(out))


TEX_LEVELS = ["part", "chapter", "section", "subsection", "subsubsection",
              "paragraph", "subparagraph"]
TEX_BLOCKS = {"theorem": "Theorem", "lemma": "Lemma", "proposition": "Proposition",
              "corollary": "Corollary", "definition": "Definition",
              "example": "Example", "remark": "Remark", "proof": "Proof",
              "exercise": "Exercise", "conjecture": "Conjecture",
              "claim": "Claim", "fact": "Fact", "note": "Note",
              "problem": "Problem", "solution": "Solution", "axiom": "Axiom",
              "notation": "Notation", "question": "Question"}
TEX_VERBATIM = r"verbatim\*?|Verbatim\*?|BVerbatim|LVerbatim|lstlisting|minted|alltt"
TEX_MATH = (r"equation|align|gather|multline|eqnarray|displaymath|flalign|"
            r"alignat|IEEEeqnarray|dmath|math")
TEX_DROP = r"tikzpicture|tikzcd|pspicture|picture|filecontents\*?|comment"
TEX_LAYOUT = (r"itshape|bfseries|slshape|scshape|upshape|mdseries|rmfamily|"
              r"sffamily|ttfamily|normalfont|em|it|bf|sl|sc|tt|rm|sf|tiny|"
              r"scriptsize|footnotesize|small|normalsize|large|Large|LARGE|huge|"
              r"Huge|centering|raggedright|raggedleft|noindent|indent|par|"
              r"newline|linebreak|pagebreak|nolinebreak|nopagebreak|clearpage|"
              r"cleardoublepage|newpage|smallskip|medskip|bigskip|vfill|hfill|"
              r"null|maketitle|tableofcontents|listoffigures|listoftables|"
              r"appendix|frontmatter|mainmatter|backmatter|protect|relax|"
              r"centerline|leavevmode|hline|toprule|midrule|bottomrule|"
              r"addlinespace|big|Big|bigg|Bigg")
TEX_NOTTEXT = (r"index|glossary|vspace|hspace|addvspace|setlength|addtolength|"
               r"addtocounter|setcounter|stepcounter|refstepcounter|"
               r"addcontentsline|markboth|markright|bibliographystyle|"
               r"bibliography|usepackage|RequirePackage|documentclass|"
               r"pagestyle|thispagestyle|pagenumbering|color|graphicspath|"
               r"input|include|includeonly|hypersetup|geometry|setcopyright|"
               r"selectlanguage|indexsee|cline|cmidrule")
TEX_STYLE = (r"emph|textbf|textit|textsl|textsc|texttt|textsf|textrm|textup|"
             r"textmd|textnormal|underline|uline|mbox|hbox|fbox|makebox|text|"
             r"textsuperscript|textsubscript|enquote|textquote|"
             r"MakeUppercase|MakeLowercase|uppercase|lowercase|"
             r"foreignlanguage\s*\{[^}]*\}|textcolor\s*\{[^}]*\}|"
             r"colorbox\s*\{[^}]*\}")
TEX_ACCENTS = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308",
               "~": "\u0303", "=": "\u0304", ".": "\u0307", "u": "\u0306",
               "v": "\u030c", "H": "\u030b", "c": "\u0327", "k": "\u0328",
               "d": "\u0323", "b": "\u0331", "r": "\u030a", "t": "\u0361"}
TEX_WORDS = {"ldots": "\u2026", "dots": "\u2026", "textellipsis": "\u2026",
             "LaTeXe": "LaTeX2e", "LaTeX": "LaTeX", "TeX": "TeX",
             "XeLaTeX": "XeLaTeX", "LuaLaTeX": "LuaLaTeX", "today": "",
             "slash": "/", "textbackslash": "\\", "ss": "\u00df",
             "o": "\u00f8", "O": "\u00d8", "ae": "\u00e6", "AE": "\u00c6",
             "oe": "\u0153", "OE": "\u0152", "aa": "\u00e5", "AA": "\u00c5",
             "l": "\u0142", "L": "\u0141", "i": "\u0131", "j": "\u0237",
             "dag": "\u2020", "ddag": "\u2021", "S": "\u00a7", "P": "\u00b6",
             "copyright": "\u00a9", "pounds": "\u00a3", "texteuro": "\u20ac",
             "euro": "\u20ac", "textregistered": "\u00ae",
             "texttrademark": "\u2122", "textdegree": "\u00b0",
             "textendash": "\u2013", "textemdash": "\u2014",
             "textquoteleft": "\u2018", "textquoteright": "\u2019",
             "textquotedblleft": "\u201c", "textquotedblright": "\u201d",
             "quad": " ", "qquad": " ", "enspace": " ", "thinspace": " ",
             "textvisiblespace": "\u2423", "textbar": "|", "textless": "<",
             "textgreater": ">", "textasciitilde": "~", "textasciicircum": "^",
             "textunderscore": "_", "checkmark": "\u2713"}


def _tex_arg(s, i):
    """The {...} group starting at s[i] (after optional spaces): (text, end),
    or (None, i) when there is none."""
    while i < len(s) and s[i] in " \t\n":
        i += 1
    if i >= len(s) or s[i] != "{":
        return None, i
    depth, j = 0, i
    while j < len(s):
        if s[j] == "\\":
            j += 2
            continue
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def _tex_strip_comments(t):
    """'%' to the end of the line, and with it the line break and the
    leading spaces of the next line; '\\%' is a percent sign."""
    return re.sub(r"(?<!\\)%[^\n]*\n?[ \t]*", "", t)


def _tex_includes(text, path, seen):
    """Read in the files named by \\input and \\include, recursively,
    except on lines where they are commented out."""
    path = Path(path).resolve()
    seen.add(path)

    def include(m):
        f = (path.parent / m.group(2).strip()).resolve()
        if not f.suffix:
            f = f.with_suffix(".tex")
        if f in seen or not f.is_file():
            return ""
        return _tex_includes(f.read_text(encoding="utf-8", errors="replace"),
                             f, seen)
    lines = []
    for line in text.split("\n"):
        c = re.search(r"(?<!\\)%", line)
        head, tail = (line[:c.start()], line[c.start():]) if c else (line, "")
        lines.append(re.sub(r"\\(input|include)\s*\{([^}]*)\}", include, head)
                     + tail)
    return "\n".join(lines)


def _tex_macros(s):
    """Read the definitions of the document's own commands and theorem
    environments, remove them, and return (text, macros, theorems)."""
    macros, theorems, out, i = {}, {}, [], 0
    defn = re.compile(r"\\(newcommand|renewcommand|providecommand|"
                      r"DeclareRobustCommand|DeclareMathOperator|newtheorem|"
                      r"def|NewDocumentCommand|RenewDocumentCommand|"
                      r"ProvideDocumentCommand)\*?\s*")

    def name_at(k):
        nm, k2 = _tex_arg(s, k)
        if nm is None:
            m = re.match(r"\s*(\\[A-Za-z@]+)", s[k:])
            nm, k2 = m.group(1), k + m.end()
        return nm.strip().lstrip("\\"), k2

    def opt(k):
        m = re.match(r"\s*\[([^\]]*)\]", s[k:])
        return (m.group(1), k + m.end()) if m else (None, k)

    while True:
        m = defn.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        kind, k = m.group(1), m.end()
        try:
            if kind == "def":
                nm = re.match(r"\\([A-Za-z@]+)([^{]*)", s[k:])
                body, k = _tex_arg(s, k + nm.end())
                if body is None or "#" in nm.group(2):
                    raise ValueError
                macros[nm.group(1)] = (0, None, body)
            elif kind == "newtheorem":
                env, k = _tex_arg(s, k)
                _, k = opt(k)
                label, k = _tex_arg(s, k)
                _, k = opt(k)
                theorems[env.strip()] = label.strip()
            elif kind.endswith("DocumentCommand"):
                nm, k = name_at(k)
                spec, k = _tex_arg(s, k)
                body, k = _tex_arg(s, k)
                macros[nm] = (len(re.findall(r"[mo]", spec or "")), None, body)
            elif kind == "DeclareMathOperator":
                nm, k = name_at(k)
                body, k = _tex_arg(s, k)
                macros[nm] = (0, None, "\\operatorname{%s}" % body)
            else:
                nm, k = name_at(k)
                n, k = opt(k)
                default, k = opt(k)
                body, k = _tex_arg(s, k)
                if body is None:
                    raise ValueError
                macros[nm] = (int(n or 0), default, body)
            i = k
        except (ValueError, AttributeError, TypeError):
            out.append(m.group(0))              # not understood: left as is
            i = m.end()
    return "".join(out), macros, theorems


def _tex_expand(t, macros, depth=0):
    if depth > 5 or not macros:
        return t
    pat = re.compile(r"\\(%s)(?![A-Za-z@])" % "|".join(
        re.escape(n) for n in sorted(macros, key=len, reverse=True)))
    res, i = [], 0
    for m in pat.finditer(t):
        if m.start() < i:
            continue
        n, default, body = macros[m.group(1)]
        k, args = m.end(), []
        if default is not None:
            o = re.match(r"\s*\[([^\]]*)\]", t[k:])
            args.append(o.group(1) if o else default)
            k += o.end() if o else 0
        while len(args) < n:
            g, k2 = _tex_arg(t, k)
            if g is None:
                one = re.match(r"\s*(\\[A-Za-z]+|\S)", t[k:])
                if not one:
                    break
                g, k2 = one.group(1), k + one.end()
            args.append(g)
            k = k2
        for a_i, a in enumerate(args, 1):
            body = body.replace("#%d" % a_i, a)
        res += [t[i:m.start()], body]
        i = k
    res.append(t[i:])
    t2 = "".join(res)
    return t2 if t2 == t else _tex_expand(t2, macros, depth + 1)


def from_latex(text, path=None):
    """LaTeX, in stages: code is protected first, then the document body is
    taken (an example document inside the text does not cut it short),
    \\input and \\include are read in, the document's own commands are
    expanded, mathematics is protected, and only then are sections,
    environments, references, text commands and special characters turned
    into text; code and mathematics are put back last."""
    kept = []
    if path:
        text = _tex_includes(text, path, set())

    def keep(kind, body):
        kept.append((kind, body))
        return "\ue000%d\ue001" % (len(kept) - 1)

    # 1. code, printed exactly as typed
    def verb_env(m):
        name, body = m.group(1), m.group(2)
        if name == "minted":
            body = re.sub(r"^\s*(\[[^\]]*\])?\s*\{[^}]*\}", "", body, count=1)
        elif name.startswith(("lstlisting", "Verbatim", "BVerbatim",
                              "LVerbatim")):
            body = re.sub(r"^\s*\[[^\]]*\]", "", body, count=1)
        return "\n\n" + keep("code", body.strip("\n")) + "\n\n"
    s = re.sub(r"\\begin\{(%s)\}(.*?)\\end\{\1\}" % TEX_VERBATIM, verb_env,
               text, flags=re.S)
    s = re.sub(r"\\verb\*?([^A-Za-z*\s])(.*?)\1",
               lambda m: keep("inline", m.group(2)), s)
    s = re.sub(r"\\(lstinline|mintinline\s*\{[^}]*\})\s*(\[[^\]]*\])?"
               r"(?:\{([^}]*)\}|([^A-Za-z\s{])(.*?)\4)",
               lambda m: keep("inline", m.group(3) if m.group(3) is not None
                              else m.group(5)), s)

    # 2. comments; the document's own commands, defined in the preamble
    # too; then the document body
    s = _tex_strip_comments(s)
    s, macros, theorems = _tex_macros(s)
    dc = re.search(r"\\documentclass", s)
    first = re.search(r"\\(begin\{(?!document)|part\b|chapter\b|section\b)", s)
    if dc and not (first and first.start() < dc.start()):
        b, e = s.find("\\begin{document}", dc.end()), s.rfind("\\end{document}")
        if b >= 0:
            s = s[b + len("\\begin{document}"):e if e > b else None]
    else:
        s = re.sub(r"\\(begin|end)\{document\}", "", s)

    # 3. the document's own commands expanded
    s = _tex_expand(s, macros)

    # 4. mathematics, kept as LaTeX
    def math(kind, body):
        body = re.sub(r"\\label\s*\{[^}]*\}|\\nonumber\b|\\notag\b", "", body)
        return keep(kind, body.strip())
    s = re.sub(r"\\begin\{((?:%s)\*?)\}(.*?)\\end\{\1\}" % TEX_MATH,
               lambda m: "\n" + math("display", "\\begin{%s}%s\\end{%s}" % (
                   m.group(1), m.group(2), m.group(1))) + "\n", s, flags=re.S)
    for pat, kind in ((r"\$\$(.+?)\$\$", "display"), (r"\\\[(.+?)\\\]", "display"),
                      (r"\\\((.+?)\\\)", "math"),
                      (r"(?<!\\)\$(.+?)(?<!\\)\$", "math")):
        s = re.sub(pat, lambda m, k=kind: (("\n%s\n" if k == "display" else "%s")
                                           % math(k, m.group(1))), s, flags=re.S)

    # 5. sections
    used = [k for k in TEX_LEVELS if re.search(r"\\%s\*?\s*[\[{]" % k, s)]
    out, i = [], 0
    for m in re.finditer(r"\\(%s)\*?\s*(\[[^\]]*\])?\s*(?=\{)"
                         % "|".join(TEX_LEVELS), s):
        if m.start() < i:
            continue
        title, end = _tex_arg(s, m.end())
        if title is None:
            continue
        out.append(s[i:m.start()])
        out.append("\n\n\ue002%d %s\ue003\n\n" % (
            min(6, used.index(m.group(1)) + 1), " ".join(title.split())))
        i = end
    s = "".join(out) + s[i:]

    # 6. environments
    blocks = dict(TEX_BLOCKS, **theorems)
    s = re.sub(r"\\begin\{(%s)\}.*?\\end\{\1\}" % TEX_DROP, "", s, flags=re.S)
    s = re.sub(r"\\includegraphics\s*(\[[^\]]*\])?\s*\{([^}]*)\}",
               r"[image: \2]", s)
    s = re.sub(r"\\caption\s*(\[[^\]]*\])?\s*\{", "\n\nCaption: {", s)
    s = re.sub(r"\\begin\{(tabular\*?|tabularx|longtable|array|minipage|"
               r"wrapfigure)\}\s*(\[[^\]]*\])?\s*(\{(?:[^{}]|\{[^{}]*\})*\})?"
               r"(\s*\{(?:[^{}]|\{[^{}]*\})*\})?", "\n\n", s)

    def begin(m):
        base, title = m.group(1).rstrip("*"), m.group(3)
        if base in blocks:
            return "\n\n%s%s.\n" % (blocks[base], " (%s)" % title if title else "")
        if base in ("itemize", "enumerate", "description", "list"):
            return "\n"
        if title and base not in ("figure", "table", "center", "quote",
                                  "quotation", "verse"):
            return "\n\n%s (%s).\n" % (base.capitalize(), title)
        return "\n\n"
    s = re.sub(r"\\begin\{([A-Za-z*]+)\}(\[([^\]]*)\])?", begin, s)
    s = re.sub(r"\\end\{[A-Za-z*]+\}", "\n\n", s)
    s = re.sub(r"\\item\s*\[([^\]]*)\]", r"\n- \1: ", s)
    s = re.sub(r"\\item\b\s*", "\n- ", s)
    s = re.sub(r"(?<!\\)&", " | ", s)

    # 7. references and citations
    s = re.sub(r"\\label\s*\{[^}]*\}", "", s)
    s = re.sub(r"\\(eqref|ref|pageref|autoref|cref|Cref|nameref|vref|vpageref|"
               r"labelcref|crefrange)\*?\s*\{([^}]*)\}(\s*\{([^}]*)\})?",
               lambda m: "[ref: %s]" % (m.group(2) + (
                   "\u2013" + m.group(4) if m.group(4) else "")), s)
    s = re.sub(r"\\(cite|citep|citet|citealp|citealt|parencite|textcite|"
               r"autocite|footcite|citeauthor|citeyear|nocite)\*?\s*"
               r"(\[[^\]]*\]\s*){0,2}\{([^}]*)\}", r"[cite: \3]", s)
    s = re.sub(r"\\href\s*\{([^}]*)\}\s*\{([^}]*)\}", r"\2 (\1)", s)
    s = re.sub(r"\\(url|path)\s*\{([^}]*)\}", r"\2", s)

    # 8. text commands
    for _ in range(3):
        s = re.sub(r"\\footnote\s*(\[[^\]]*\])?\s*\{([^{}]*(\{[^{}]*\}[^{}]*)*)\}",
                   r" [footnote: \2]", s)
    for _ in range(4):
        s = re.sub(r"\\(%s)\*?\s*\{([^{}]*)\}" % TEX_STYLE, r"\2", s)
    s = re.sub(r"\\(%s)\*?\s*(\[[^\]]*\])?((\s*\{[^{}]*\})*)" % TEX_NOTTEXT, " ", s)
    s = re.sub(r"\\(%s)(?![A-Za-z])\s?" % TEX_LAYOUT, " ", s)

    # 9. special characters and accents
    def accent(m):
        ch = m.group(2) or m.group(3)
        ch = {"\\i": "i", "\\j": "j"}.get(ch, ch)
        return unicodedata.normalize("NFC", ch + TEX_ACCENTS[m.group(1)])
    s = re.sub(r"\\(['`^\"~=.]|[uvHckdbrt](?![A-Za-z]))\s*(?:\{(\\?[A-Za-z])\}|"
               r"(\\[ij](?![A-Za-z])|[A-Za-z]))", accent, s)
    s = re.sub(r"\\(%s)(?![A-Za-z])(\{\})?" % "|".join(
        sorted(TEX_WORDS, key=len, reverse=True)), lambda m: TEX_WORDS[m.group(1)], s)
    s = re.sub(r"\\([%&_#${}])", r"\1", s)
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = re.sub(r"\\\\\*?(\[[^\]]*\])?", "\n", s)
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    s = s.replace("``", "\u201c").replace("''", "\u201d")
    s = re.sub(r"(?<!\\)~", " ", s)
    # a command still left is the document's own: its arguments are kept
    # (several are separated by spaces), or its name when it has none
    for _ in range(3):
        s = re.sub(r"\\([A-Za-z@]+)\*?\s*(\[[^\]]*\])?((\s*\{[^{}]*\})+)",
                   lambda m: " ".join(re.findall(r"\{([^{}]*)\}", m.group(3))), s)
    s = re.sub(r"\\([A-Za-z@]+)\*?", r"\1", s)
    s = re.sub(r"(?<=[A-Za-z])\{(?=[A-Za-z])", " ", s)  # a!!a{word}: no gluing
    s = re.sub(r"(?<!\\)[{}]", "", s)

    # 10. code, mathematics and headings back in place
    def put(m):
        kind, body = kept[int(m.group(1))]
        return {"code": "\n```\n%s\n```\n", "inline": "`%s`",
                "display": "\n$$%s$$\n", "math": "$%s$"}[kind] % body
    for _ in range(3):
        s = re.sub("\ue000(\\d+)\ue001", put, s)
    s = re.sub("\ue002(\\d) ([^\ue003]*)\ue003",
               lambda m: "\n\n%s %s\n\n" % ("#" * int(m.group(1)),
                                            m.group(2).strip()), s)
    return _tidy(_guard_hashes_body(s))



def _guard_hashes_body(text):
    """Like _guard_hashes, but keeps the headings this module wrote, which
    always stand alone between blank lines."""
    lines = text.split("\n")
    out = []
    for k, l in enumerate(lines):
        alone = ((k == 0 or not lines[k - 1].strip())
                 and (k + 1 == len(lines) or not lines[k + 1].strip()))
        out.append(l if not l.startswith("#") or (alone and re.match(
            r"#{1,6} \S", l)) else " " + l)
    return "\n".join(out)


class _HTMLText:
    """Collect the text of an HTML page with '#' headings."""
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "svg",
            "button", "form", "template"}
    BLOCK = {"p", "div", "section", "article", "blockquote", "pre", "table",
             "tr", "ul", "ol", "dl", "dd", "dt", "figure", "figcaption",
             "main", "aside", "br", "hr"}

    def __init__(self):
        import html.parser
        outer = self

        class P(html.parser.HTMLParser):
            def handle_starttag(self, tag, attrs):
                outer.start(tag, dict(attrs))

            def handle_endtag(self, tag):
                outer.end(tag)

            def handle_data(self, data):
                outer.data(data)

        self.parser = P(convert_charrefs=True)
        self.out, self.skip, self.head, self.math = [], 0, None, None

    def start(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "math":
            self.math = {"tex": attrs.get("alttext"), "text": [],
                         "display": attrs.get("display") == "block",
                         "ann": False}
        elif self.math is not None and tag == "annotation" and \
                "tex" in (attrs.get("encoding") or ""):
            self.math["ann"], self.math["tex"] = True, ""
        elif re.fullmatch(r"h[1-6]", tag):
            self.head = (int(tag[1]), [])
        elif tag == "li":
            self.out.append("\n- ")
        elif tag in self.BLOCK:
            self.out.append("\n\n" if tag != "br" else "\n")

    def end(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "math" and self.math is not None:
            m, self.math = self.math, None
            tex = (m["tex"] or "".join(m["text"])).strip()
            self.out.append(("\n$$%s$$\n" if m["display"] else "$%s$") % tex)
        elif tag == "annotation" and self.math is not None:
            self.math["ann"] = False
        elif self.head and tag == "h%d" % self.head[0]:
            level, words = self.head
            self.head = None
            self.out.append(_heading(level, "".join(words)))
        elif tag in self.BLOCK:
            self.out.append("\n\n")

    def data(self, data):
        if self.skip:
            return
        if self.math is not None:
            if self.math["ann"]:
                self.math["tex"] += data
            elif not self.math["tex"]:
                self.math["text"].append(data)
            return
        (self.head[1] if self.head else self.out).append(data)


def from_html(text, path=None):
    """HTML: <h1>-<h6> become headings; scripts, styles and navigation are
    left out; MathML is replaced by its TeX annotation or alttext; MathJax
    source (\\( \\), $$ $$) is text already."""
    h = _HTMLText()
    h.parser.feed(text)
    h.parser.close()
    raw = "".join(h.out)
    lines = []
    for l in raw.split("\n"):
        lines.append(l if re.match(r"#{1,6} \S", l) else " ".join(l.split()))
    return _tidy(_guard_hashes_body("\n".join(lines)))


PTX_DIVISIONS = {"part": 1, "chapter": 1, "appendix": 1, "section": 2,
                 "exercises": 2, "references": 2, "reading-questions": 2,
                 "glossary": 2, "solutions": 2, "worksheet": 2,
                 "subsection": 3, "subsubsection": 4, "paragraphs": 5}
PTX_BLOCKS = {"theorem", "lemma", "proposition", "corollary", "definition",
              "example", "remark", "proof", "exercise", "algorithm", "claim",
              "fact", "axiom", "conjecture", "note", "observation", "warning",
              "insight", "question", "problem", "activity", "exploration",
              "hint", "answer", "solution", "assumption", "principle",
              "identity", "heuristic", "hypothesis", "objectives",
              "outcomes", "project", "investigation", "computation",
              "technology", "data", "convention", "notation-block"}
PTX_SKIP = {"sage", "idx", "notation", "latex-image", "image", "index",
            "docinfo", "macros", "latex-image-preamble", "video",
            "interactive", "program", "console", "sageplot", "asymptote",
            "frontmatter", "backmatter", "bookinfo"}
PTX_EMPTY = {"mdash": "\u2014", "ndash": "\u2013", "ellipsis": "\u2026",
             "nbsp": " ", "tex": "TeX", "latex": "LaTeX", "pretext": "PreTeXt",
             "xetex": "XeTeX", "ie": "i.e.", "eg": "e.g.", "etc": "etc.",
             "vs": "vs.", "copyright": "\u00a9", "times": "\u00d7",
             "lsq": "\u2018", "rsq": "\u2019", "lq": "\u201c", "rq": "\u201d",
             "today": "", "timeofday": "", "midpoint": "\u00b7",
             "swungdash": "\u2053", "permille": "\u2030", "pilcrow": "\u00b6",
             "section-mark": "\u00a7", "minus": "\u2212", "solidus": "/",
             "obelus": "\u00f7", "plusminus": "\u00b1", "degree": "\u00b0",
             "prime": "\u2032", "dblprime": "\u2033", "icon": ""}


def _xml_load(path, seen=None):
    """Parse an XML file and read its xi:include files in (relative to it)."""
    import xml.etree.ElementTree as ET
    seen = seen or set()
    path = Path(path).resolve()
    seen.add(path)
    root = ET.parse(path).getroot()
    xi = "{http://www.w3.org/2001/XInclude}include"

    def expand(el):
        kids = list(el)
        for k, ch in enumerate(kids):
            if ch.tag == xi:
                f = (path.parent / ch.get("href", "")).resolve()
                if ch.get("parse") == "text":
                    new = ET.Element("pre")
                    new.text = f.read_text(encoding="utf-8") if f.is_file() \
                        else ""
                elif f.is_file() and f not in seen:
                    new = _xml_load(f, seen)
                else:
                    new = ET.Element("skipped")
                new.tail = ch.tail
                el.remove(ch)
                el.insert(k, new)
            else:
                expand(ch)
    expand(root)
    return root


PTX_IDS = {}            # xml:id -> label, filled by from_xml
PTX_NUM = {}            # id(element) -> its number, filled by from_xml
PTX_NUMBERED = {"theorem", "lemma", "proposition", "corollary", "definition",
                "example", "remark", "algorithm", "claim", "fact", "axiom",
                "conjecture", "note", "observation", "principle", "identity",
                "insight", "assumption", "heuristic", "hypothesis",
                "question", "problem", "activity", "exploration",
                "investigation", "project", "computation", "technology",
                "data", "convention", "warning", "figure", "table",
                "listing", "list"}


def _ptx_number(root):
    """Number blocks as PreTeXt does by default: one counter per chapter
    shared by theorems, examples, figures and the like (Example 3.8,
    Table 3.10); exercises in an exercises division count on their own.
    The numbers are right only when the whole book is deposited."""
    nums, state = {}, {"ch": 0, "blk": 0}

    def walk(el, in_ex):
        tag = el.tag.split("}")[-1] if isinstance(el.tag, str) else ""
        if tag in PTX_SKIP:
            return
        if tag == "chapter":
            state["ch"] += 1
            state["blk"] = 0
            nums[id(el)] = str(state["ch"])
        elif tag in ("exercises", "reading-questions", "worksheet"):
            in_ex = [0]
        elif tag == "exercise" and in_ex:
            in_ex[0] += 1
            nums[id(el)] = str(in_ex[0])
        elif tag in PTX_NUMBERED or (tag == "exercise" and not in_ex):
            state["blk"] += 1
            nums[id(el)] = ("%d.%d" % (state["ch"], state["blk"])
                            if state["ch"] else str(state["blk"]))
        for ch in el:
            walk(ch, in_ex)
    walk(root, None)
    return nums


def _ptx_label(el):
    tag = el.tag.split("}")[-1]
    t = el.find("title")
    title = " ".join("".join(_ptx_inline(t)).split()) if t is not None else ""
    kind = tag.replace("-", " ").capitalize()
    num = PTX_NUM.get(id(el))
    if num:
        kind += " " + num
    xid = el.get("{http://www.w3.org/XML/1998/namespace}id")
    # an untitled, unnumbered block is named by its id, so a reference to
    # it can still be followed
    return "%s: %s" % (kind, title) if title else \
        (kind if num or not xid else "%s %s" % (kind, xid))


def _ptx_text(el, out, level_shift=0):
    tag = el.tag.split("}")[-1] if isinstance(el.tag, str) else ""
    if not tag:                                       # comment
        if el.tail:
            out.append(el.tail)
        return

    def kids():
        if el.text:
            out.append(el.text)
        for ch in el:
            _ptx_text(ch, out, level_shift)

    if tag in PTX_SKIP:
        pass
    elif tag in PTX_EMPTY and not len(el) and not (el.text or "").strip():
        out.append(PTX_EMPTY[tag])
    elif tag in PTX_DIVISIONS:
        t = el.find("title")
        title = "".join(_ptx_inline(t)) if t is not None else \
            tag.replace("-", " ").title()
        out.append(_heading(PTX_DIVISIONS[tag] + level_shift, title))
        if el.text:
            out.append(el.text)
        for ch in el:
            if ch is not t:
                _ptx_text(ch, out, level_shift)
    elif tag in ("title", "subtitle", "shorttitle", "plaintitle"):
        pass                                          # used by its parent
    elif tag == "m":
        out.append("$%s$" % (el.text or "").strip())
    elif tag in ("me", "men"):
        out.append("\n$$%s$$" % " ".join((el.text or "").split()))
    elif tag in ("md", "mdn"):
        rows = [" ".join((r.text or "").split()) for r in el.findall("mrow")]
        if len(rows) <= 1:
            out.append("\n$$%s$$" % (rows[0] if rows else
                                       " ".join((el.text or "").split())))
        else:
            body = " \\\\\n".join(rows) if rows else (el.text or "").strip()
            out.append("\n$$\\begin{aligned}\n%s\n\\end{aligned}$$" % body)
    elif tag == "c":
        out.append("`%s`" % (el.text or ""))
    elif tag == "q":
        out.append("\u201c")
        kids()
        out.append("\u201d")
    elif tag == "xref":
        ref = el.get("ref") or el.get("provisional") or ""
        out.append("[%s]" % ", ".join(PTX_IDS.get(r, "ref: " + r)
                                      for r in ref.replace(",", " ").split()))
    elif tag in ("url", "dataurl"):
        out.append(el.text or el.get("href") or "")
    elif tag == "li":
        out.append("\n- ")
        kids()
    elif tag in ("figure", "table", "listing", "list"):
        out.append("\n\n%s.\n" % _ptx_label(el))
        kids()
        out.append("\n\n")
    elif tag == "cell":
        kids()
        out.append(" | ")
    elif tag == "row":
        out.append("\n| ")
        kids()
        out.append("\n")
    elif tag in ("p", "ol", "ul", "dl", "statement", "introduction",
                 "conclusion", "blockquote", "pre", "cd", "tabular",
                 "biblio", "task", "sidebyside", "caption"):
        out.append("\n\n")
        kids()
        out.append("\n\n")
    elif tag in PTX_BLOCKS:
        t = el.find("title")
        label = _ptx_label(el)
        out.append("\n\n%s.\n" % label)
        if el.text:
            out.append(el.text)
        for ch in el:
            if ch is not t:
                _ptx_text(ch, out, level_shift)
        out.append("\n\n")
    else:
        kids()
    if el.tail:
        out.append(el.tail)


def _ptx_inline(el):
    out = []
    if el.text:
        out.append(el.text)
    for ch in el:
        _ptx_text(ch, out)
    return out


def from_xml(text, path=None):
    """XML. PreTeXt (and its older name, MathBook XML): chapters, sections and
    subsections become headings, xi:include files are read in, <m>, <me> and
    <md> become LaTeX, theorems, definitions and examples get a label, and
    Sage cells, index entries and pictures are left out. Other XML: its text."""
    import xml.etree.ElementTree as ET
    root = _xml_load(path) if path else ET.fromstring(text)
    out = []
    names = {e.tag.split("}")[-1] for e in root.iter() if isinstance(e.tag, str)}
    if names & {"pretext", "mathbook", "chapter", "section", "subsection"} \
            or "m" in names:
        xid = "{http://www.w3.org/XML/1998/namespace}id"
        PTX_NUM.clear()
        PTX_NUM.update(_ptx_number(root))
        PTX_IDS.clear()
        PTX_IDS.update((e.get(xid), _ptx_label(e)) for e in root.iter()
                       if isinstance(e.tag, str) and e.get(xid))
        _ptx_text(root, out)
        raw = "".join(out)
    else:
        raw = "\n\n".join(" ".join(t.split()) for t in root.itertext()
                          if t.strip())
    return _tidy(_guard_hashes_body(_unwrap(raw)))


def _unwrap(raw):
    """Join the lines of each paragraph; keep headings, '- ' items and
    displayed mathematics on lines of their own."""
    lines = [l if re.match(r"#{1,6} \S", l) else " ".join(l.split())
             for l in raw.split("\n")]
    text = re.sub(r"\n-\n+", "\n- ", "\n".join(lines))
    paras = []
    for para in re.split(r"\n\s*\n", text):
        out, in_math = [], False
        for l in para.split("\n"):
            if not l:
                continue
            starts = l.startswith("$$")
            if in_math or starts or l.startswith(("- ", "#")) or not out \
                    or out[-1].endswith("$$") and not l[:1] in ".,;:)":
                out.append(l)
            else:
                out[-1] += ("" if l[:1] in ".,;:)" else " ") + l
            if l.count("$$") % 2:
                in_math = not in_math
        paras.append("\n".join(out))
    return "\n\n".join(p for p in paras if p)


SOURCE_FORMATS = {".txt": None, ".text": None, ".md": from_markdown,
                  ".markdown": from_markdown, ".tex": from_latex,
                  ".latex": from_latex, ".html": from_html, ".htm": from_html,
                  ".xhtml": from_html, ".xml": from_xml, ".ptx": from_xml}


def source_to_text(path):
    """Read a book for deposit: (text, format name)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".pdf", ".docx", ".doc", ".epub", ".odt", ".rtf"):
        die("%s is not a source format. Look for the book's source first "
            "(LaTeX, PreTeXt XML, HTML or Markdown: arXiv and open textbooks "
            "often publish it); otherwise turn it into Markdown with: "
            "ingest %s --deposit" % (ext, path.name))
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        die("not a UTF-8 text file")
    conv = SOURCE_FORMATS.get(ext)
    if conv is None:
        return text, "text"
    try:
        return conv(text, path), ext.lstrip(".")
    except Exception as e:          # malformed XML and the like
        die("could not read %s as %s: %s" % (path.name, ext, e))


def select_parts(text, parts):
    """Keep only the top-level ('# ') divisions named in parts, by their
    position (1, 2, ...) or their title (case-insensitive)."""
    pieces, cur = [], None
    for line in text.split("\n"):
        if line.startswith("# "):
            cur = [line[2:].strip(), [line]]
            pieces.append(cur)
        elif cur:
            cur[1].append(line)
    if not pieces:
        die("--part needs '#' headings, and this book has none")
    keep, missing = [], []
    for want in parts:
        hit = [k for k, (title, _) in enumerate(pieces, 1)
               if want.strip() == str(k)
               or want.strip().casefold() == title.casefold()]
        if not hit:
            missing.append(want)
        keep += [k for k in hit if k not in keep]
    if missing:
        die("no top-level division %s; the book has: %s" % (
            ", ".join(repr(m) for m in missing), "; ".join(
                "%d %s" % (k, t) for k, (t, _) in enumerate(pieces, 1))))
    return "\n".join("\n".join(pieces[k - 1][1]) for k in sorted(keep)), \
        [pieces[k - 1][0] for k in sorted(keep)]


def cmd_deposit(a):
    """Store a book in the library as text with '#' headings, converted
    from its source format; identical content is refused."""
    st = Store(a.project)
    src = Path(a.file)
    if not src.is_file():
        die("no such file: %s" % src)
    text, fmt = source_to_text(src)
    name = src.name
    if getattr(a, "part", None):
        text, titles = select_parts(text, a.part)
        name = "%s [%s]" % (src.name, "; ".join(titles))
    if not text.strip():
        die("the file is empty")
    sha = hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()
    ldir, idx, books = _library(st)
    for b, info in books.items():
        if info["sha256"] == sha:
            die("same content already deposited as %s (%s)" % (b, info["name"]))
    bid = sha[:8]
    (ldir / (bid + ".txt")).write_text(text, encoding="utf-8")
    books[bid] = {"name": name, "sha256": sha, "chars": len(text),
                  "format": fmt, "date": time.strftime("%Y-%m-%d")}
    idx.write_text(json.dumps(books, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    heads = sum(1 for l in text.split("\n") if re.match(r"#{1,6} \S", l))
    print("deposited %s  %s  (%s, %d characters, %d headings)"
          % (bid, name, fmt, len(text), heads))


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
        title, body, stack = name, [], []
        for line in lines:
            if line.startswith("#"):
                if "\n".join(body).strip():
                    chunks.append((title, "\n".join(body).strip()))
                depth = len(line) - len(line.lstrip("#"))
                stack = [h for h in stack if h[0] < depth]
                stack.append((depth, line.lstrip("#").strip() or name))
                # the headings above it too: two sections may share a title
                title, body = " > ".join(h[1] for h in stack), []
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
    rows = [st.nodes[nid] for nid in st.subtree_ids(parent)]
    if not rows:
        die("the tree is empty; create files first")
    ok_targets = {n.id for n in rows}
    table = "\n".join("%s | %s | %s | %s" % (n.id, st.number(n.id),
                       LEVELS[n.level - 1], n.title) for n in rows)
    use_model(a)
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
            ans = call_model(st.root, a.backend, prompt_for(title, body))[0]
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


# ------------------------------------------------------- managing the trees
# Restructuring keeps ids, so paths (which store ids) stay valid; only the
# positional numbers change.

def _paths_using(st, ids):
    """{path name: [ids of the given set it uses]} over saved paths."""
    out = {}
    ids = set(ids)
    if st.pdir.is_dir():
        for f in sorted(st.pdir.glob("*.path")):
            try:
                lines = st.load_path(f.stem, True)       # comments removed
            except Die:
                continue
            toks = set(re.findall(r"\b[0-9a-f]{6}\b", " ".join(lines)))
            hit = sorted(toks & ids)
            if hit:
                out[f.stem] = hit
    return out


def _renumber(st, parent, keep_first=None):
    """Give the children of parent the orders 1, 2, ...; keep_first is a
    list of ids placed in that order (the rest follow as they were)."""
    kids = st.children(parent)
    if keep_first is not None:
        kids = [st.nodes[i] for i in keep_first]
    for k, n in enumerate(kids, 1):
        if n.order != k:
            n.order = k
            st.save(n)


def _place(st, nid, parent, position=None, before=None, after=None):
    """Put nid among the children of parent at a position (1-based) or
    before / after a sibling; default last."""
    sibs = [s.id for s in st.children(parent) if s.id != nid]
    if before:
        if before not in sibs:
            die("--before must name a child of the new parent")
        k = sibs.index(before)
    elif after:
        if after not in sibs:
            die("--after must name a child of the new parent")
        k = sibs.index(after) + 1
    elif position:
        k = max(0, min(len(sibs), position - 1))
    else:
        k = len(sibs)
    sibs.insert(k, nid)
    _renumber(st, parent, sibs)


def _shift_levels(st, ids, delta):
    deepest = max(st.nodes[i].level for i in ids) + delta
    if deepest > len(LEVELS):
        die("the branch would go deeper than level %d (%s)"
            % (len(LEVELS), LEVELS[-1]))
    if min(st.nodes[i].level for i in ids) + delta < 1:
        die("the branch would go above level 1")
    for i in ids:
        if delta:
            st.nodes[i].level += delta


def _target_parent(st, a):
    if getattr(a, "root", False):
        return ""
    if not a.to:
        die("give the new parent with --to REF, or --root for a new tree")
    return st.resolve(a.to)


def cmd_move(a):
    """Move a file and everything under it to a new parent."""
    st = Store(a.project)
    nid = st.resolve(a.ref)
    ids = st.subtree_ids(nid)
    parent = _target_parent(st, a)
    if parent in ids:
        die("a file cannot move under itself or its own branch")
    before = {i: st.number(i) for i in ids}
    old_parent = st.nodes[nid].parent
    new_level = st.nodes[parent].level + 1 if parent else (
        a.level and parse_level(a.level) or st.nodes[nid].level)
    delta = new_level - st.nodes[nid].level
    for i in ids:
        _backup(st, st.nodes[i])
    _shift_levels(st, ids, delta)
    st.nodes[nid].parent = parent
    for i in ids:
        st.save(st.nodes[i])
    _place(st, nid, parent, a.position,
           st.resolve(a.before) if a.before else None,
           st.resolve(a.after) if a.after else None)
    if old_parent != parent:
        _renumber(st, old_parent)
    for i in ids:
        print("%-10s -> %-10s [%s] %s" % (before[i], st.number(i), i,
                                         st.nodes[i].title))
    used = _paths_using(st, ids)
    if used:
        print("paths name files by id and need no change: "
              + ", ".join(sorted(used)))


def _among_deciders(st, nid):
    """Ids of files that choose nid with AMONG in some saved path."""
    out = set()
    for name in _paths_using(st, [nid]):
        try:
            flow = st.load_flow(name)
        except FlowError:
            continue
        out |= {e["src"] for e in flow.edges
                if e["among"] is not None and e["dst"] == nid}
    return sorted(out)


def cmd_rename(a):
    """Change a file's title; AMONG deciders that name it get the new one."""
    st = Store(a.project)
    nid = st.resolve(a.ref)
    n = st.nodes[nid]
    old, new = n.title, " ".join(a.title.split())
    if not new:
        die("the title is empty")
    if old == new:
        print("no change")
        return
    _backup(st, n)
    n.title = new
    st.save(n)
    print("%s [%s] %s -> %s" % (st.number(nid), nid, old, new))
    pat = re.compile(re.escape(old), re.I)
    for d in _among_deciders(st, nid):
        dn = st.nodes[d]
        if pat.search(dn.instructions) and not a.no_update:
            _backup(st, dn)
            dn.instructions = pat.sub(new, dn.instructions)
            st.save(dn)
            print("  updated the instructions of %s [%s] %s, which choose "
                  "among files by title" % (st.number(d), d, dn.title))
        elif pat.search(dn.instructions):
            print("  left as is (--no-update): %s [%s] names the old title"
                  % (st.number(d), d))
    others = [m for m in st.nodes.values() if m.id != nid
              and m.id not in _among_deciders(st, nid)
              and pat.search(m.instructions + "\n" + m.code)]
    for m in others:
        print("  note: %s [%s] %s mentions the old title in its machine side"
              % (st.number(m.id), m.id, m.title))


def cmd_copy(a):
    """Copy a file (or with --branch, its whole branch) under a new parent,
    with its sides, contracts, model and tags; links inside the copied set
    point to the copies."""
    st = Store(a.project)
    nid = st.resolve(a.ref)
    ids = st.subtree_ids(nid) if a.branch else [nid]
    parent = _target_parent(st, a)
    if a.branch and parent in ids:
        die("a branch cannot be copied into itself")
    base = st.nodes[nid].level
    new_level = st.nodes[parent].level + 1 if parent else base
    delta = new_level - base
    if max(st.nodes[i].level for i in ids) + delta > len(LEVELS):
        die("the copy would go deeper than level %d" % len(LEVELS))
    mapping = {}
    for i in ids:
        nn = uuid.uuid4().hex[:6]
        while nn in st.nodes or nn in mapping.values():
            nn = uuid.uuid4().hex[:6]
        mapping[i] = nn
    for i in ids:
        o = st.nodes[i]
        c = Node(id=mapping[i], title=o.title, level=o.level + delta,
                 parent=mapping.get(o.parent, parent) if i != nid else parent,
                 order=o.order, links=[(mapping.get(t, t), ty) for t, ty in o.links],
                 model=o.model, expects=o.expects, produces=o.produces,
                 built_sha=o.built_sha, human=o.human,
                 instructions=o.instructions, code=o.code, reading=o.reading,
                 tags=list(o.tags))
        st.save(c)
    if a.title:
        st.nodes[mapping[nid]].title = " ".join(a.title.split())
        st.save(st.nodes[mapping[nid]])
    _place(st, mapping[nid], parent)
    for i in ids:
        print("%s [%s] copied to %s [%s]" % (st.number(i), i,
                                             st.number(mapping[i]), mapping[i]))


def _links_into(st, ids, exclude=()):
    ids = set(ids)
    return [(m.id, t) for m in st.nodes.values() if m.id not in exclude
            for t, _ in m.links if t in ids]


def _remove_files(st, ids):
    for i in ids:
        _backup(st, st.nodes[i])
        st.file_of(i).unlink()
        del st.nodes[i]
        st.forget(i)


def cmd_rm(a):
    """Remove a file, or a branch with --branch, only when no saved path
    uses it; a copy of each file stays in history/ (revert brings it back)."""
    st = Store(a.project)
    nid = st.resolve(a.ref)
    ids = st.subtree_ids(nid)
    if len(ids) > 1 and not a.branch:
        die("%s has %d file(s) under it; move them first, or remove the whole "
            "branch with --branch" % (st.nodes[nid].title, len(ids) - 1))
    used = _paths_using(st, ids)
    if used:
        die("not removed: used by path(s) %s. Change or remove those paths "
            "first" % ", ".join("%s (%s)" % (p, ", ".join(v))
                                for p, v in sorted(used.items())))
    links = _links_into(st, ids, exclude=ids)
    if links and not a.unlink:
        die("not removed: linked from %s. Add --unlink to remove those links"
            % ", ".join("%s [%s]" % (st.nodes[s].title, s) for s, _ in links))
    for i in ids:
        print("%s %s [%s] %s" % ("remove" if a.yes else "would remove",
                                 st.number(i), i, st.nodes[i].title))
    if not a.yes:
        print("nothing removed; run again with --yes")
        return
    for s, _ in links:
        m = st.nodes[s]
        _backup(st, m)
        m.links = [(t, ty) for t, ty in m.links if t not in ids]
        st.save(m)
    parent = st.nodes[nid].parent
    _remove_files(st, ids)
    _renumber(st, parent)
    print("removed %d file(s); copies are in history/ (bring one back with: "
          "revert ID)" % len(ids))


def _passages(reading):
    """The reading side cut into its '## ' passages (text before the first
    one, if any, comes first)."""
    parts, cur = [], []
    for line in reading.split("\n"):
        if line.startswith("## ") and cur:
            parts.append("\n".join(cur).strip("\n"))
            cur = []
        cur.append(line)
    if "\n".join(cur).strip():
        parts.append("\n".join(cur).strip("\n"))
    return parts


def _pick(spec, count):
    out = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        try:
            lo, hi = int(lo), int(hi or lo)
        except ValueError:
            die("passages are given as numbers and ranges, such as 2,4-6")
        if not (1 <= lo <= hi <= count):
            die("passage numbers run from 1 to %d" % count)
        out |= set(range(lo, hi + 1))
    return sorted(out)


def cmd_split(a):
    """Move some reading passages of a file into a new file placed right
    after it; without --passages, list the passages."""
    st = Store(a.project)
    nid = st.resolve(a.ref)
    n = st.nodes[nid]
    parts = _passages(n.reading)
    if not a.passages:
        for k, p in enumerate(parts, 1):
            print("%3d  %s  (%d characters)" % (k, p.split("\n")[0][:70],
                                                 len(p)))
        print("split with: split %s --passages 3-5 --title TITLE" % a.ref)
        return
    if not a.title:
        die("give the new file a --title")
    picked = _pick(a.passages, len(parts))
    if len(picked) == len(parts):
        die("that is every passage; use move or rename instead")
    _backup(st, n)
    new = st.new_node(a.title, n.level, n.parent)
    new.reading = "\n\n".join(parts[k - 1] for k in picked)
    new.tags = list(n.tags)
    st.save(new)
    n.reading = "\n\n".join(p for k, p in enumerate(parts, 1)
                            if k not in picked)
    st.save(n)
    _place(st, new.id, n.parent, after=nid)
    print("%d passage(s) moved to %s [%s] %s" % (len(picked),
                                                 st.number(new.id), new.id,
                                                 new.title))
    if n.is_stale():
        print("note: %s [%s] is now stale (its reading side changed); "
              "review its machine side, then: confirm %s" % (n.title, nid,
                                                             nid))


def cmd_merge(a):
    """Merge the human and reading sides of B into A, then remove B
    (unless --keep). Links to B come to A."""
    st = Store(a.project)
    aid, bid = st.resolve(a.a), st.resolve(a.b)
    if aid == bid:
        die("a file cannot merge with itself")
    A, B = st.nodes[aid], st.nodes[bid]
    if not a.keep:
        if st.children(bid):
            die("%s has files under it; move them first, or use --keep"
                % B.title)
        if B.instructions.strip() or B.code.strip():
            die("%s has a machine side, which merge does not move; clear it "
                "or use --keep" % B.title)
        used = _paths_using(st, [bid])
        if used:
            die("%s is used by path(s) %s; change them first, or use --keep"
                % (B.title, ", ".join(sorted(used))))
    _backup(st, A)
    note = "<!-- merged from %s [%s] -->" % (B.title, bid)
    if B.human.strip():
        A.human = "\n\n".join(x for x in (A.human.strip("\n"), note,
                                          B.human.strip("\n")) if x)
    if B.reading.strip():
        A.reading = "\n\n".join(x for x in (A.reading.strip("\n"),
                                            B.reading.strip("\n")) if x)
    for t, ty in B.links:
        if t != aid and (t, ty) not in A.links:
            A.links.append((t, ty))
    A.tags = sorted(set(A.tags) | set(B.tags))
    st.save(A)
    print("merged %s [%s] into %s [%s]" % (B.title, bid, A.title, aid))
    if not a.keep:
        for s, _ in _links_into(st, [bid], exclude=[bid]):
            m = st.nodes[s]
            if m.id == aid:
                m.links = [(t, ty) for t, ty in m.links if t != bid]
            else:
                _backup(st, m)
                m.links = [(aid if t == bid else t, ty) for t, ty in m.links]
            st.save(m)
        parent = B.parent
        _remove_files(st, [bid])
        _renumber(st, parent)
        print("removed %s [%s]; a copy is in history/" % (B.title, bid))
    if A.is_stale():
        print("note: %s is now stale; review its machine side, then: "
              "confirm %s" % (A.title, aid))


# ---------------------------------------------------------------- history
def _versions(st, nid):
    """Saved versions of a file, newest first: [(label, path, Node)]."""
    hdir = st.root / "history"
    out = []
    if hdir.is_dir():
        for f in hdir.glob(nid + "-*.md"):
            m = re.fullmatch(re.escape(nid) + r"-(\d{8})-(\d{6})(?:-(\d+))?\.md",
                             f.name)
            if not m:
                continue
            try:
                node = Node.parse(f.read_text(encoding="utf-8"))
            except ValueError:
                continue
            key = (m.group(1), m.group(2), int(m.group(3) or 1))
            out.append((key, f, node))
    out.sort(key=lambda x: x[0], reverse=True)
    return [("%s-%s-%s %s:%s:%s" % (k[0][:4], k[0][4:6], k[0][6:], k[1][:2],
                                     k[1][2:4], k[1][4:]), f, node)
            for k, f, node in out]


HEADER_FIELDS = ("title", "parent", "level", "order", "links", "model",
                 "expects", "produces", "tags")


def _changes(old, new):
    """What differs between two versions of a file."""
    out = [s for s in SECTIONS
           if getattr(old, s).strip("\n") != getattr(new, s).strip("\n")]
    out += [h for h in HEADER_FIELDS if getattr(old, h) != getattr(new, h)]
    return out


def _resolve_any(st, ref):
    """A file id even when the file was removed (it has saved versions)."""
    if ref not in st.nodes and re.fullmatch(r"[0-9a-f]{6}", ref) \
            and _versions(st, ref):
        return ref
    return st.resolve(ref)


def cmd_log(a):
    """The saved versions of a file, newest first, and what changed."""
    st = Store(a.project)
    nid = _resolve_any(st, a.ref)
    cur = st.nodes.get(nid)
    vers = _versions(st, nid)
    if cur:
        print("current  %s [%s] %s%s" % (st.number(nid), nid, cur.title,
                                         "  (stale)" if cur.is_stale() else ""))
    else:
        print("removed  [%s]: bring it back with: revert %s" % (nid, nid))
    later = cur
    for k, (stamp, f, node) in enumerate(vers, 1):
        what = _changes(node, later) if later else []
        print("%3d  %s  %s" % (k, stamp, ("then changed: " + ", ".join(what))
                               if what else "(nothing changed after it)"))
        later = node
    if not vers:
        print("no saved versions yet (they are kept whenever a command "
              "changes the file)")


def _side_names(spec):
    if not spec:
        return list(SECTIONS)
    out = []
    for s in spec:
        s = {"h": "human", "i": "instructions", "c": "code", "r": "reading"
             }.get(s, s)
        if s == "machine":
            out += ["instructions", "code"]
        elif s in SECTIONS:
            out.append(s)
        else:
            die("unknown side '%s' (human, instructions, code, machine, "
                "reading)" % s)
    return out


def _version(st, nid, k):
    vers = _versions(st, nid)
    if not vers:
        die("no saved versions of %s" % nid)
    if not 1 <= k <= len(vers):
        die("versions run from 1 (newest) to %d; see: log %s" % (len(vers),
                                                                 nid))
    return vers[k - 1]


def cmd_diff(a):
    """Differences between a saved version (1 = newest) and the current
    file, side by side as a unified diff."""
    st = Store(a.project)
    nid = st.resolve(a.ref)
    stamp, f, old = _version(st, nid, a.version)
    cur = st.nodes[nid]
    shown = 0
    for h in HEADER_FIELDS:
        if getattr(old, h) != getattr(cur, h):
            print("%s: %s -> %s" % (h, getattr(old, h), getattr(cur, h)))
            shown += 1
    for s in _side_names(a.side):
        o, c = getattr(old, s).strip("\n"), getattr(cur, s).strip("\n")
        if o == c:
            continue
        shown += 1
        sys.stdout.writelines(difflib.unified_diff(
            (o + "\n").splitlines(True), (c + "\n").splitlines(True),
            "%s version %d (%s)" % (s, a.version, stamp), "%s current" % s))
    if not shown:
        print("no differences")


def cmd_revert(a):
    """Bring back sides of a saved version (1 = newest). The current file is
    saved first, so a revert can itself be reverted. A removed file comes
    back whole."""
    st = Store(a.project)
    nid = _resolve_any(st, a.ref)
    stamp, f, old = _version(st, nid, a.version)
    if nid not in st.nodes:
        if old.parent and old.parent not in st.nodes:
            die("its parent [%s] no longer exists; restore that first"
                % old.parent)
        st.save(old)
        _place(st, nid, old.parent)
        print("restored %s [%s] %s from %s" % (st.number(nid), nid, old.title,
                                               stamp))
        return
    cur = st.nodes[nid]
    sides = _side_names(a.side)
    _backup(st, cur)
    for s in sides:
        setattr(cur, s, getattr(old, s))
    if a.header:
        for h in ("title", "links", "model", "expects", "produces", "tags"):
            setattr(cur, h, getattr(old, h))
    if any(s in sides for s in ("instructions", "code")):
        # the machine side comes back with the sources it was built from
        cur.built_sha = old.built_sha
    st.save(cur)
    print("reverted %s of %s [%s] to %s%s" % (", ".join(sides), cur.title,
                                              nid, stamp,
                                              " (and its header)" if a.header
                                              else ""))
    if cur.is_stale():
        print("note: the machine side is older than the current human or "
              "reading side (stale)")


# ------------------------------------------------------- state, tags, sync
def cmd_status(a):
    """Only what is unfinished: stale files, steps without a machine side,
    machine sides without contracts, empty human sides, edits made outside
    Tawjih, edits left in work/."""
    st = Store(a.project)
    scope = st.subtree_ids(st.resolve(a.branch)) if a.branch else \
        st.subtree_ids("")
    in_paths = set()
    for v in _paths_using(st, scope).values():
        in_paths |= set(v)
    def has_machine(i):
        return bool(st.nodes[i].instructions.strip() or st.nodes[i].code.strip())
    groups = [
        ("stale (the human or reading side changed after the machine side "
         "was built)", [i for i in scope if st.nodes[i].is_stale()]),
        ("in a path but without a machine side",
         [i for i in scope if i in in_paths and not has_machine(i)]),
        ("machine side without contracts",
         [i for i in scope if has_machine(i)
          and not (st.nodes[i].expects or st.nodes[i].produces)]),
        ("machine side that no saved path uses",
         [i for i in scope if has_machine(i) and i not in in_paths]),
        ("empty human side", [i for i in scope if not st.nodes[i].human.strip()]),
        ("changed outside Tawjih since the last sync",
         [i for i in scope if st.outside_changes(i)]),
    ]
    total = 0
    for title, ids in groups:
        if not ids:
            continue
        total += len(ids)
        print("%s: %d" % (title, len(ids)))
        for i in ids:
            print("  %-10s [%s] %s" % (st.number(i), i, st.nodes[i].title))
    work = st.root / "work"
    if work.is_dir():
        left = sorted(d.name for d in work.iterdir() if d.is_dir())
        if left:
            total += len(left)
            print("edits left unfinished in work/: %s (see: sync "
                  "--clean-work)" % ", ".join(left))
    if not total:
        print("nothing unfinished in %d file(s)" % len(scope))


def _clean_tags(words):
    out = []
    for w in words:
        for t in w.split(","):
            t = t.strip().lstrip("#")
            if t and not re.fullmatch(r"[\w\-.]+", t):
                die("a tag is one word (letters, digits, '-', '_', '.'): " + t)
            if t:
                out.append(t)
    return out


def cmd_tag(a):
    """Add tags to a file; without tags, list every tag and its files."""
    st = Store(a.project)
    if not a.ref:
        tags = {}
        for n in st.nodes.values():
            for t in n.tags:
                tags.setdefault(t, []).append(n.id)
        for t in sorted(tags):
            print("%s (%d)" % (t, len(tags[t])))
            for i in sorted(tags[t], key=st.number):
                print("  %-10s [%s] %s" % (st.number(i), i, st.nodes[i].title))
        if not tags:
            print("no tags yet (add one with: tag REF WORD)")
        return
    nid = st.resolve(a.ref)
    n = st.nodes[nid]
    if not a.tags:
        print(", ".join(n.tags) or "no tags")
        return
    add = [t for t in _clean_tags(a.tags) if t not in n.tags]
    if add:
        n.tags = n.tags + add
        st.save(n)
    print("%s [%s] tags: %s" % (n.title, nid, ", ".join(n.tags)))


def cmd_untag(a):
    st = Store(a.project)
    nid = st.resolve(a.ref)
    n = st.nodes[nid]
    drop = set(_clean_tags(a.tags))
    missing = drop - set(n.tags)
    if missing:
        print("not tagged: " + ", ".join(sorted(missing)))
    n.tags = [t for t in n.tags if t not in drop]
    st.save(n)
    print("%s [%s] tags: %s" % (n.title, nid, ", ".join(n.tags) or "none"))


def cmd_sync(a):
    """Find files edited outside Tawjih since the last sync, record them,
    and with --clean-work remove unfinished edits left in work/."""
    while True:
        st = Store(a.project)
        changed = {i: st.outside_changes(i) for i in st.nodes}
        changed = {i: c for i, c in changed.items() if c}
        new = [i for i in st.nodes if i not in st.state]
        gone = [i for i in st.state if i not in st.nodes]
        for i, sides in sorted(changed.items(), key=lambda x: st.number(x[0])):
            n = st.nodes[i]
            note = ""
            if "machine" in sides and a.confirm_machine:
                n.built_sha = n.source_digest()
                st.save(n)
                note = "  (machine side recorded as built from the current "\
                       "sides)"
            elif "machine" in sides and n.built_sha:
                note = "  (if you rebuilt the machine side by hand: sync "\
                       "--confirm-machine)"
            print("changed outside Tawjih: %-10s [%s] %s: %s%s" % (
                st.number(i), i, n.title, ", ".join(sides), note))
            if n.is_stale():
                print("  now stale: the machine side is older than its "
                      "sources")
        for i in new:
            if st.state:
                print("new file, not made by Tawjih: %s [%s] %s" % (
                    st.number(i), i, st.nodes[i].title))
        for i in gone:
            print("removed outside Tawjih: [%s]" % i)
        st.record_all()
        work = st.root / "work"
        if work.is_dir():
            for d in sorted(x for x in work.iterdir() if x.is_dir()):
                if a.clean_work:
                    shutil.rmtree(str(d), ignore_errors=True)
                    print("removed the unfinished edit in work/%s" % d.name)
                else:
                    print("unfinished edit in work/%s (open in an editor, or "
                          "left by an interrupted open); remove it with "
                          "--clean-work" % d.name)
            if a.clean_work:
                with contextlib.suppress(OSError):
                    work.rmdir()
        if not (changed or gone or (new and st.state)):
            print("in sync: %d file(s)" % len(st.nodes))
        if not a.watch:
            return
        try:
            time.sleep(a.watch)
        except KeyboardInterrupt:
            return


def cmd_refs(a):
    """Everything that refers to a file: links in and out, paths and their
    neighbours, AMONG choices, shared tags, mentions of its title."""
    st = Store(a.project)
    nid = st.resolve(a.ref)
    n = st.nodes[nid]

    def name(i):
        return "%s [%s] %s" % (st.number(i), i, st.nodes[i].title) \
            if i in st.nodes else "[%s] (missing)" % i
    print("%s" % name(nid))
    if n.parent:
        print("  parent: " + name(n.parent))
    kids = st.children(nid)
    if kids:
        print("  files under it: %d" % len(st.subtree_ids(nid)[1:]))
    for t, ty in n.links:
        print("  links to: %s (%s)" % (name(t), ty))
    for s, _ in _links_into(st, [nid]):
        ty = next(ty for t, ty in st.nodes[s].links if t == nid)
        print("  linked from: %s (%s)" % (name(s), ty))
    for p in sorted(_paths_using(st, [nid])):
        try:
            flow = st.load_flow(p)
        except FlowError as e:
            print("  path %s: unreadable (%s)" % (p, e))
            continue
        ins = [e for e in flow.edges if e["dst"] == nid]
        outs = [e for e in flow.edges if e["src"] == nid]
        print("  path %s: from %s; to %s" % (
            p, ", ".join(name(e["src"]) + (" (AMONG)" if e["among"] is not None
                                           else "") for e in ins) or "the request",
            ", ".join(name(e["dst"]) for e in outs) or "the result"))
    for t in n.tags:
        same = [m.id for m in st.nodes.values() if t in m.tags and m.id != nid]
        if same:
            print("  tag %s also on: %s" % (t, "; ".join(name(i) for i in same)))
    pat = re.compile(re.escape(n.title), re.I)
    for m in st.nodes.values():
        if m.id != nid and pat.search(m.instructions + "\n" + m.code):
            print("  machine side mentions its title: " + name(m.id))


def cmd_clean(a):
    """Report (and with --yes, fix) leftovers: unfinished edits in work/,
    old runs, broken links, files whose parent is missing, paths naming
    missing files."""
    st = Store(a.project)
    acts = []
    work = st.root / "work"
    if work.is_dir():
        for d in sorted(x for x in work.iterdir() if x.is_dir()):
            acts.append(("unfinished edit work/%s" % d.name,
                         lambda d=d: shutil.rmtree(str(d), ignore_errors=True)))
    if a.runs is not None and st.rdir.is_dir():
        limit = time.time() - a.runs * 86400
        for f in sorted(st.rdir.iterdir()):
            if f.is_file() and f.stat().st_mtime < limit:
                acts.append(("run record older than %d day(s): runs/%s"
                             % (a.runs, f.name), lambda f=f: f.unlink()))
    for n in list(st.nodes.values()):
        bad = [t for t, _ in n.links if t not in st.nodes]
        if bad:
            def fix(n=n, bad=bad):
                _backup(st, n)
                n.links = [(t, ty) for t, ty in n.links if t not in bad]
                st.save(n)
            acts.append(("broken link(s) in %s [%s]: %s" % (
                n.title, n.id, ", ".join(bad)), fix))
        if n.parent and n.parent not in st.nodes:
            print("problem (not fixed automatically): %s [%s] has a missing "
                  "parent [%s]; give it one with move" % (n.title, n.id,
                                                          n.parent))
    if st.pdir.is_dir():
        for f in sorted(st.pdir.glob("*.path")):
            try:
                st.load_flow(f.stem)
            except FlowError as e:
                print("problem (not fixed automatically): path %s: %s"
                      % (f.stem, e))
    for what, _ in acts:
        print(("fixed: " if a.yes else "found: ") + what)
    if a.yes:
        for _, fn in acts:
            fn()
        with contextlib.suppress(OSError):
            work.rmdir()
    elif acts:
        print("nothing changed; run again with --yes")
    if not acts:
        print("nothing to clean")


# -------------------------------------------------------------------- audit
AUDIT_CALLS = {
    "os.system": "runs a shell command", "os.popen": "runs a shell command",
    "os.remove": "deletes a file", "os.unlink": "deletes a file",
    "os.rmdir": "deletes a folder", "os.removedirs": "deletes folders",
    "os.rename": "moves a file", "os.replace": "overwrites a file",
    "os.chmod": "changes permissions", "os.chown": "changes ownership",
    "os.kill": "stops a process", "os.fork": "starts a process",
    "shutil.rmtree": "deletes a folder tree", "shutil.move": "moves files",
    "shutil.copy": "writes files", "shutil.copyfile": "writes files",
    "shutil.copytree": "writes files", "shutil.make_archive": "writes files",
    "eval": "runs text as code", "exec": "runs text as code",
    "compile": "turns text into code", "__import__": "imports by name",
    "getattr": "reaches attributes by name (cannot be checked)",
    "globals": "changes the program's state", "setattr": "changes objects by name",
}
AUDIT_PREFIX = {"subprocess.": "runs programs", "os.exec": "replaces the process",
                "os.spawn": "starts programs", "socket.": "opens network "
                "connections", "urllib.": "uses the network", "http.": "uses "
                "the network", "requests.": "uses the network", "ctypes.":
                "calls machine code", "importlib.": "imports by name",
                "pickle.load": "runs code from data", "marshal.load":
                "runs code from data", "shelve.": "writes files"}
AUDIT_METHODS = {"unlink": "deletes a file", "rmdir": "deletes a folder",
                 "write_text": "writes a file", "write_bytes": "writes a file",
                 "touch": "creates a file", "chmod": "changes permissions",
                 "rename": "moves a file", "replace": "overwrites a file",
                 "mkdir": "creates a folder"}
AUDIT_MODULES = {"subprocess", "socket", "ctypes", "urllib", "http", "requests",
                 "shutil", "importlib", "pickle", "marshal", "multiprocessing",
                 "threading", "asyncio", "ftplib", "smtplib", "telnetlib",
                 "webbrowser"}


def audit_code(code):
    """Findings in a code side: [(line, level, message)], level 'danger'
    or 'note'. A static check: it finds what is written, not everything a
    program can do."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [(e.lineno or 0, "danger", "syntax error: %s" % e.msg)]
    alias, out = {}, []

    def dotted(node):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(alias.get(node.id, node.id))
            return ".".join(reversed(parts))
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else None
            for al in node.names:
                if mod:
                    alias[al.asname or al.name] = mod + "." + al.name
                elif al.asname:
                    alias[al.asname] = al.name
                else:
                    alias[al.name.split(".")[0]] = al.name.split(".")[0]
                top = (mod or al.name).split(".")[0]
                if top in AUDIT_MODULES:
                    out.append((node.lineno, "danger" if top in (
                        "subprocess", "socket", "ctypes", "shutil",
                        "multiprocessing") else "note",
                        "imports %s" % (mod or al.name)))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = dotted(node.func)
        if name is None:
            if isinstance(node.func, ast.Attribute) and \
                    node.func.attr in AUDIT_METHODS:
                out.append((node.lineno, "danger", "%s: %s" % (
                    node.func.attr, AUDIT_METHODS[node.func.attr])))
            continue
        msg = AUDIT_CALLS.get(name)
        if not msg:
            msg = next((m for p, m in AUDIT_PREFIX.items()
                        if name.startswith(p)), None)
        if not msg and isinstance(node.func, ast.Attribute) and \
                node.func.attr in AUDIT_METHODS and not name.startswith(
                    ("vars.", "steps.", "str.")):
            msg = AUDIT_METHODS[node.func.attr]
        if name == "open":
            mode = ""
            if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                mode = str(node.args[1].value)
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            if set(mode) & set("wax+"):
                msg = "opens a file for writing"
            elif len(node.args) > 1 and not isinstance(node.args[1],
                                                        ast.Constant):
                msg = "opens a file in a mode that cannot be checked"
        if msg:
            level = "note" if name in ("getattr", "setattr", "globals") \
                or msg.endswith("(cannot be checked)") else "danger"
            out.append((node.lineno, level, "%s: %s" % (name, msg)))
    return sorted(set(out))


def cmd_audit(a):
    """Look through code sides for what touches the system: running
    programs, deleting or writing files, the network, running text as code.
    A warning, not a sandbox: it cannot prove code harmless."""
    st = Store(a.project)
    if a.path:
        ids = list(_flow_from_args(st, a.path, []).nodes)
    elif a.ref:
        ids = st.subtree_ids(st.resolve(a.ref))
    else:
        ids = st.subtree_ids("")
    dangers = 0
    checked = 0
    for i in ids:
        n = st.nodes[i]
        if not n.code.strip():
            continue
        checked += 1
        found = audit_code(n.code)
        if not found:
            continue
        print("%s [%s] %s" % (st.number(i), i, n.title))
        for line, level, msg in found:
            print("  line %-4s %-6s %s" % (line, level, msg))
            dangers += level == "danger"
    print("%d code side(s) checked, %d finding(s) to read before "
          "--allow-code. This is a check of what the code says, not a "
          "sandbox." % (checked, dangers))
    return 1 if dangers else 0


def audit_warnings(st, flow):
    """Short warnings printed before a run with --allow-code."""
    for i in flow.nodes:
        found = [f for f in audit_code(st.nodes[i].code)
                 if f[1] == "danger"] if st.nodes[i].code.strip() else []
        if found:
            print("  audit: %s [%s]: %s" % (st.nodes[i].title, i, "; ".join(
                "line %d %s" % (l, m) for l, _, m in found[:3])),
                file=sys.stderr)


# ---------------------------------------------------------- ranked search
def _words(text):
    return [w.casefold() for w in re.findall(r"\w+", text) if len(w) > 1]


def _search_chunks(st, ids, sides):
    """Searchable pieces: the title, each reading passage, each paragraph
    of the other sides. [(id, side, text)]"""
    out = []
    for i in ids:
        n = st.nodes[i]
        out.append((i, "title", n.title))
        for s in sides:
            text = getattr(n, s)
            if not text.strip():
                continue
            pieces = _passages(text) if s == "reading" else \
                [p for p in re.split(r"\n\s*\n", text) if p.strip()]
            for p in pieces:
                out.append((i, s, p.strip()))
    return out


def bm25(query, chunks, k1=1.5, b=0.75):
    """Okapi BM25 scores of chunks [(.., .., text)] for a query."""
    docs = [_words(c[2]) for c in chunks]
    if not docs:
        return []
    avg = sum(len(d) for d in docs) / len(docs) or 1
    df = {}
    for d in docs:
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    q = _words(query)
    scores = []
    for d in docs:
        tf, s = {}, 0.0
        for w in d:
            tf[w] = tf.get(w, 0) + 1
        for w in q:
            if w in tf:
                idf = math.log(1 + (len(docs) - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (
                    tf[w] + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    return scores


# embeddings: a provider turns texts into vectors
def embed_ollama(texts, model):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if not host.startswith("http"):
        host = "http://" + host
    return _post(host + "/api/embed", {"model": model, "input": texts},
                 {})["embeddings"]


def embed_openai(texts, model):
    """The same provider settings as the openai backend."""
    base = os.environ.get("TAWJIH_BASE_URL", "https://api.openai.com/v1")
    key = os.environ.get("TAWJIH_API_KEY", "")
    data = _post(base.rstrip("/") + "/embeddings",
                 {"model": model, "input": texts},
                 {"Authorization": "Bearer " + key} if key else {})["data"]
    return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


EMBEDDERS = {"ollama": embed_ollama, "openai": embed_openai}


def _cosine(u, v):
    num = sum(x * y for x, y in zip(u, v))
    den = math.sqrt(sum(x * x for x in u)) * math.sqrt(sum(y * y for y in v))
    return num / den if den else 0.0


def _index_file(st):
    return st.root / "index" / "embeddings.json"


def cmd_index(a):
    """Build or update the semantic index of the human and reading sides
    (and titles) with an embedding model. Only new or changed pieces are
    sent to the provider."""
    st = Store(a.project)
    model = a.model or os.environ.get("TAWJIH_EMBED_MODEL", "nomic-embed-text")
    f = _index_file(st)
    old = {}
    if f.is_file():
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("backend") == a.backend and data.get("model") == model:
            old = {c["sha"]: c["vector"] for c in data["chunks"]}
    chunks = _search_chunks(st, st.subtree_ids(""), ["human", "reading"])
    rows, todo = [], []
    for i, side, text in chunks:
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        rows.append({"id": i, "side": side, "sha": sha, "text": text[:4000]})
        if sha not in old:
            todo.append(rows[-1])
    print("%d piece(s), %d to embed with %s/%s" % (len(rows), len(todo),
                                                   a.backend, model),
          file=sys.stderr)
    for k in range(0, len(todo), 32):
        batch = todo[k:k + 32]
        try:
            vecs = EMBEDDERS[a.backend]([r["text"] for r in batch], model)
        except Exception as e:
            die("the embedding provider failed: %s: %s" % (type(e).__name__, e))
        for r, v in zip(batch, vecs):
            old[r["sha"]] = v
    for r in rows:
        r["vector"] = old[r["sha"]]
        r.pop("text")
    f.parent.mkdir(exist_ok=True)
    f.write_text(json.dumps({"backend": a.backend, "model": model,
                             "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                             "chunks": rows}), encoding="utf-8")
    print("index: %s (%d pieces)" % (f, len(rows)))


def search_ranked(st, query, ids, sides, semantic=False, top=10):
    chunks = _search_chunks(st, ids, sides)
    if semantic:
        f = _index_file(st)
        if not f.is_file():
            die("no semantic index yet; build it with: index --backend ollama")
        data = json.loads(f.read_text(encoding="utf-8"))
        vecs = {c["sha"]: c["vector"] for c in data["chunks"]}
        try:
            q = EMBEDDERS[data["backend"]]([query], data["model"])[0]
        except Exception as e:
            die("the embedding provider failed: %s: %s" % (type(e).__name__, e))
        scored, missing = [], 0
        for c in chunks:
            sha = hashlib.sha256(c[2].encode("utf-8")).hexdigest()[:16]
            if sha in vecs:
                scored.append((_cosine(q, vecs[sha]), c))
            elif c[1] in ("human", "reading", "title"):
                missing += 1
        if missing:
            print("note: %d piece(s) changed since the index was built; run "
                  "index again" % missing, file=sys.stderr)
    else:
        scored = list(zip(bm25(query, chunks), chunks))
        scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: -x[0])
    return scored[:top]


def cmd_search(a):
    """Search across trees or within a branch and side: literal by default,
    ranked by relevance with --rank (BM25), or by meaning with --semantic
    (an index built by the index command)."""
    st = Store(a.project)
    root = st.resolve(a.branch) if a.branch else ""
    ids = st.subtree_ids(root)
    if a.tag:
        ids = [i for i in ids if a.tag in st.nodes[i].tags]
    side_map = {"h": "human", "human": "human",
                "i": "instructions", "instructions": "instructions",
                "c": "code", "code": "code",
                "r": "reading", "reading": "reading"}
    if a.side and a.side.lower() not in side_map:
        die("unknown side '%s'" % a.side)
    target_sides = [side_map[a.side.lower()]] if a.side else SECTIONS
    if a.rank or a.semantic:
        hits = search_ranked(st, a.query, ids, target_sides, a.semantic,
                             a.top)
        for score, (i, side, text) in hits:
            print("%6.3f  %-8s [%s] %s  (%s)" % (score, st.number(i), i,
                                                 st.nodes[i].title, side))
            print("         " + " ".join(text.split())[:110])
        print("%d result(s)" % len(hits))
        return
    q = a.query.casefold()
    hits = 0
    for nid in ids:
        n = st.nodes[nid]
        matched = []
        if q in n.title.casefold() or q in nid.casefold():
            matched.append(("title", n.title))
        for sec in target_sides:
            text = getattr(n, sec)
            if q in text.casefold():
                line = next((l.strip() for l in text.splitlines()
                             if q in l.casefold()), "")
                matched.append((sec, line[:90]))
        if matched:
            hits += 1
            num = st.number(nid)
            print("%-8s [%s] %-8s %s" % (num, nid, LEVELS[n.level - 1],
                                         n.title))
            for sec, snippet in matched:
                print("         (%s) %s" % (sec, snippet))
    print("%d matching file(s)" % hits)


# ------------------------------------------------------------------ export
def _md_to_html(md):
    """The small Markdown that export writes: headings, paragraphs, '>'
    quotes, '- ' lists, fenced code. Mathematics stays as written."""
    import html as H

    def inline(t):
        t = H.escape(t)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        return re.sub(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])", r"<em>\1</em>", t)
    out, para, quote, items, code = [], [], [], [], None

    def flush():
        if para:
            out.append('<p dir="auto">%s</p>' % inline(" ".join(para)))
            para.clear()
        if quote:
            paras, cur = [], []
            for q in quote + [""]:
                if q:
                    cur.append(inline(q))
                elif cur:
                    paras.append("<br>".join(cur))
                    cur = []
            out.append('<blockquote dir="auto">%s</blockquote>' % "".join(
                "<p>%s</p>" % p for p in paras))
            quote.clear()
        if items:
            out.append("<ul>%s</ul>" % "".join('<li dir="auto">%s</li>'
                                               % inline(x) for x in items))
            items.clear()
    for line in md.split("\n"):
        if code is not None:
            if line.startswith("```"):
                out.append("<pre><code>%s</code></pre>" % H.escape(
                    "\n".join(code)))
                code = None
            else:
                code.append(line)
            continue
        if line.startswith("```"):
            flush()
            code = []
            continue
        m = re.match(r"(#{1,6}) (.*)", line)
        if m:
            flush()
            out.append('<h%d dir="auto">%s</h%d>' % (
                len(m.group(1)), inline(m.group(2)), len(m.group(1))))
        elif line.startswith(">"):
            if para or items:
                flush()
            quote.append(line[1:].strip())
        elif line.startswith("- "):
            if para or quote:
                flush()
            items.append(line[2:])
        elif not line.strip():
            if quote:
                quote.append("")
            else:
                flush()
        else:
            if quote or items:
                flush()
            para.append(line.strip())
    flush()
    return "\n".join(out)


def cmd_export(a):
    """The trees (or one branch) as one readable book: the human side of
    each file and, unless left out, its reading passages with sources;
    --machine adds the machine side."""
    st = Store(a.project)
    root = st.resolve(a.branch) if a.branch else ""
    ids = st.subtree_ids(root)
    if not ids:
        die("nothing to export")
    top = min(st.nodes[i].level for i in ids)
    parts = []
    for i in ids:
        n = st.nodes[i]
        depth = min(6, n.level - top + 1)
        parts.append("%s %s %s" % ("#" * depth, st.number(i), n.title))
        if n.tags:
            parts.append("*%s*" % ", ".join(n.tags))
        if n.human.strip():
            parts.append(n.human.strip())
        if a.machine and (n.instructions.strip() or n.code.strip()):
            if n.instructions.strip():
                parts.append("**Instructions**\n\n" + n.instructions.strip())
            if n.code.strip():
                parts.append("```python\n%s\n```" % n.code.strip())
        if n.reading.strip() and not a.no_reading:
            for p in _passages(n.reading):
                lines = p.split("\n")
                if lines[0].startswith("## "):
                    lines[0] = "**%s**" % lines[0][3:]
                parts.append("\n".join("> " + l if l.strip() else ">"
                                       for l in lines))
    md = "\n\n".join(parts) + "\n"
    title = st.nodes[root].title if root else "Tawjih project"
    fmt = a.format or ("html" if (a.out or "").lower().endswith((".html",
                                                                 ".htm"))
                       else "md")
    if fmt == "html":
        import html as H
        mathjax = ""
        if not a.no_mathjax and re.search(r"\$[^$\n]+\$", md):
            # formulas are shown by MathJax, loaded from the web when the
            # page is opened; --no-mathjax keeps the page offline
            mathjax = ("<script>window.MathJax={tex:{inlineMath:[['$','$']]}};"
                       "</script><script defer src=\"https://cdn.jsdelivr.net/"
                       "npm/mathjax@3.2.2/es5/tex-chtml.js\"></script>")
        text = ("<!doctype html>\n<html><head><meta charset=\"utf-8\">"
                "<meta name=\"viewport\" content=\"width=device-width, "
                "initial-scale=1\"><title>%s</title><style>body{max-width:"
                "46rem;margin:2rem auto;padding:0 1rem;font-family:serif;"
                "line-height:1.6}blockquote{border-inline-start:3px solid #bbb;"
                "margin:1rem 0;padding:0 1rem;color:#333}pre{background:#f4f4f4;"
                "padding:.8rem;overflow:auto}</style>%s</head><body>\n%s\n"
                "</body></html>\n" % (H.escape(title), mathjax,
                                       _md_to_html(md)))
    else:
        text = md
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print("exported %d file(s) to %s (%s)" % (len(ids), a.out, fmt))
    else:
        sys.stdout.write(text)


# -------------------------------------------------------------- pack, unpack
def cmd_pack(a):
    """A branch (or the files of paths) with their sides, contracts and the
    paths among them, in one zip file for another project."""
    import zipfile
    st = Store(a.project)
    ids, names = [], list(a.path or [])
    for ref in a.branch or []:
        ids += [i for i in st.subtree_ids(st.resolve(ref)) if i not in ids]
    for name in names:
        ids += [i for i in _flow_from_args(st, name, []).nodes if i not in ids]
    if not ids:
        die("give --branch REF and/or --path NAME")
    for name, used in _paths_using(st, ids).items():
        try:
            if name not in names and set(st.load_flow(name).nodes) <= set(ids):
                names.append(name)
        except FlowError:
            pass
    roots = [i for i in ids if st.nodes[i].parent not in ids]
    manifest = {"tawjih": __version__, "created": time.strftime(
        "%Y-%m-%d %H:%M:%S"), "files": ids, "roots": roots,
        "paths": sorted(names)}
    out = Path(a.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False,
                                               indent=2))
        for i in ids:
            z.write(st.file_of(i), "files/%s.md" % i)
        for name in names:
            z.write(st.pdir / (name + ".path"), "paths/%s.path" % name)
    print("packed %d file(s) and %d path(s) into %s" % (len(ids), len(names),
                                                       out))


def cmd_unpack(a):
    """Add a pack to this project: its trees become new trees, or go under
    --parent. Ids that already exist are refused unless --new-ids."""
    import zipfile
    st = Store(a.project)
    try:
        z = zipfile.ZipFile(a.file)
    except (OSError, zipfile.BadZipFile) as e:
        die("not a pack: %s" % e)
    with z:
        bad = [n for n in z.namelist() if not re.fullmatch(
            r"manifest\.json|files/[0-9a-f]{6}\.md|paths/[\w\-]+\.path", n)]
        if bad:
            die("not a Tawjih pack (unexpected entries: %s)" % ", ".join(bad[:5]))
        try:
            man = json.loads(z.read("manifest.json"))
            files, pnames = list(man["files"]), list(man.get("paths", []))
        except (KeyError, ValueError, TypeError):
            die("the pack has no readable manifest.json")
        if not all(isinstance(i, str) and re.fullmatch(r"[0-9a-f]{6}", i)
                   for i in files) or not all(
                isinstance(p, str) and re.fullmatch(r"[\w\-]+", p)
                for p in pnames):
            die("the pack's manifest names files or paths that are not valid")
        nodes = {}
        for i in files:
            try:
                nodes[i] = Node.parse(z.read("files/%s.md" % i).decode("utf-8"))
            except (KeyError, ValueError) as e:
                die("pack file %s: %s" % (i, e))
        try:
            paths = {p: z.read("paths/%s.path" % p).decode("utf-8")
                     for p in pnames}
        except (KeyError, UnicodeDecodeError) as e:
            die("pack path: %s" % e)
    clash = [i for i in nodes if i in st.nodes]
    if clash and not a.new_ids:
        die("%d file id(s) already exist here (%s); add --new-ids to give the "
            "unpacked files new ones" % (len(clash), ", ".join(clash[:5])))
    mapping = {i: i for i in nodes}
    if a.new_ids:
        for i in nodes:
            nn = uuid.uuid4().hex[:6]
            while nn in st.nodes or nn in mapping.values():
                nn = uuid.uuid4().hex[:6]
            mapping[i] = nn
    parent = st.resolve(a.parent) if a.parent else ""
    roots = [i for i in nodes if nodes[i].parent not in nodes]
    for r in roots:
        delta = (st.nodes[parent].level + 1 - nodes[r].level) if parent else 0
        sub = [i for i in nodes if r in _chain(nodes, i)]
        if max(nodes[i].level for i in sub) + delta > len(LEVELS):
            die("%s would go deeper than level %d under that parent"
                % (nodes[r].title, len(LEVELS)))
        for i in sub:
            nodes[i].level += delta
    for i, n in nodes.items():
        n.id = mapping[i]
        n.parent = mapping[n.parent] if n.parent in nodes else (
            parent if i in roots else n.parent)
        n.links = [(mapping.get(t, t), ty) for t, ty in n.links]
        st.save(n)
    for r in roots:
        _place(st, mapping[r], parent)
    for name, text in paths.items():
        for old, new in mapping.items():
            text = re.sub(r"\b%s\b" % old, new, text)
        target, k = name, 1
        while (st.pdir / (target + ".path")).exists():
            k += 1
            target = "%s-%d" % (name, k)
        st.pdir.mkdir(exist_ok=True)
        (st.pdir / (target + ".path")).write_text(text, encoding="utf-8")
        print("path %s%s" % (target, "" if target == name else
                             " (renamed: %s exists)" % name))
    for r in roots:
        print("unpacked %s [%s] %s" % (st.number(mapping[r]), mapping[r],
                                       nodes[r].title))
    print("%d file(s) added" % len(nodes))


def _chain(nodes, i):
    seen = []
    while i in nodes and i not in seen:
        seen.append(i)
        i = nodes[i].parent
    return seen


# ------------------------------------------------------------------ ingest
def _docx_to_md(path):
    """Word .docx to Markdown: headings by style, lists, tables, text.
    Formulas (Office Math) are kept as their plain text between $ signs
    and counted, since they need checking."""
    import xml.etree.ElementTree as ET
    import zipfile
    W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    M = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
    with zipfile.ZipFile(path) as z:
        doc = ET.fromstring(z.read("word/document.xml"))
        try:
            styles = ET.fromstring(z.read("word/styles.xml"))
        except KeyError:
            styles = None
    level_of = {}
    if styles is not None:
        for s in styles.iter(W + "style"):
            sid = s.get(W + "styleId", "")
            name = s.find(W + "name")
            name = name.get(W + "val", "") if name is not None else ""
            ol = s.find(".//" + W + "outlineLvl")
            m = re.search(r"(?:heading|عنوان)\s*(\d)", name, re.I) or \
                re.search(r"heading\s*(\d)", sid, re.I)
            if ol is not None:
                level_of[sid] = int(ol.get(W + "val", "0")) + 1
            elif m:
                level_of[sid] = int(m.group(1))
            elif name.lower() == "title" or sid.lower() == "title":
                level_of[sid] = 1
    formulas = [0]

    def para_text(p):
        parts = []
        for el in p.iter():
            if el.tag == M + "oMath":
                formulas[0] += 1
                parts.append("$%s$" % "".join(t.text or "" for t in
                                              el.iter(M + "t")))
            elif el.tag == W + "t" and not _inside(el, p, M + "oMath"):
                parts.append(el.text or "")
            elif el.tag == W + "tab":
                parts.append("\t")
            elif el.tag in (W + "br", W + "cr"):
                parts.append("\n")
        return "".join(parts)
    parents = {c: q for q in doc.iter() for c in q}

    def _inside(el, stop, tag):
        while el is not None and el is not stop:
            if el.tag == tag:
                return True
            el = parents.get(el)
        return False
    out = []
    body = doc.find(W + "body")
    for block in (body if body is not None else []):
        if block.tag == W + "p":
            text = para_text(block).strip()
            if not text:
                continue
            ps = block.find(W + "pPr/" + W + "pStyle")
            sid = ps.get(W + "val", "") if ps is not None else ""
            if sid in level_of:
                out.append("%s %s" % ("#" * min(6, level_of[sid]),
                                      " ".join(text.split())))
            elif block.find(W + "pPr/" + W + "numPr") is not None:
                out.append("- " + text)
            else:
                out.append(text)
        elif block.tag == W + "tbl":
            for row in block.iter(W + "tr"):
                cells = [" ".join(para_text(p).strip() for p in
                                  c.iter(W + "p")) for c in row.iter(W + "tc")]
                out.append("| %s |" % " | ".join(cells))
    md = ""
    for k, line in enumerate(out):
        both = k and out[k - 1].startswith(("- ", "| ")) and \
            line[:2] == out[k - 1][:2]
        md += ("\n" if both else "\n\n" if k else "") + line
    return _tidy(md), formulas[0]


def _epub_to_md(path):
    import xml.etree.ElementTree as ET
    import zipfile
    with zipfile.ZipFile(path) as z:
        cont = ET.fromstring(z.read("META-INF/container.xml"))
        opf_path = next(e.get("full-path") for e in cont.iter()
                        if e.get("full-path"))
        opf = ET.fromstring(z.read(opf_path))
        base = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
        items = {e.get("id"): e.get("href") for e in opf.iter()
                 if e.tag.endswith("item")}
        spine = [e.get("idref") for e in opf.iter() if e.tag.endswith("itemref")]
        parts = []
        for ref in spine:
            href = items.get(ref)
            if href:
                try:
                    parts.append(from_html(z.read(base + href).decode(
                        "utf-8", "replace")))
                except KeyError:
                    pass
    return _tidy("\n\n".join(parts))


INGEST_TOOLS = {        # what each converter reads, best first per type
    ".pdf": ["docling", "marker", "pdftotext"],
    ".docx": ["builtin", "pandoc", "docling"],    # pandoc first if formulas
    ".epub": ["pandoc", "builtin"],
    ".odt": ["pandoc"], ".rtf": ["pandoc"], ".pptx": ["docling"],
}
INGEST_EXE = {"docling": "docling", "marker": "marker_single",
              "pandoc": "pandoc", "pdftotext": "pdftotext"}


def _docx_has_math(path):
    import zipfile
    try:
        with zipfile.ZipFile(path) as z:
            return b"oMath" in z.read("word/document.xml")
    except (OSError, KeyError, zipfile.BadZipFile):
        return False


def _run_tool(cmd, what):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
    except OSError as e:
        die("%s could not start: %s" % (what, e))
    if r.returncode:
        die("%s failed: %s" % (what, (r.stderr or r.stdout).strip()[-800:]))


def _external(tool, src):
    """Markdown from an installed converter."""
    import tempfile
    exe = shutil.which(INGEST_EXE[tool])
    if not exe:
        die("%s is not installed" % tool)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if tool == "pandoc":
            out = tmp / "out.md"
            # plain Markdown: no direction spans or attributes around the
            # text, formulas as $...$ LaTeX
            _run_tool([exe, str(src), "-t",
                       "markdown-smart-bracketed_spans-native_spans-raw_html"
                       "-fenced_divs-native_divs-link_attributes"
                       "-raw_attribute", "--wrap=none", "-o", str(out)],
                      "pandoc")
        elif tool == "docling":
            _run_tool([exe, str(src), "--to", "md", "--output", str(tmp)],
                      "docling")
            out = next(tmp.rglob("*.md"), None)
        elif tool == "marker":
            _run_tool([exe, str(src), "--output_dir", str(tmp)], "marker")
            out = next(tmp.rglob("*.md"), None)
        else:
            out = tmp / "out.txt"
            _run_tool([exe, "-enc", "UTF-8", str(src), str(out)], "pdftotext")
        if out is None or not out.is_file():
            die("%s wrote no Markdown" % tool)
        text = out.read_text(encoding="utf-8", errors="replace")
    return _tidy(text.replace("\f", "\n\n"))


def cmd_ingest(a):
    """Turn an office document into Markdown for deposit. With --with auto
    the best converter installed is used: for PDF Docling, then Marker,
    then pdftotext; for Word and EPUB Pandoc, then the built-in reader."""
    src = Path(a.file)
    if not src.is_file():
        die("no such file: %s" % src)
    ext = src.suffix.lower()
    note = ""
    if ext in SOURCE_FORMATS and ext not in INGEST_TOOLS:
        md, used = source_to_text(src)[0], "builtin"
    else:
        if ext not in INGEST_TOOLS:
            die("ingest reads %s (and the deposit formats)" % ", ".join(
                sorted(INGEST_TOOLS)))
        tools = INGEST_TOOLS[ext]
        if ext == ".docx" and a.tool == "auto" and _docx_has_math(src):
            tools = ["pandoc", "builtin"]   # pandoc turns formulas into LaTeX
        if a.tool != "auto":
            if a.tool not in tools:
                die("%s does not read %s files; for them: %s" % (
                    a.tool, ext, ", ".join(tools)))
            tools = [a.tool]
        used = next((t for t in tools if t == "builtin"
                     or shutil.which(INGEST_EXE[t])), None)
        if used is None:
            die("no converter for %s is installed. Install one of: %s "
                "(Docling: pip install docling; Marker: pip install "
                "marker-pdf; pdftotext: poppler-utils; Pandoc: pandoc.org). "
                "Better still, look for the book's source (LaTeX, PreTeXt, "
                "HTML) and deposit that"
                % (ext, ", ".join(t for t in tools if t != "builtin")))
        if used == "builtin" and ext == ".docx":
            md, formulas = _docx_to_md(src)
            if formulas:
                note = ("%d formula(s) were flattened to plain text between $ "
                        "signs and are not usable as they are; install Pandoc "
                        "to get them as LaTeX" % formulas)
        elif used == "builtin":
            md = _epub_to_md(src)
        else:
            md = _external(used, src)
            if used == "pdftotext":
                note = ("pdftotext gives no headings and damages formulas "
                        "(exponents, fractions, arrows); add '#' headings and "
                        "check the mathematics before filtering, or install "
                        "Docling or Marker")
            elif ext == ".pdf":
                note = ("formulas read from a PDF by a model-based converter "
                        "can be wrong while looking right; check them")
    out = Path(a.out) if a.out else src.with_suffix(".md")
    if out.resolve() == src.resolve():
        die("the output would overwrite the input; give -o")
    out.write_text(md, encoding="utf-8")
    heads = sum(1 for l in md.split("\n") if re.match(r"#{1,6} \S", l))
    print("written %s with %s (%d characters, %d headings)" % (
        out, used, len(md), heads))
    if note:
        print("note: " + note)
    if a.deposit:
        a.file = str(out)
        a.part = None
        cmd_deposit(a)


# -------------------------------------------------------------- redact, stats
def cmd_redact(a):
    """Create redact.txt, or show what it hides in a text."""
    st = Store(a.project)
    f = st.root / "redact.txt"
    if a.init:
        if f.exists():
            die("redact.txt already exists")
        f.write_text(REDACT_TEMPLATE, encoding="utf-8")
        print("created %s: from now on every model call is redacted; edit "
              "it to add your own names and words" % f)
        return
    red = Redactor.load(st.root)
    if red is None:
        die("no redact.txt in this project (create it with: redact --init)")
    text = Path(a.file).read_text(encoding="utf-8") if a.file else a.text
    if text is None:
        die("give a text, or --file")
    masked, mapping = red.mask(text)
    print(masked)
    for ph in sorted(mapping, key=lambda k: (k.rsplit("_", 1)[0],
                                             int(k.rsplit("_", 1)[1][:-1]))):
        print("  %s = %s" % (ph, mapping[ph]), file=sys.stderr)
    print("%d item(s) would be hidden" % len(mapping), file=sys.stderr)


def _prices(st):
    f = st.root / "prices.json"
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except ValueError as e:
        die("prices.json: %s" % e)


def cmd_stats(a):
    """What runs cost: per path the runs, how many finished, model calls,
    retries, fallbacks, answers from the cache, tokens and time; a cost when
    prices.json gives a price per million tokens for the model; and the
    latest evaluation."""
    st = Store(a.project)
    hdir = st.root / "history"
    since = time.time() - a.days * 86400 if a.days else 0
    rows, prices = {}, _prices(st)
    for f in sorted(hdir.glob("production-*.json")) if hdir.is_dir() else []:
        if f.stat().st_mtime < since:
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        name = rec.get("name", "?")
        if a.path and name != a.path:
            continue
        r = rows.setdefault(name, {"runs": 0, "done": 0, "calls": 0,
                                   "retries": 0, "fallbacks": 0, "cached": 0,
                                   "in": 0, "out": 0, "seconds": 0.0,
                                   "cost": 0.0, "priced": True, "known": True})
        r["runs"] += 1
        r["done"] += rec.get("status") == "done"
        r["cached"] += rec.get("from_cache", 0)
        u = rec.get("usage")
        if not u:
            r["known"] = False         # a run made before 0.8
            continue
        for k in ("calls", "retries", "fallbacks", "in", "out", "seconds"):
            r[k] += u.get(k, 0)
        for p in rec.get("steps", []):
            pu = p.get("usage")
            if not pu:
                continue
            price = prices.get(p.get("model") or rec.get("model") or "")
            if price:
                r["cost"] += (pu.get("in", 0) * price.get("input", 0)
                              + pu.get("out", 0) * price.get("output", 0)) / 1e6
            elif pu.get("in") or pu.get("out"):
                r["priced"] = False
    evals = {}
    for f in sorted(st.rdir.glob("eval-*.json")) if st.rdir.is_dir() else []:
        try:
            e = json.loads(f.read_text(encoding="utf-8"))
        except ValueError:
            continue
        evals[e.get("path")] = (e.get("passed", 0), e.get("total", 0),
                                f.stem[-15:])
    if not rows:
        print("no runs recorded%s" % (" for path " + a.path if a.path else ""))
    for name in sorted(rows):
        r = rows[name]
        cost = ("%.4f" % r["cost"]) if r["priced"] and prices else "-"
        print("%s: %d run(s), %d finished; %d model call(s), %d retried, %d "
              "fallback(s), %d answer(s) from the cache" % (
                  name, r["runs"], r["done"], r["calls"], r["retries"],
                  r["fallbacks"], r["cached"]))
        print("  tokens in %d, out %d; model time %.1f s; cost %s%s" % (
            r["in"], r["out"], r["seconds"], cost,
            "" if r["known"] else "  (runs before 0.8 kept no usage)"))
        if name in evals:
            p_, t_, when = evals[name]
            print("  latest evaluation: %d of %d passed (%s)" % (p_, t_, when))
    if not prices:
        print("costs: write prices.json with the price per million tokens, "
              'e.g. {"MODEL": {"input": 3.0, "output": 15.0}}; local models '
              "cost nothing to call")
    print("tokens are those the provider reports; manual and dry report none")


# -------------------------------------------------------------------- serve
def cmd_serve(a):
    """Offer saved paths to other programs over HTTP on this computer.
    POST /paths/NAME with {"input": "..."} and the token runs the path and
    returns its result; GET /paths lists them. One run at a time."""
    import http.server
    import secrets
    import threading
    st = Store(a.project)
    token = a.token or os.environ.get("TAWJIH_TOKEN") or secrets.token_urlsafe(18)
    allowed = set(a.paths.split(",")) if a.paths else None
    lock = threading.Lock()
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: serving on %s makes the paths reachable from other "
              "computers" % a.host, file=sys.stderr)

    def path_names():
        names = sorted(f.stem for f in st.pdir.glob("*.path"))
        return [n for n in names if allowed is None or n in allowed]

    class Handler(http.server.BaseHTTPRequestHandler):
        def send(self, code, obj):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            return secrets.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + token)

        def log_message(self, fmt, *args):
            print("serve: " + fmt % args, file=sys.stderr)

        def do_GET(self):
            if self.path == "/health":
                return self.send(200, {"tawjih": __version__})
            if not self.authorized():
                return self.send(401, {"error": "missing or wrong token"})
            if self.path == "/paths":
                fresh = Store(a.project)
                out = []
                for name in path_names():
                    try:
                        flow = fresh.load_flow(name)
                    except FlowError as e:
                        out.append({"name": name, "error": str(e)})
                        continue
                    out.append({"name": name, "steps": len(flow.nodes),
                                "code": any(fresh.nodes[i].code.strip()
                                            for i in flow.nodes)})
                return self.send(200, out)
            self.send(404, {"error": "unknown address"})

        def do_POST(self):
            if not self.authorized():
                return self.send(401, {"error": "missing or wrong token"})
            m = re.fullmatch(r"/paths/([\w\-]+)", self.path)
            if not m or m.group(1) not in path_names():
                return self.send(404, {"error": "no such path"})
            size = int(self.headers.get("Content-Length", "0") or 0)
            if size > 1_000_000:
                return self.send(413, {"error": "the request is too large"})
            try:
                req = json.loads(self.rfile.read(size) or b"{}")
                text = req["input"]
                if not isinstance(text, str):
                    raise TypeError
            except (ValueError, KeyError, TypeError):
                return self.send(400, {"error": 'send JSON: {"input": "..."}'})
            name = m.group(1)
            with lock:
                fresh = Store(a.project)
                try:
                    flow = fresh.load_flow(name)
                except FlowError as e:
                    return self.send(500, {"error": str(e)})
                if any(fresh.nodes[i].code.strip() for i in flow.nodes) \
                        and not a.allow_code:
                    return self.send(403, {"error": "this path has code; "
                                           "start serve with --allow-code"})
                err = io.StringIO()
                try:
                    with contextlib.redirect_stderr(err):
                        result = run_flow(fresh, name, text, a.backend,
                                          a.allow_code, flow,
                                          resume=a.resume,
                                          code_timeout=a.code_timeout,
                                          retries=a.retries,
                                          approve=a.approve or "deny")
                except Die as e:
                    return self.send(500, {"error": str(e.code),
                                           "messages": err.getvalue()})
                except Exception as e:          # a fault in the program
                    return self.send(500, {"error": "%s: %s" % (
                        type(e).__name__, e), "messages": err.getvalue()})
                return self.send(200, {"path": name, "result": result,
                                       "messages": err.getvalue()})

    server = http.server.ThreadingHTTPServer((a.host, a.port), Handler)
    host, port = server.server_address[:2]
    print("serving %d path(s) on http://%s:%d  (backend %s%s)" % (
        len(path_names()), host, port, a.backend,
        ", code allowed" if a.allow_code else ""))
    print("token: %s" % token)
    print('try: curl -H "Authorization: Bearer %s" http://%s:%d/paths'
          % (token, host, port))
    if getattr(a, "_ready", None):
        a._ready(server)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# --------------------------------------------------------------------- eval
def _case_passes(case, output):
    how = case.get("match", "json" if isinstance(case.get("expect"), dict)
                   else "contains")
    exp = case.get("expect")
    if how == "json":
        obj, _ = _json_in(output)
        if not isinstance(obj, dict):
            return False, "the output is not a JSON object"
        wrong = [k for k, v in exp.items() if obj.get(k) != v]
        return (not wrong), ("wrong or missing: " + ", ".join(wrong)
                             if wrong else "")
    exps = exp if isinstance(exp, list) else [exp]
    if how == "contains":
        miss = [e for e in exps if str(e).casefold() not in output.casefold()]
        return (not miss), ("missing: " + "; ".join(map(str, miss))
                            if miss else "")
    if how == "equals":
        ok = output.strip() == str(exps[0]).strip()
        return ok, "" if ok else "not equal"
    if how == "regex":
        miss = [e for e in exps if not re.search(str(e), output, re.S)]
        return (not miss), ("no match: " + "; ".join(miss) if miss else "")
    if how == "not-contains":
        hit = [e for e in exps if str(e).casefold() in output.casefold()]
        return (not hit), ("present: " + "; ".join(map(str, hit))
                           if hit else "")
    return False, "unknown match '%s' (contains, not-contains, equals, regex, json)" % how


def cmd_eval(a):
    """Run a path on test cases with known answers and report how many
    pass. Cases: paths/NAME.cases.jsonl (or --cases), one JSON object per
    line: {"input": ..., "expect": ..., "match": contains | not-contains |
    equals | regex | json}."""
    st = Store(a.project)
    flow = _flow_from_args(st, a.name, [])
    cfile = Path(a.cases) if a.cases else st.pdir / (a.name + ".cases.jsonl")
    if not cfile.is_file():
        die("no test cases: write %s, one JSON object per line, such as\n"
            '  {"input": "Is Z6 cyclic?", "expect": ["cyclic", "1", "5"]}'
            % cfile)
    cases = []
    for k, line in enumerate(cfile.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip() and not line.lstrip().startswith("#"):
            try:
                c = json.loads(line)
                if "input" not in c or "expect" not in c:
                    raise ValueError("needs input and expect")
            except ValueError as e:
                die("%s line %d: %s" % (cfile, k, e))
            c.setdefault("name", "case %d" % (len(cases) + 1))
            cases.append(c)
    if not cases:
        die("%s has no cases" % cfile)
    coded = [i for i in flow.nodes if st.nodes[i].code.strip()]
    if coded and not a.allow_code:
        die("the path has code; pass --allow-code after reading it")
    use_model(a)
    results, passed = [], 0
    for c in cases:
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                out = run_flow(st, a.name, c["input"], a.backend, a.allow_code,
                               flow, resume=a.resume,
                               model_override=getattr(a, "model", None) or "",
                               code_timeout=a.code_timeout,
                               retries=a.retries, approve=a.approve or "deny")
            ok, why = _case_passes(c, out)
        except Die as e:
            out, ok, why = "", False, "the run failed: %s" % e.code
        except Exception as e:                  # a fault in the program
            out, ok, why = "", False, "the run failed: %s: %s" % (
                type(e).__name__, e)
        log = next((l.split(": ", 1)[1] for l in err.getvalue().splitlines()
                    if l.startswith("log: ")), "")
        passed += ok
        results.append({"name": c["name"], "input": c["input"],
                        "expect": c["expect"], "match": c.get("match", ""),
                        "output": out, "pass": ok, "why": why, "log": log})
        print("%-4s %-28s %s" % ("PASS" if ok else "FAIL", c["name"][:28],
                                 why[:90]))
    rec = _stamped(st.rdir, "eval-" + a.name, ".json")
    rec.write_text(json.dumps({"tawjih": __version__, "path": a.name,
                               "backend": a.backend, "model": os.environ.get(
                                   "TAWJIH_MODEL", ""),
                               "passed": passed, "total": len(cases),
                               "cases": results}, ensure_ascii=False,
                              indent=2), encoding="utf-8")
    print("%d of %d passed (%.0f%%); record: %s" % (
        passed, len(cases), 100 * passed / len(cases), rec))
    return 0 if passed == len(cases) else 1


# ------------------------------------------------------------------ dataset
def cmd_dataset(a):
    """Write question-answer pairs as JSONL for fine-tuning tools (Unsloth,
    axolotl and the like): from passed evaluations (default), from finished
    runs, or each model step's prompt and answer (--steps). Read them
    before training: Tawjih does not judge their quality."""
    st = Store(a.project)
    rows = []
    if a.source == "eval":
        for f in sorted(st.rdir.glob("eval-*.json")) if st.rdir.is_dir() else []:
            data = json.loads(f.read_text(encoding="utf-8"))
            if a.path and data.get("path") != a.path:
                continue
            for c in data["cases"]:
                if c["pass"] or a.include_failed:
                    rows.append((c["input"], c["output"], {
                        "path": data["path"], "case": c["name"],
                        "pass": c["pass"]}))
    elif a.source == "runs":
        hdir = st.root / "history"
        for f in sorted(hdir.glob("production-*.json")) if hdir.is_dir() else []:
            rec = json.loads(f.read_text(encoding="utf-8"))
            if rec.get("status") != "done" or "input" not in rec:
                continue
            if a.path and rec.get("name") != a.path:
                continue
            try:        # the record may hold a path relative to another folder
                out = (st.rdir / Path(rec["result"]).name).read_text(
                    encoding="utf-8").strip()
            except (OSError, KeyError):
                continue
            rows.append((rec["input"], out, {"path": rec["name"],
                                             "time": rec.get("finished", "")}))
    else:
        cdir = st.root / "cache"
        for f in sorted(cdir.glob("*.json")) if cdir.is_dir() else []:
            c = json.loads(f.read_text(encoding="utf-8"))
            if "prompt" in c and (not a.step or c.get("step") == a.step):
                rows.append((c["prompt"], c["answer"], {
                    "step": c.get("step", ""), "model": c.get("model", "")}))
    seen, out = set(), []
    for q, ans, meta in rows:
        key = (q, ans)
        if key in seen or not ans.strip():
            continue
        seen.add(key)
        if a.format == "chat":
            item = {"messages": [{"role": "user", "content": q},
                                 {"role": "assistant", "content": ans}]}
        else:
            item = {"prompt": q, "completion": ans}
        if a.meta:
            item["meta"] = meta
        out.append(json.dumps(item, ensure_ascii=False))
    Path(a.out).write_text("\n".join(out) + ("\n" if out else ""),
                           encoding="utf-8")
    print("%d example(s) written to %s (from %s). Read them before training."
          % (len(out), a.out, a.source))
    if not out:
        hint = {"eval": "run eval first", "runs": "runs made before 0.7 do "
                "not keep their input", "steps": "answers cached before 0.7 "
                "do not keep their prompt"}[a.source]
        print("note: nothing found (%s)" % hint)


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

    s = sub.add_parser("new", help="create a three-sided file")
    s.add_argument("title")
    s.add_argument("--parent", help="id or number of the parent file")
    s.add_argument("--level", help="1-6 or name; default: parent level + 1")
    s.add_argument("--model", help="preferred model for this file")
    s.add_argument("--open", action="store_true",
                   help="open the editors on the new file")
    s.set_defaults(fn=cmd_new)

    s = sub.add_parser("open", help="open an editor for each side of a file")
    s.add_argument("ref")
    s.add_argument("--editor", help="editor command (default: TAWJIH_EDITOR, "
                   "VISUAL, EDITOR, then notepad on Windows or nano/vim)")
    s.add_argument("--reading", action="store_true",
                   help="open the reading side even when it is empty")
    s.add_argument("--discard", action="store_true",
                   help="throw away the text left by an unfinished edit")
    s.add_argument("--split", action="store_true",
                   help="open the machine side as two files, instructions.md "
                   "and code.py (always done for VS Code)")
    s.set_defaults(fn=cmd_open)

    s = sub.add_parser("tree", help="show trees or a branch (H/I/C/R = sides filled, ! = stale)")
    s.add_argument("--branch", help="id or number of a branch to display")
    s.set_defaults(fn=cmd_tree)

    s = sub.add_parser("search", help="search across trees or within a branch")
    s.add_argument("query", help="text to search for")
    s.add_argument("--branch", help="limit search to this branch id or number")
    s.add_argument("--side", choices=["H", "I", "C", "R", "h", "i", "c", "r",
                                      "human", "instructions", "code", "reading"],
                   help="limit search to one side")
    s.add_argument("--tag", help="only files with this tag")
    s.add_argument("--rank", action="store_true",
                   help="rank by relevance (BM25), any of the words")
    s.add_argument("--semantic", action="store_true",
                   help="rank by meaning, with the index built by: index")
    s.add_argument("--top", type=int, default=10,
                   help="results shown with --rank or --semantic")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("index", help="build the semantic search index "
                       "(embeddings of the human and reading sides)")
    s.add_argument("--backend", choices=sorted(EMBEDDERS), default="ollama")
    s.add_argument("--model", help="embedding model (default: "
                   "TAWJIH_EMBED_MODEL, else nomic-embed-text)")
    s.set_defaults(fn=cmd_index)

    # restructuring
    def where(s):
        s.add_argument("--to", metavar="REF", help="the new parent")
        s.add_argument("--root", action="store_true",
                       help="make it the root of a new tree")
    s = sub.add_parser("move", aliases=["mv"], help="move a file and its "
                       "branch to a new parent (ids, and so paths, stay)")
    s.add_argument("ref")
    where(s)
    s.add_argument("--position", type=int, help="place among the new siblings")
    s.add_argument("--before", metavar="REF")
    s.add_argument("--after", metavar="REF")
    s.add_argument("--level", help="with --root: the level of the new root")
    s.set_defaults(fn=cmd_move)

    s = sub.add_parser("rename", help="change a file's title; AMONG "
                       "deciders that name it are updated")
    s.add_argument("ref")
    s.add_argument("title")
    s.add_argument("--no-update", action="store_true",
                   help="leave the instructions of AMONG deciders as they are")
    s.set_defaults(fn=cmd_rename)

    s = sub.add_parser("copy", aliases=["cp"], help="copy a file (or a "
                       "branch) with its sides and contracts")
    s.add_argument("ref")
    where(s)
    s.add_argument("--branch", action="store_true",
                   help="copy everything under it too")
    s.add_argument("--title", help="title of the copy")
    s.set_defaults(fn=cmd_copy)

    s = sub.add_parser("rm", aliases=["remove"], help="remove a file that no "
                       "path uses (a copy stays in history/)")
    s.add_argument("ref")
    s.add_argument("--branch", action="store_true",
                   help="remove everything under it too")
    s.add_argument("--unlink", action="store_true",
                   help="also remove links to it from other files")
    s.add_argument("--yes", action="store_true", help="really remove")
    s.set_defaults(fn=cmd_rm)

    s = sub.add_parser("split", help="move reading passages into a new file "
                       "placed after it (without --passages: list them)")
    s.add_argument("ref")
    s.add_argument("--passages", help="numbers and ranges, such as 3-5,7")
    s.add_argument("--title", help="title of the new file")
    s.set_defaults(fn=cmd_split)

    s = sub.add_parser("merge", help="merge B's human and reading sides "
                       "into A, then remove B")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--keep", action="store_true", help="keep B")
    s.set_defaults(fn=cmd_merge)

    # versions
    s = sub.add_parser("log", help="saved versions of a file and what "
                       "changed")
    s.add_argument("ref")
    s.set_defaults(fn=cmd_log)

    s = sub.add_parser("diff", help="a saved version against the current file")
    s.add_argument("ref")
    s.add_argument("--version", type=int, default=1,
                   help="1 is the newest saved version (see: log)")
    s.add_argument("--side", action="append", help="human, instructions, "
                   "code, machine or reading (repeatable)")
    s.set_defaults(fn=cmd_diff)

    s = sub.add_parser("revert", help="bring back sides of a saved version, "
                       "or a removed file")
    s.add_argument("ref")
    s.add_argument("--version", type=int, default=1,
                   help="1 is the newest saved version (see: log)")
    s.add_argument("--side", action="append", help="human, instructions, "
                   "code, machine or reading (default: all four)")
    s.add_argument("--header", action="store_true", help="also the title, "
                   "links, model, contracts and tags")
    s.set_defaults(fn=cmd_revert)

    # state
    s = sub.add_parser("status", help="only what is unfinished")
    s.add_argument("--branch", help="limit to this branch")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("tag", help="add tags to a file; without arguments "
                       "list all tags")
    s.add_argument("ref", nargs="?")
    s.add_argument("tags", nargs="*")
    s.set_defaults(fn=cmd_tag)

    s = sub.add_parser("untag", help="remove tags from a file")
    s.add_argument("ref")
    s.add_argument("tags", nargs="+")
    s.set_defaults(fn=cmd_untag)

    s = sub.add_parser("sync", help="find files edited outside Tawjih and "
                       "unfinished edits in work/")
    s.add_argument("--confirm-machine", action="store_true",
                   help="machine sides edited by hand count as rebuilt")
    s.add_argument("--clean-work", action="store_true",
                   help="remove unfinished edits left in work/")
    s.add_argument("--watch", type=float, metavar="SECONDS",
                   help="check again every SECONDS until Ctrl+C")
    s.set_defaults(fn=cmd_sync)

    s = sub.add_parser("refs", help="everything that refers to a file")
    s.add_argument("ref")
    s.set_defaults(fn=cmd_refs)

    s = sub.add_parser("clean", help="find (and with --yes remove) leftovers "
                       "and broken links")
    s.add_argument("--runs", type=int, metavar="DAYS",
                   help="also run records older than DAYS")
    s.add_argument("--yes", action="store_true", help="really clean")
    s.set_defaults(fn=cmd_clean)

    s = sub.add_parser("audit", help="look through code sides for what "
                       "touches the system (not a sandbox)")
    s.add_argument("ref", nargs="?", help="a file or branch (default: all)")
    s.add_argument("--path", help="the code steps of a path")
    s.set_defaults(fn=cmd_audit)

    # outputs
    s = sub.add_parser("export", help="the trees as one readable book "
                       "(Markdown or HTML)")
    s.add_argument("--branch", help="only this branch")
    s.add_argument("--format", choices=["md", "html"],
                   help="default: from the -o extension, else md")
    s.add_argument("--machine", action="store_true",
                   help="include the machine sides")
    s.add_argument("--no-reading", action="store_true",
                   help="leave out the reading passages")
    s.add_argument("--no-mathjax", action="store_true",
                   help="HTML without MathJax (formulas stay as LaTeX text; "
                   "otherwise MathJax is loaded from the web)")
    s.add_argument("-o", "--out", help="output file (default: print)")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("pack", help="a branch or the files of paths in one "
                       "zip file for another project")
    s.add_argument("--branch", action="append", metavar="REF")
    s.add_argument("--path", action="append", metavar="NAME")
    s.add_argument("-o", "--out", required=True)
    s.set_defaults(fn=cmd_pack)

    s = sub.add_parser("unpack", help="add a pack to this project")
    s.add_argument("file")
    s.add_argument("--parent", help="put its trees under this file")
    s.add_argument("--new-ids", action="store_true",
                   help="give the files new ids (needed when they clash)")
    s.set_defaults(fn=cmd_unpack)

    s = sub.add_parser("ingest", help="turn .docx, .epub or .pdf into "
                       "Markdown for deposit")
    s.add_argument("file")
    s.add_argument("-o", "--out", help="output file (default: FILE.md)")
    s.add_argument("--deposit", action="store_true",
                   help="deposit the result too")
    s.add_argument("--with", dest="tool", default="auto",
                   choices=["auto", "docling", "marker", "pandoc",
                            "pdftotext", "builtin"],
                   help="converter (default auto: the best one installed)")
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("serve", help="offer saved paths to other programs "
                       "over HTTP on this computer")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--backend", choices=sorted(BACKENDS), default="dry")
    s.add_argument("--paths", help="only these paths, comma separated")
    s.add_argument("--token", help="access token (default: TAWJIH_TOKEN, "
                   "else a new one is printed)")
    s.add_argument("--allow-code", action="store_true",
                   help="allow paths with code steps")
    s.add_argument("--resume", action="store_true",
                   help="take kept answers from the cache")
    s.add_argument("--retries", type=int, choices=range(10), metavar="N",
                   help="ask a model again up to N times when its answer "
                   "breaks the contract (default: each file's retries line)")
    s.add_argument("--approve", choices=["ask", "auto", "deny"],
                   help="steps marked approve: ask on the terminal "
                   "(default on a terminal), auto, or deny (default "
                   "otherwise)")
    s.add_argument("--code-timeout", type=float, default=CODE_TIMEOUT,
                   metavar="SECONDS")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("eval", help="run a path on test cases with known "
                       "answers")
    s.add_argument("name")
    s.add_argument("--cases", help="JSONL file (default: "
                   "paths/NAME.cases.jsonl)")
    s.add_argument("--backend", choices=sorted(BACKENDS), default="dry")
    s.add_argument("--model", help="model for every step")
    s.add_argument("--allow-code", action="store_true")
    s.add_argument("--resume", action="store_true",
                   help="take kept answers from the cache")
    s.add_argument("--retries", type=int, choices=range(10), metavar="N",
                   help="ask a model again up to N times when its answer "
                   "breaks the contract (default: each file's retries line)")
    s.add_argument("--approve", choices=["ask", "auto", "deny"],
                   help="steps marked approve: ask on the terminal "
                   "(default on a terminal), auto, or deny (default "
                   "otherwise)")
    s.add_argument("--code-timeout", type=float, default=CODE_TIMEOUT,
                   metavar="SECONDS")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("dataset", help="question-answer pairs as JSONL for "
                       "fine-tuning tools")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--source", choices=["eval", "runs", "steps"],
                   default="eval", help="passed evaluations (default), "
                   "finished runs, or each model step")
    s.add_argument("--path", help="only this path")
    s.add_argument("--step", help="with --source steps: only this file id")
    s.add_argument("--include-failed", action="store_true",
                   help="with --source eval: failed cases too")
    s.add_argument("--format", choices=["chat", "prompt"], default="chat")
    s.add_argument("--meta", action="store_true",
                   help="add where each example came from")
    s.set_defaults(fn=cmd_dataset)

    s = sub.add_parser("show", help="print one file and its location")
    s.add_argument("ref")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("link", help="link a file to any file, any tree")
    s.add_argument("src")
    s.add_argument("dst")
    s.add_argument("--type", default="related")
    s.set_defaults(fn=cmd_link)

    s = sub.add_parser("contract", help="show or set a file's contracts and "
                       "model (expects, produces: json, json(a, b), text; "
                       "'none' clears)")
    s.add_argument("ref")
    s.add_argument("--expects", help="contract on the step's input")
    s.add_argument("--produces", help="contract on the step's output")
    s.add_argument("--model", help="the model this file's step uses")
    s.add_argument("--retries", help="0-9: ask the model again when its "
                   "answer breaks the produces contract")
    s.add_argument("--fallback", metavar="REF", help="a file run instead "
                   "when this step fails (same input)")
    s.add_argument("--approve", help="before, after or both: a person "
                   "approves the step")
    s.set_defaults(fn=cmd_contract)

    s = sub.add_parser("redact", help="show what redact.txt hides before "
                       "text is sent to a model; --init creates it")
    s.add_argument("text", nargs="?", help="text to try (or --file)")
    s.add_argument("--file", help="a file to try")
    s.add_argument("--init", action="store_true",
                   help="create redact.txt with the built-in patterns")
    s.set_defaults(fn=cmd_redact)

    s = sub.add_parser("stats", help="runs, tokens, time, cost and "
                       "evaluation results")
    s.add_argument("--path", help="only this path")
    s.add_argument("--days", type=int, help="only the last DAYS days")
    s.set_defaults(fn=cmd_stats)

    s = sub.add_parser("confirm", help="record that a machine side still fits "
                       "its human and reading sides (clears the stale mark)")
    s.add_argument("refs", nargs="+", metavar="REF")
    s.set_defaults(fn=cmd_confirm)

    s = sub.add_parser("check", help="validate trees, contracts, links and paths")
    s.add_argument("--branch", help="limit check to a specific branch")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("tangle", help="export code sides as Python functions, one per step")
    s.add_argument("--branch", help="id or number of a branch to export")
    s.add_argument("--path", help="export code steps of a saved path in order")
    s.add_argument("-o", "--out", help="output .py file path")
    s.set_defaults(fn=cmd_tangle)

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
    s.add_argument("--retries", type=int, choices=range(10), metavar="N",
                   help="ask a model again up to N times when its answer "
                   "breaks the contract (default: each file's retries line)")
    s.add_argument("--approve", choices=["ask", "auto", "deny"],
                   help="steps marked approve: ask on the terminal "
                   "(default on a terminal), auto, or deny (default "
                   "otherwise)")
    s.add_argument("--code-timeout", type=float, default=CODE_TIMEOUT,
                   metavar="SECONDS", help="stop a code step that runs longer "
                   "(default %g; 0 for no limit)" % CODE_TIMEOUT)
    s.add_argument("--allow-code", action="store_true",
                   help="execute Python in the code sections (not sandboxed)")
    s.add_argument("--model", help="model name (default: the TAWJIH_MODEL "
                   "environment variable)")
    s.add_argument("--resume", action="store_true",
                   help="reuse kept answers for steps whose prompt, provider "
                   "and model are unchanged")
    s.add_argument("--redo", action="append", metavar="REF",
                   help="with --resume: ask this step again (repeatable)")
    s.add_argument("--no-cache", action="store_true",
                   help="do not keep the model's answers in cache/")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("cache", help="show or empty the kept model answers")
    s.add_argument("--clear", action="store_true", help="remove them all")
    s.set_defaults(fn=cmd_cache)

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
    s.add_argument("--retries", type=int, choices=range(10), metavar="N",
                   help="ask a model again up to N times when its answer "
                   "breaks the contract (default: each file's retries line)")
    s.add_argument("--approve", choices=["ask", "auto", "deny"],
                   help="steps marked approve: ask on the terminal "
                   "(default on a terminal), auto, or deny (default "
                   "otherwise)")
    s.add_argument("--code-timeout", type=float, default=CODE_TIMEOUT,
                   metavar="SECONDS", help="stop a code step that runs longer "
                   "(default %g; 0 for no limit)" % CODE_TIMEOUT)
    s.add_argument("--allow-code", action="store_true",
                   help="execute Python in the code sections (not sandboxed)")
    s.add_argument("--model", help="model name (default: the TAWJIH_MODEL "
                   "environment variable)")
    s.add_argument("--resume", action="store_true",
                   help="reuse kept answers for steps whose prompt, provider "
                   "and model are unchanged")
    s.add_argument("--redo", action="append", metavar="REF",
                   help="with --resume: ask this step again (repeatable)")
    s.add_argument("--no-cache", action="store_true",
                   help="do not keep the model's answers in cache/")
    s.set_defaults(fn=cmd_produce)

    s = sub.add_parser("graph", help="draw the trees or a path (Mermaid, "
                       "Markdown or Graphviz DOT)")
    s.add_argument("--path", help="draw this saved path instead of the trees")
    s.add_argument("--branch", help="draw only this branch of the trees")
    s.add_argument("--format", choices=GRAPH_FORMATS,
                   help="md (Mermaid in Markdown, shown as a picture on "
                   "GitHub), mermaid, or dot; default: from the -o "
                   "extension (.mmd, .dot or .gv), else md")
    s.add_argument("-o", "--out", help="write to this file")
    s.set_defaults(fn=cmd_graph)

    s = sub.add_parser("generate", help="propose a path from contracts or "
                       "from a goal; accept it to save it")
    s.add_argument("name")
    s.add_argument("--start", help="first file (with --end): link by contracts")
    s.add_argument("--end", help="last file (with --start)")
    s.add_argument("--goal", help="what the path should do: a model proposes it")
    s.add_argument("--scope", help="only files under this branch")
    s.add_argument("--backend", choices=sorted(BACKENDS), default="dry")
    s.add_argument("--model", help="model name (default: TAWJIH_MODEL)")
    s.add_argument("--accept", action="store_true",
                   help="turn the reviewed proposal into a path")
    s.add_argument("--replace", action="store_true",
                   help="with --accept: overwrite a path of the same name")
    s.set_defaults(fn=cmd_generate)

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
    s.add_argument("--model", help="model name (default: the TAWJIH_MODEL "
                   "environment variable)")
    s.add_argument("--yes", action="store_true", help="save without asking")
    s.set_defaults(fn=cmd_edit)

    s = sub.add_parser("deposit", help="add a book to the library: text, "
                       "Markdown, LaTeX, HTML or XML (PreTeXt)")
    s.add_argument("file")
    s.add_argument("--part", action="append", metavar="N|TITLE",
                   help="keep only this top-level division (chapter), by "
                   "number or title; repeat for more. Depositing the whole "
                   "book this way keeps its own numbering")
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
    s.add_argument("--model", help="model name (default: the TAWJIH_MODEL "
                   "environment variable)")
    s.set_defaults(fn=cmd_filter)

    s = sub.add_parser("accept", help="add the kept passages to reading sides")
    s.add_argument("book")
    s.set_defaults(fn=cmd_accept)

    s = sub.add_parser("read", help="read the reading side of a file")
    s.add_argument("ref")
    s.add_argument("--all", action="store_true", help="include the subtree")
    s.set_defaults(fn=cmd_read)

    a = p.parse_args()
    rc = a.fn(a)
    if isinstance(rc, int) and rc:
        sys.exit(rc)


if __name__ == "__main__":
    main()
