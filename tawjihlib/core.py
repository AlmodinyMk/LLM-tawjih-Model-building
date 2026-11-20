"""Constants, the three-sided file model, examples and the Store.

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
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
import urllib.request
import uuid
from collections.abc import Mapping
from pathlib import Path



__version__ = "0.10.0"

DESCRIPTION = "tawjih - personal LLM command-directing (توجيه أوامر الآلة)."
LEVELS = ["general", "section", "chapter", "topic", "branch", "part"]
LEVELS_AR = {"عام": 1, "قسم": 2, "فصل": 3, "موضوع": 4, "فرع": 5, "جزء": 6}
SECTIONS = ["human", "instructions", "code", "reading"]
# a fifth, optional section: worked questions and answers, the material for
# evaluation cases and training data (never sent when a path runs)
EXTRA = ["examples"]
SPLITS = ("train", "eval", "ignore")
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
                 retries="", fallback="", approve="", examples="",
                 split="", weight="", examples_sha="", locked="",
                 aliases=None, each="", reviewed_sha="", reviewed_on=""):
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
        self.examples = examples
        self.split = split.strip()            # train | eval | ignore
        self.weight = str(weight).strip()     # importance in training data
        self.examples_sha = examples_sha.strip()
        self.locked = locked.strip()          # "yes": content may not change
        self.aliases = list(aliases or [])    # other names usable as REF
        self.each = each.strip()              # run once per item of a list
        self.reviewed_sha = reviewed_sha.strip()
        self.reviewed_on = reviewed_on.strip()
        self.mount = ""                       # set for a mounted project's file

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

    def content_key(self):
        """Everything a lock protects: the sides and what they mean. Not
        the place in the tree, review marks or build digests."""
        return json.dumps([getattr(self, k) for k in (
            "title", "human", "instructions", "code", "reading", "examples",
            "expects", "produces", "model", "retries", "fallback", "approve",
            "split", "weight", "each", "tags", "links")], ensure_ascii=False)

    def content_digest(self):
        return hashlib.sha256(self.content_key().encode("utf-8")).hexdigest()[:16]

    def examples_stale(self):
        """True if the human or reading side changed after the examples
        were written through the program."""
        return bool(self.examples.strip() and self.examples_sha
                    and self.source_digest() != self.examples_sha)

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
        if self.aliases:
            out.append("aliases: " + ", ".join(self.aliases))
        for k in ("retries", "fallback", "approve", "split", "weight",
                  "examples_sha", "each", "locked", "reviewed_sha",
                  "reviewed_on"):
            if getattr(self, k):
                out.append("%s: %s" % (k, getattr(self, k)))
        out.append("---")
        for s in SECTIONS:
            out += [MARK.format(s), getattr(self, s).strip("\n")]
        if self.examples.strip():
            out += [MARK.format("examples"), self.examples.strip("\n")]
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
        sec, cur = {s: [] for s in SECTIONS + EXTRA}, None
        for line in lines[end + 1:]:
            if line.strip() == MARK.format("anti_examples"):
                line = MARK.format("examples")    # one section holds both
            hit = [s for s in SECTIONS + EXTRA
                   if line.strip() == MARK.format(s)]
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
                    split=meta.get("split", ""),
                    weight=meta.get("weight", ""),
                    examples_sha=meta.get("examples_sha", ""),
                    locked=meta.get("locked", ""),
                    aliases=[t.strip() for t in meta.get("aliases", "")
                             .split(",") if t.strip()],
                    each=meta.get("each", ""),
                    reviewed_sha=meta.get("reviewed_sha", ""),
                    reviewed_on=meta.get("reviewed_on", ""),
                    examples="\n".join(sec["examples"]).strip("\n"),
                    **{s: "\n".join(sec[s]).strip("\n") for s in SECTIONS})


# ----------------------------------------------------------------- examples
EXAMPLE_KEYS = {"q": "q", "س": "q", "question": "q",
                "a": "a", "ج": "a", "answer": "a",
                "e": "e", "expect": "e", "متوقع": "e",
                "bad": "bad", "خطأ": "bad"}


def parse_examples(text):
    """The examples section: blocks that start with 'Q:' (or 'س:'), then
    'A:' ('ج:') the good answer, optional 'E:' a short text the answer must
    contain (for evaluation), and any number of 'BAD:' ('خطأ:') wrong
    answers. A line without a key continues the line before it.
    Returns (examples, problems)."""
    out, problems, cur, field = [], [], None, None
    for k, line in enumerate(text.split("\n"), 1):
        m = re.match(r"\s*([A-Za-z]+|[؀-ۿ]+)\s*[:：]\s?(.*)", line)
        key = EXAMPLE_KEYS.get(m.group(1).casefold()) if m else None
        if key == "q":
            cur = {"q": m.group(2).strip(), "a": "", "e": "", "bad": [],
                   "line": k}
            out.append(cur)
            field = "q"
        elif key:
            if cur is None:
                problems.append("line %d: '%s' before any question" % (
                    k, m.group(1)))
                continue
            if key == "bad":
                cur["bad"].append(m.group(2).strip())
            else:
                cur[key] = m.group(2).strip()
            field = key
        elif line.strip() and cur is not None and field:
            if field == "bad":
                cur["bad"][-1] += "\n" + line.rstrip()
            else:
                cur[field] += "\n" + line.rstrip()
        elif line.strip() and not line.lstrip().startswith("#"):
            problems.append("line %d: text before any question" % k)
    for ex in out:
        if not ex["q"]:
            problems.append("line %d: empty question" % ex["line"])
        if not ex["a"] and not ex["e"]:
            problems.append("line %d: question without an answer (A:) or an "
                            "expected text (E:)" % ex["line"])
    return out, problems


def _norm_q(q):
    return " ".join(re.findall(r"\w+", q.casefold()))


def examples_lint(st, ids=None):
    """(errors, warnings) about the examples of the given files: unreadable
    blocks, the same question in train and eval files (leakage), the same
    question with different answers (a contradiction), duplicates."""
    errors, warns, seen = [], [], {}
    for i in ids if ids is not None else st.nodes:
        n = st.nodes[i]
        if n.split and n.split not in SPLITS:
            errors.append("%s (%s): split must be train, eval or ignore"
                          % (i, n.title))
        if n.weight:
            try:
                if float(n.weight) < 0:
                    raise ValueError
            except ValueError:
                errors.append("%s (%s): weight must be a number of 0 or more"
                              % (i, n.title))
        if not n.examples.strip():
            continue
        exs, probs = parse_examples(n.examples)
        errors += ["%s (%s): examples %s" % (i, n.title, p) for p in probs]
        if n.examples_stale():
            warns.append("%s (%s): examples are older than the human or "
                         "reading side" % (i, n.title))
        for ex in exs:
            if n.split == "ignore":
                continue
            seen.setdefault(_norm_q(ex["q"]), []).append(
                (i, n.split or "train", ex["a"].strip()))
    for q, where in seen.items():
        if len(where) < 2:
            continue
        splits = {w[1] for w in where}
        ids_ = ", ".join(sorted({w[0] for w in where}))
        if {"train", "eval"} <= splits:
            errors.append("the same question is in train and eval files (%s): "
                          "evaluation would measure memory" % ids_)
        if len({w[2] for w in where if w[2]}) > 1:
            warns.append("the same question has different answers in %s"
                         % ids_)
        else:
            warns.append("duplicate question in %s" % ids_)
    return errors, warns


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
        # mounted projects: their files take part, read-only
        self.mounts = {}
        mf = self.root / "mounts.json"
        if mf.is_file():
            try:
                self.mounts = json.loads(mf.read_text(encoding="utf-8"))
            except ValueError:
                die("mounts.json is not readable JSON")
        self._mount_dirs = {}
        for name, where in self.mounts.items():
            mdir = (self.root / where).resolve() / "files"
            if not mdir.is_dir():
                print("warning: mounted project %s is not found at %s"
                      % (name, where), file=sys.stderr)
                continue
            for f in sorted(mdir.glob("*.md")):
                try:
                    n = Node.parse(f.read_text(encoding="utf-8"))
                except Exception as e:
                    die("mounted %s, %s: %s" % (name, f.name, e))
                if n.id in self.nodes:
                    die("mounted project %s has file %s, which is also here; "
                        "unmount it" % (name, n.id))
                n.mount = name
                self.nodes[n.id] = n
                self._mount_dirs[n.id] = mdir

    def file_of(self, nid):
        if nid in getattr(self, "_mount_dirs", {}):
            return self._mount_dirs[nid] / (nid + ".md")
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
                           for i, m in self.nodes.items() if not m.mount}
        self.state[n.id] = self.side_digests(n)
        self._write_state()

    def record_all(self):
        self._state = {i: self.side_digests(n) for i, n in self.nodes.items()
                       if not n.mount}
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

    def guard(self, n):
        """Refuse to change a mounted file, or the content of a locked one."""
        if n.mount:
            die("%s [%s] belongs to the mounted project %s and is read-only"
                % (n.title, n.id, n.mount))
        f = self.file_of(n.id)
        if f.is_file():
            try:
                old = Node.parse(f.read_text(encoding="utf-8"))
            except ValueError:
                return
            if old.locked and n.locked and \
                    old.content_key() != n.content_key():
                die("%s [%s] is locked; unlock it first (unlock %s)"
                    % (old.title, n.id, n.id))

    def save(self, n):
        self.guard(n)
        self.file_of(n.id).write_text(n.dump(), encoding="utf-8")
        self.nodes[n.id] = n
        self.record(n)

    def children(self, pid):
        # mounted trees come after this project's own, so mounting another
        # project never changes the numbers of local files
        return sorted((n for n in self.nodes.values() if n.parent == pid),
                      key=lambda n: (bool(n.mount), n.mount or "", n.order,
                                     n.id))

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
        low = ref.casefold()
        hits = [n.id for n in self.nodes.values()
                if any(x.casefold() == low for x in n.aliases)]
        if len(hits) == 1:
            return hits[0]
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
        while nid in self.nodes or (self.root / "archive" / (
                nid + ".md")).exists():
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
            if n.locked and n.locked != "yes":
                errors.append(tag + ": locked must be yes")
            if n.each and not re.fullmatch(r"yes|[\w\-]+", n.each):
                errors.append(tag + ": each must be yes or a JSON key")
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
        e2, w2 = examples_lint(self, sorted(scope))
        return errors + e2, warns + w2


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


__all__ = [
    'CONTRACT_HELP', 'DESCRIPTION', 'Die', 'EXAMPLE_KEYS', 'EXTRA',
    'FALSE_WORDS', 'Flow', 'FlowError', 'KEYWORDS', 'LEVELS', 'LEVELS_AR',
    'MARK', 'Mapping', 'Node', 'Path', 'SECTIONS', 'SPLITS', 'Store',
    '__version__', '_json_in', '_kw', '_norm_q', '_parse_contract',
    '_validate_contract', 'argparse', 'build_flow', 'condition_holds',
    'conform', 'contextlib', 'contract_fits', 'die', 'difflib',
    'examples_lint', 'flow_condition_notes', 'flow_contract_errors',
    'hashlib', 'io', 'json', 'math', 'os', 'parse_examples',
    'parse_level', 're', 'shlex', 'shutil', 'subprocess', 'sys',
    'textwrap', 'threading', 'time', 'unicodedata', 'urllib', 'uuid',
]
