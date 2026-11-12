#!/usr/bin/env python3
"""tawjih - personal LLM command-directing (توجيه أوامر الآلة).

A project is a folder of three-sided files arranged as trees:
    human side   : the expressive human concept (context)
    machine side : instructions for the model + optional Python code
    reading side : filtered passages of deposited books and research
A path is an ordered list of files; running it executes their machine
sides as steps, each step's output feeding the next (a linear workflow).
The human and reading sides are used only when building a file ('edit').

Standard library only. Python 3.9+.
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

__version__ = "0.1.0"

LEVELS = ["general", "section", "chapter", "topic", "branch", "part"]
LEVELS_AR = {"عام": 1, "قسم": 2, "فصل": 3, "موضوع": 4, "فرع": 5, "جزء": 6}
SECTIONS = ["human", "instructions", "code", "reading"]
MARK = "=== {} ==="


def die(msg):
    print("error: " + msg, file=sys.stderr)
    sys.exit(1)


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
        n = Node(id=nid, title=title, level=level, parent=parent, order=order)
        self.save(n)
        return n

    def load_path(self, name):
        if not re.fullmatch(r"[\w\-]+", name):
            die("path names may contain only letters, digits, '_' and '-'")
        f = self.pdir / (name + ".path")
        if not f.is_file():
            die("no path named '%s'" % name)
        ids = [l.split("#", 1)[0].strip()
               for l in f.read_text(encoding="utf-8").splitlines()]
        return [i for i in ids if i]

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
                for nid in self.load_path(f.stem):
                    n = self.nodes.get(nid)
                    if n is None:
                        errors.append("path %s: file %s does not exist"
                                      % (f.stem, nid))
                    elif not n.instructions.strip() and not n.code.strip():
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


def run_path(store, name, text, backend, allow_code):
    steps = store.load_path(name)
    for nid in steps:
        if nid not in store.nodes:
            die("path %s: file %s does not exist" % (name, nid))
    prev, shared, log = text, {}, ["# Run of path '%s'" % name]
    store.rdir.mkdir(exist_ok=True)
    stamp, k = time.strftime("%Y%m%d-%H%M%S"), 1
    logfile = store.rdir / ("%s-%s.md" % (name, stamp))
    while logfile.exists():
        k += 1
        logfile = store.rdir / ("%s-%s-%d.md" % (name, stamp, k))

    def fail(i, n, what, e):
        log.append("### FAILED (%s)\n%s: %s" % (what, type(e).__name__, e))
        logfile.write_text("\n".join(log) + "\n", encoding="utf-8")
        die("step %d [%s] %s failed in %s: %s: %s (log: %s)"
            % (i, n.id, n.title, what, type(e).__name__, e, logfile))

    for i, nid in enumerate(steps, 1):
        n = store.nodes[nid]
        print("step %d/%d  %s [%s] %s" % (i, len(steps), store.number(nid),
                                          nid, n.title), file=sys.stderr)
        code_out = ""
        log.append("\n## Step %d: %s [%s]" % (i, n.title, nid))
        if n.code.strip():
            if allow_code:
                try:
                    code_out = run_code(n, prev, shared, text)
                except Exception as e:
                    fail(i, n, "code", e)
            else:
                print("  code skipped (pass --allow-code to execute it)",
                      file=sys.stderr)
        if n.instructions.strip():
            prompt = build_prompt(store, n, prev, code_out, text)
            print("  prompt: %d characters" % len(prompt), file=sys.stderr)
            log.append("### Prompt\n" + prompt)
            try:
                prev = BACKENDS[backend](prompt)
            except Exception as e:
                fail(i, n, "model call", e)
        elif code_out:
            prev = code_out
        log.append("### Output\n" + prev)
    logfile.write_text("\n".join(log) + "\n", encoding="utf-8")
    print("log: %s" % logfile, file=sys.stderr)
    return prev


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


def cmd_path(a):
    st = Store(a.project)
    st.pdir.mkdir(exist_ok=True)
    if not re.fullmatch(r"[\w\-]+", a.name):
        die("path names may contain only letters, digits, '_' and '-'")
    if a.refs:
        ids = [st.resolve(r) for r in a.refs]
        lines = ["%s  # %s" % (i, st.nodes[i].title) for i in ids]
        f = st.pdir / (a.name + ".path")
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("path saved: %s" % f)
    for i, nid in enumerate(st.load_path(a.name), 1):
        n = st.nodes.get(nid)
        print("%d. %s [%s] %s" % (i, st.number(nid) if n else "?", nid,
                                  n.title if n else "(missing)"))


def cmd_run(a):
    st = Store(a.project)
    text = a.input
    if a.input_file:
        text = Path(a.input_file).read_text(encoding="utf-8")
    print(run_path(st, a.name, text, a.backend, a.allow_code))


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
    s.add_argument("refs", nargs="*", help="ordered ids or numbers")
    s.set_defaults(fn=cmd_path)

    s = sub.add_parser("run", help="execute a path step by step")
    s.add_argument("name")
    s.add_argument("--input", default="")
    s.add_argument("--input-file")
    s.add_argument("--backend", choices=sorted(BACKENDS), default="dry")
    s.add_argument("--allow-code", action="store_true",
                   help="execute Python in the code sections (not sandboxed)")
    s.set_defaults(fn=cmd_run)

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
