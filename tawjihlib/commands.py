"""The basic commands, the network, path generation and the side editors.

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import contextlib
import difflib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from .core import (
    CONTRACT_HELP, Die, EXTRA, FlowError, LEVELS, MARK, SECTIONS, SPLITS,
    Store, _kw, _parse_contract, build_flow, contract_fits, die,
    flow_condition_notes, flow_contract_errors, load_schema, parse_examples,
    parse_level, schema_file, schema_problems,
)
from .providers import (
    backend_kind, call_model, use_model,
)
from .engine import (
    flow_text, plan_steps, print_plan, run_flow, split_statements,
    write_flow,
)


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


def display_width(text):
    """Columns a terminal gives the text: marks take none, wide
    characters two."""
    import unicodedata
    w = 0
    for ch in text:
        if unicodedata.combining(ch) or unicodedata.category(ch) in (
                "Mn", "Me", "Cf"):
            continue
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


def _pad(text, width, right=False):
    gap = " " * max(0, width - display_width(text))
    return gap + text if right else text + gap


def _fancy():
    """Box and triangle characters, except in the old Windows console,
    whose fonts lack some of them."""
    return not (os.name == "nt" and sys.stdout.isatty() and not (
        os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM")))


def tree_lines(rows, fancy=True):
    """rows: [(id, parent id)] in tree order. Returns {id: the number's
    prefix}: ├─ └─ │ joining each file to its parent, ▾ before a file with
    files under it and • before one without."""
    kids, parent = {}, {}
    for i, p in rows:
        kids.setdefault(p or "", []).append(i)
        parent[i] = p or ""
    last = {i: k[-1] == i for k in kids.values() for i in k}
    out = {}
    for i, _ in rows:
        chain, p = [], parent[i]
        while p in parent:
            chain.insert(0, p)
            p = parent[p]
        pre = "".join("   " if last[x] else "│  " for x in chain[1:])
        if chain:
            pre += "└─ " if last[i] else "├─ "
        out[i] = pre + (("▾ " if fancy else "▼ ") if i in kids else "• ")
    return out


def _colors():
    """ANSI colours only on a terminal that shows them."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return None
    if os.name == "nt" and not (os.environ.get("WT_SESSION")
                                or os.environ.get("TERM_PROGRAM")):
        return None
    return {"dim": "\033[2m", "H": "\033[34m", "I": "\033[32m",
            "C": "\033[33m", "R": "\033[35m", "end": "\033[0m"}


def cmd_tree(a):
    """The trees as columns: Code (the stable id), No. (the number on lines
    joining it to its parent), Title, Sides (H human, I instructions,
    C code, R reading; '·' for an empty side), then '!' when the file is
    stale and its model."""
    st = Store(a.project)
    ref = getattr(a, "ref", None) or getattr(a, "branch", None)
    root = st.resolve(ref) if ref else ""
    nodes = []

    def walk(pid):
        for n in st.children(pid):
            nodes.append(n)
            walk(n.id)

    if root:
        nodes.append(st.nodes[root])
        walk(root)
    else:
        walk("")
    if not nodes:
        print("no files yet (create one with: new TITLE)")
        return
    ids = {n.id for n in nodes}
    fancy = _fancy()
    pre = tree_lines([(n.id, n.parent if n.parent in ids else "")
                      for n in nodes], fancy)
    nos = {n.id: pre[n.id] + st.number(n.id) for n in nodes}
    wn = max(display_width(x) for x in list(nos.values()) + ["No."])
    wt = min(40, max(display_width(n.title) for n in nodes))
    wt = max(wt, len("Title"))
    col = _colors()
    print("%-6s    %s  %s  %s" % ("Code", _pad("No.", wn),
                                   _pad("Title", wt, right=True), "Sides"))
    for n in nodes:
        sides = [c if getattr(n, s).strip() else "·"
                 for c, s in zip("HICR", SECTIONS)]
        code = "%s %s " % (n.id, "┄┄" if fancy else "··")
        if col:
            code = col["dim"] + code + col["end"]
            sides = [col[c] + c + col["end"] if c in col else c
                     for c in sides]
        title = n.title
        while display_width(title) > wt:
            title = title[:-2] + "…"
        extra = ("  !" if n.is_stale() else "") + \
            (("  (%s)" % n.model) if n.model else "")
        print("%s%s  %s  %s%s" % (code, _pad(nos[n.id], wn),
                                  _pad(title, wt, right=True),
                                  " ".join(sides), extra))


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


def install_schema(st, src):
    """Copy a JSON Schema file into the project's schemas/ (named by the
    file's stem) after checking it uses only what is checked. Returns the
    name."""
    f = Path(src)
    if not f.is_file():
        die("no such file: %s" % src)
    name = re.sub(r"[^\w\-]", "_", f.stem)
    try:
        schema = json.loads(f.read_text(encoding="utf-8-sig"))
    except ValueError as e:
        die("%s is not JSON (%s)" % (src, e))
    probs = schema_problems(schema)
    if probs:
        die("%s uses what Tawjih does not check: %s" % (src, "; ".join(
            probs[:5])))
    target = schema_file(st.root, name)
    if target.resolve() != f.resolve():
        if target.is_file() and json.loads(target.read_text(
                encoding="utf-8")) != schema:
            die("schemas/%s.json exists with other content; rename the file"
                % name)
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(schema, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    return name


def cmd_contract(a):
    """Show or set a file's contracts (expects, produces) and model."""
    st = Store(a.project)
    n = st.nodes[st.resolve(a.ref)]
    for side in ("expects", "produces"):          # --expects-schema FILE
        src = getattr(a, side + "_schema", None)
        if src:
            if getattr(a, side):
                die("give --%s or --%s-schema, not both" % (side, side))
            setattr(a, side, "schema(%s)" % install_schema(st, src))
    changes = {k: v for k, v in (("expects", a.expects),
                                 ("produces", a.produces),
                                 ("model", a.model),
                                 ("retries", a.retries),
                                 ("fallback", a.fallback),
                                 ("approve", a.approve),
                                 ("split", a.split),
                                 ("weight", a.weight),
                                 ("each", getattr(a, "each", None)))
               if v is not None}
    ev = changes.get("each", "none").strip()
    if ev.lower() not in ("", "none", "yes") and not re.fullmatch(r"[\w\-]+",
                                                                   ev):
        die("each is yes (the input is a list) or the key holding the list")
    if changes.get("split", "none").strip().lower() not in ("", "none") + SPLITS:
        die("split is train, eval or ignore")
    if changes.get("weight", "none").strip().lower() not in ("", "none"):
        try:
            if float(changes["weight"]) < 0:
                raise ValueError
        except ValueError:
            die("weight is a number of 0 or more")
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
        if kind == "schema":
            err = load_schema(st.root, keys[0])[1]
            if err:
                die("%s: %s" % (k, err))
            changes[k] = "schema(%s)" % keys[0]
            continue
        changes[k] = "json(%s)" % ", ".join(keys) if kind == "json" and keys \
            else kind
    if changes:
        _backup(st, n)
        for k, v in changes.items():
            setattr(n, k, "" if v.strip().lower() in ("", "none") else v)
        st.save(n)
    print("%s [%s] %s" % (st.number(n.id), n.id, n.title))
    for k in ("expects", "produces", "model", "retries", "fallback",
              "approve", "split", "weight", "each"):
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
        if a.examples:
            if not n.examples.strip():
                die("%s has no examples to confirm" % n.title)
            n.examples_sha = n.source_digest()
            st.save(n)
            print("%s [%s] %s: examples confirmed" % (st.number(n.id), n.id,
                                                       n.title))
            continue
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
    titles = {x.casefold() for n in st.nodes.values()
              for x in [n.title] + list(n.aliases)}
    words = []
    for r in refs or []:
        # a whole path given in one quoted word: "1 TO 1.1"
        parts = r.split()
        if len(parts) > 1 and r.casefold() not in titles and any(
                _kw(w) for w in parts):
            words += parts
        else:
            words.append(r)
    refs = words
    try:
        if refs:
            flow = build_flow(split_statements(refs), st.resolve,
                              st.path_statements, (name,))
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
        text = Path(a.input_file).read_text(encoding="utf-8-sig")
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
                    retries=a.retries, approve=a.approve,
                    sandbox=getattr(a, "sandbox", False))


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
            if e.get("order"):
                label, style = "LINK", "order"
            elif e["among"] is not None:
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
            arrow = "-.->" if style in ("among", "order") else (
                "==>" if cross else "-->")
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
            if style == "order":
                attrs.append("style=dotted")
            if cross:
                attrs.append("penwidth=2.5")
            out.append("  n_%s -> n_%s%s;" % (
                src, dst, " [%s]" % ", ".join(attrs) if attrs else ""))
        out.append("}")
    text = "\n".join(out) + "\n"
    if fmt == "md":
        what = ("path '%s'" % name) if path is not None else "the trees"
        text = ("# Tawjih network: %s\n\nThick arrows cross from one tree to "
                "another; dashed arrows are AMONG choices or LINKs (order only); STALE marks a "
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
        return contract_fits(pu, ev, st.root) is None
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
        flow = build_flow(statements, quiet, st.path_statements)
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
        for line in prop.read_text(encoding="utf-8-sig").splitlines():
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
        if backend_kind(st.root, a.backend) == "dry":
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
    for line in src.read_text(encoding="utf-8-sig").splitlines():
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
    "examples": "The examples side holds worked questions for evaluation "
                "and training, never sent when a path runs. Write blocks: "
                "'Q:' the question, 'A:' a complete correct answer, "
                "optionally 'E:' a short text any correct answer must "
                "contain, and optionally 'BAD:' lines with typical wrong "
                "answers. Leave a blank line between blocks.",
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
    for sec in SECTIONS + (EXTRA if a.side == "examples" else []):
        parts.append("## Current %s side\n%s" % (sec, getattr(n, sec) or "(empty)"))
    if a.source:
        parts.append("# Source material\n"
                     + Path(a.source).read_text(encoding="utf-8-sig"))
    parts.append("# Request\n" + a.ask)
    prompt = "\n\n".join(parts)
    use_model(a)
    step_model = (getattr(a, "model", None) or n.model
                  or os.environ.get("TAWJIH_MODEL", ""))
    if backend_kind(st.root, a.backend) == "dry":
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
    if any(l.strip() == MARK.format(sec) for l in new.splitlines()
           for sec in SECTIONS + EXTRA):
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
    if getattr(a, "stage", False):
        f = stage_change(st, n, a.side, new, a.ask)
        print("staged for review (%s); see: stage --show, stage --apply"
              % f.name)
        return
    if not a.yes:
        print("save this change? [y/N] ", end="", flush=True)
        if sys.stdin.readline().strip().lower() not in ("y", "yes", "نعم"):
            print("not saved")
            return
    bak = _backup(st, n)
    if a.side == "examples":
        probs = parse_examples(new)[1]
        if probs:
            die("the proposed examples are not readable (%s); file unchanged"
                % "; ".join(probs[:3]))
    setattr(n, a.side, new)
    if a.side in ("instructions", "code"):
        n.built_sha = n.source_digest()
    if a.side == "examples":
        n.examples_sha = n.source_digest()
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


def audit_warnings(st, flow):
    """Short warnings printed before a run with --allow-code."""
    for i in flow.nodes:
        found = [f for f in audit_code(st.nodes[i].code)
                 if f[1] == "danger"] if st.nodes[i].code.strip() else []
        if found:
            print("  audit: %s [%s]: %s" % (st.nodes[i].title, i, "; ".join(
                "line %d %s" % (l, m) for l, _, m in found[:3])),
                file=sys.stderr)


# --------------------------------------------------------------------- stage
def _stage_dir(st):
    return st.root / "stage"


def stage_change(st, n, side, new, why=""):
    d = _stage_dir(st)
    d.mkdir(exist_ok=True)
    f = d / ("%s.%s.json" % (n.id, side))
    f.write_text(json.dumps({"id": n.id, "side": side, "text": new,
                             "base": getattr(n, side), "why": why,
                             "staged": time.strftime("%Y-%m-%d %H:%M:%S")},
                            ensure_ascii=False, indent=2), encoding="utf-8")
    return f


__all__ = [
    'AUDIT_CALLS', 'AUDIT_METHODS', 'AUDIT_MODULES', 'AUDIT_PREFIX',
    'EDITOR_ALIASES', 'GENERATE_HEAD', 'GRAPH_FORMATS', 'GUI_PARALLEL',
    'NEEDS_WAIT', 'SIDE_NOTES', 'SPLIT_FLAGS', 'TANGLE_HEAD', 'VSCODE',
    '_backup', '_dot', '_flow_from_args', '_mm', '_node_lines',
    '_proposal_file', '_run_from_args', '_scope_ids', '_stage_dir',
    'audit_code', 'audit_warnings', 'check_proposal', 'cmd_cache',
    'cmd_check', 'cmd_confirm', 'cmd_contract', 'cmd_edit',
    'cmd_generate', 'cmd_graph', 'cmd_import', 'cmd_init', 'cmd_link',
    'cmd_new', 'cmd_open', 'cmd_path', 'cmd_produce', 'cmd_run',
    'cmd_show', 'cmd_tangle', 'cmd_tree', 'cmd_trees', 'contract_route',
    'editor_command', 'graph_source', 'open_editors', 'stage_change',
    'strip_fence', 'tangle_source',
]
