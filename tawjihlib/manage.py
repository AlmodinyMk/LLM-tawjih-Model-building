"""Managing the trees, history, state and sync, tags, audit.

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import contextlib
import difflib
import re
import shutil
import sys
import time
import uuid

from .core import (
    Die, EXTRA, FlowError, LEVELS, Node, SECTIONS, Store, _kw, die,
    parse_level,
)
from .commands import (
    _backup, _flow_from_args, audit_code,
)


# ------------------------------------------------------- managing the trees
# Restructuring keeps ids, so paths (which store ids) stay valid; only the
# positional numbers change.

def _paths_using(st, ids):
    """{path name: [ids of the given set it uses]} over saved paths."""
    out, calls = {}, {}
    ids = set(ids)
    if st.pdir.is_dir():
        for f in sorted(st.pdir.glob("*.path")):
            try:
                lines = st.load_path(f.stem, True)       # comments removed
            except Die:
                continue
            words = " ".join(lines).split()
            toks = set(re.findall(r"\b[0-9a-f]{6}\b", " ".join(lines)))
            calls[f.stem] = {b.strip("'\"") for a, b in zip(words, words[1:])
                             if _kw(a) == "CALL"}
            hit = sorted(toks & ids)
            if hit:
                out[f.stem] = hit
    changed = True                  # a path that CALLs one using them
    while changed:
        changed = False
        for name, called in calls.items():
            got = sorted(set(out.get(name, [])).union(
                *(out[c] for c in called if c in out)))
            if got != out.get(name, []):
                out[name] = got
                changed = True
    return {k: v for k, v in out.items() if v}


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
                 tags=list(o.tags), retries=o.retries, approve=o.approve,
                 examples=o.examples, split=o.split, weight=o.weight,
                 examples_sha=o.examples_sha)
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
    blocked = [st.nodes[i] for i in ids if st.nodes[i].locked
               or st.nodes[i].mount]
    if blocked:
        die("not removed: %s %s locked or mounted" % (
            ", ".join("%s [%s]" % (n.title, n.id) for n in blocked),
            "is" if len(blocked) == 1 else "are"))
    for i in ids:
        _backup(st, st.nodes[i])
        st.remove_file(i)
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
    if B.examples.strip():
        A.examples = "\n\n".join(x for x in (A.examples.strip("\n"),
                                             B.examples.strip("\n")) if x)
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
                node = Node.parse(f.read_text(encoding="utf-8-sig"))
            except ValueError:
                continue
            key = (m.group(1), m.group(2), int(m.group(3) or 1))
            out.append((key, f, node))
    out.sort(key=lambda x: x[0], reverse=True)
    return [("%s-%s-%s %s:%s:%s" % (k[0][:4], k[0][4:6], k[0][6:], k[1][:2],
                                     k[1][2:4], k[1][4:]), f, node)
            for k, f, node in out]


HEADER_FIELDS = ("title", "parent", "level", "order", "links", "model",
                 "expects", "produces", "tags", "retries", "fallback",
                 "approve", "split", "weight", "each", "locked", "aliases")


def _changes(old, new):
    """What differs between two versions of a file."""
    out = [s for s in SECTIONS + EXTRA
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
        elif s in SECTIONS + EXTRA:
            out.append(s)
        else:
            die("unknown side '%s' (human, instructions, code, machine, "
                "reading, examples)" % s)
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
        ("examples older than the human or reading side",
         [i for i in scope if st.nodes[i].examples_stale()]),
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


__all__ = [
    'HEADER_FIELDS', '_among_deciders', '_changes', '_clean_tags',
    '_links_into', '_passages', '_paths_using', '_pick', '_place',
    '_remove_files', '_renumber', '_resolve_any', '_shift_levels',
    '_side_names', '_target_parent', '_version', '_versions', 'cmd_audit',
    'cmd_clean', 'cmd_copy', 'cmd_diff', 'cmd_log', 'cmd_merge',
    'cmd_move', 'cmd_refs', 'cmd_rename', 'cmd_revert', 'cmd_rm',
    'cmd_split', 'cmd_status', 'cmd_sync', 'cmd_tag', 'cmd_untag',
]
