"""Lock, alias, review, stage, mount, sandbox, trace, release, seal, inherit, templates, gc, compact, format, archive, glossary, doctor, schedules and triggers.

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import contextlib
import difflib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .core import (
    Die, EXTRA, FlowError, HEAD_FILE, LEVELS, Node, SECTIONS, Store, __version__,
    RECORD_WHO, _git_name, _kw, die, first_identity, identity_file,
    load_identity, parse_examples, who,
    who_label,
    key_values,
)
from .providers import (
    BACKENDS, DEFAULT_POLICY, POLICY_DEFAULT, POLICY_FILE, POLICY_KEYS,
    ProviderError, call_model, load_policy, load_providers, providers_file,
)
from .engine import (
    CodeTimeout, SANDBOX_LIMITS, machine_side, run_code, run_flow,
)
from .commands import (
    _backup, _stage_dir,
)
from .manage import (
    _among_deciders, _changes, _links_into, _paths_using, _place,
    _renumber,
)
from .outputs import (
    _run_tool, cmd_stats,
)
from .evaluate import (
    _read_cases,
)


# ------------------------------------------------- lock, alias, review, mount
def _targets(st, a):
    ids = []
    for ref in a.refs:
        nid = st.resolve(ref)
        ids += [i for i in (st.subtree_ids(nid) if getattr(a, "branch", False)
                            else [nid]) if i not in ids]
    return ids


def cmd_lock(a):
    """Lock files (with --branch, whole branches): their sides, title,
    contracts and links cannot change until unlocked; they cannot be
    removed. Moving them stays possible."""
    st = Store(a.project)
    for i in _targets(st, a):
        n = st.nodes[i]
        if n.mount:
            continue
        if not n.locked:
            n.locked = "yes"
            st.save(n)
        print("locked  %s [%s] %s" % (st.number(i), i, n.title))


def cmd_unlock(a):
    st = Store(a.project)
    for i in _targets(st, a):
        n = st.nodes[i]
        if n.locked:
            n.locked = ""
            st.save(n)
        print("unlocked  %s [%s] %s" % (st.number(i), i, n.title))


def cmd_alias(a):
    """Give a file other names usable wherever a REF is asked for; without
    names, list all aliases."""
    st = Store(a.project)
    if not a.ref:
        rows = [(x, n) for n in st.nodes.values() for x in n.aliases]
        for x, n in sorted(rows, key=lambda r: r[0].casefold()):
            print("%-20s %s [%s] %s" % (x, st.number(n.id), n.id, n.title))
        if not rows:
            print("no aliases yet (add one with: alias REF NAME)")
        return
    nid = st.resolve(a.ref)
    n = st.nodes[nid]
    for name in a.names:
        name = name.strip()
        if not re.fullmatch(r"[\w\-.]+", name) or re.fullmatch(
                r"[0-9a-f]{6}|\d+(\.\d+)*", name):
            die("an alias is one word that is neither an id nor a number: "
                + name)
        if name in st.nodes:
            die("%s is an id" % name)
        if _kw(name):
            die("%s is a word of the path language" % name)
        other = [m for m in st.nodes.values() if m.id != nid and any(
            x.casefold() == name.casefold() for x in m.aliases)]
        if other:
            die("%s is already an alias of %s" % (name, other[0].title))
        if name.casefold() not in (x.casefold() for x in n.aliases):
            n.aliases.append(name)
    st.save(n)
    print("%s [%s] aliases: %s" % (n.title, nid, ", ".join(n.aliases) or "none"))


def cmd_unalias(a):
    st = Store(a.project)
    nid = st.resolve(a.ref)
    n = st.nodes[nid]
    drop = {x.casefold() for x in a.names}
    n.aliases = [x for x in n.aliases if x.casefold() not in drop]
    st.save(n)
    print("%s [%s] aliases: %s" % (n.title, nid, ", ".join(n.aliases) or "none"))


def cmd_review(a):
    """Mark files as reviewed by a person (--ok), or list the files that
    were never reviewed or changed after their review."""
    st = Store(a.project)
    if a.ok:
        if not a.refs:
            die("name the files you reviewed")
        for i in _targets(st, a):
            n = st.nodes[i]
            n.reviewed_sha = n.content_digest()
            n.reviewed_on = time.strftime("%Y-%m-%d")
            n.reviewed_by = who_label(who())
            st.save(n)
            print("reviewed  %s [%s] %s%s" % (st.number(i), i, n.title,
                                              "  by " + n.reviewed_by
                                              if n.reviewed_by else ""))
        return
    ids = _targets(st, a) if a.refs else st.subtree_ids("")
    never = [i for i in ids if not st.nodes[i].reviewed_sha]
    changed = [i for i in ids if st.nodes[i].reviewed_sha and
               st.nodes[i].reviewed_sha != st.nodes[i].content_digest()]
    for title, group in (("changed since reviewed", changed),
                         ("never reviewed", never)):
        if group:
            print("%s: %d" % (title, len(group)))
            for i in group:
                n = st.nodes[i]
                print("  %-10s [%s] %s%s" % (
                    st.number(i), i, n.title,
                    "  (reviewed %s%s)" % (n.reviewed_on, " by " +
                                           n.reviewed_by if n.reviewed_by
                                           else "") if n.reviewed_on else ""))
    if not (never or changed):
        print("every file is reviewed as it is (%d)" % len(ids))


def cmd_mount(a):
    """Use the files of another project, read-only, in this one's paths;
    without arguments, list mounts."""
    root = Path(a.project)
    mf = root / "mounts.json"
    mounts = json.loads(mf.read_text(encoding="utf-8-sig")) if mf.is_file() else {}
    if not a.where:
        for name, where in sorted(mounts.items()):
            print("%-12s %s" % (name, where))
        if not mounts:
            print("nothing mounted")
        return
    name = a.name or Path(a.where).resolve().name
    if not re.fullmatch(r"[\w\-]+", name):
        die("the mount name is one word")
    if name in mounts:
        die("%s is already mounted" % name)
    if not (Path(a.where) / "files").is_dir():
        die("%s is not a Tawjih project" % a.where)
    if Path(a.where).resolve() == root.resolve():
        die("a project cannot mount itself")
    mounts[name] = os.path.relpath(Path(a.where).resolve(), root.resolve())
    mf.write_text(json.dumps(mounts, indent=2), encoding="utf-8")
    try:
        st = Store(a.project)
    except Die:
        del mounts[name]
        mf.write_text(json.dumps(mounts, indent=2), encoding="utf-8")
        raise
    count = sum(1 for n in st.nodes.values() if n.mount == name)
    print("mounted %s: %d file(s), read-only" % (name, count))


def cmd_unmount(a):
    root = Path(a.project)
    mf = root / "mounts.json"
    mounts = json.loads(mf.read_text(encoding="utf-8-sig")) if mf.is_file() else {}
    if a.name not in mounts:
        die("%s is not mounted" % a.name)
    st = Store(a.project)
    ids = [i for i, n in st.nodes.items() if n.mount == a.name]
    used = _paths_using(st, ids)
    if used:
        die("not unmounted: path(s) %s use its files" % ", ".join(sorted(used)))
    del mounts[a.name]
    mf.write_text(json.dumps(mounts, indent=2), encoding="utf-8")
    print("unmounted %s" % a.name)


def _staged(st):
    d = _stage_dir(st)
    out = []
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            out.append((f, json.loads(f.read_text(encoding="utf-8-sig"))))
        except ValueError:
            continue
    return out


def apply_side(st, n, side, new):
    """Write one side as edit does: a copy in history/, digests updated."""
    if side == "examples":
        probs = parse_examples(new)[1]
        if probs:
            die("the examples are not readable (%s)" % "; ".join(probs[:3]))
    if side == "code" and new.strip():
        try:
            compile(new, n.id + ".code", "exec")
        except SyntaxError as e:
            die("the code has a syntax error (%s)" % e)
    bak = _backup(st, n)
    setattr(n, side, new)
    if side in ("instructions", "code"):
        n.built_sha = n.source_digest()
    if side == "examples":
        n.examples_sha = n.source_digest()
    st.save(n)
    return bak


def cmd_stage(a):
    """Changes proposed by edit --stage, waiting for review: list them,
    --show one, --apply or --drop them (all, or those of REF)."""
    st = Store(a.project)
    items = _staged(st)
    if a.ref:
        nid = st.resolve(a.ref)
        items = [(f, d) for f, d in items if d["id"] == nid]
    if not items:
        print("nothing staged")
        return
    for f, d in items:
        n = st.nodes.get(d["id"])
        label = "%s [%s] %s: %s" % (st.number(d["id"]) if n else "?", d["id"],
                                    n.title if n else "(missing)", d["side"])
        if a.show:
            print("== " + label)
            for line in difflib.unified_diff(
                    d["base"].splitlines(), d["text"].splitlines(), "before",
                    "after", lineterm=""):
                print(line)
        elif a.apply:
            if n is None:
                print("skipped (file missing): " + label)
                continue
            if getattr(n, d["side"]) != d["base"]:
                print("skipped (the side changed since it was staged; drop "
                      "it and stage again): " + label)
                continue
            apply_side(st, n, d["side"], d["text"])
            f.unlink()
            print("applied " + label)
        elif a.drop:
            f.unlink()
            print("dropped " + label)
        else:
            print("%s  (staged %s)%s" % (label, d["staged"],
                                         "  " + d["why"] if d["why"] else ""))
    with contextlib.suppress(OSError):
        _stage_dir(st).rmdir()


def cmd_sandbox(a):
    """Run one file's code side in the sandbox on an input, to see what it
    does there."""
    st = Store(a.project)
    nid = st.resolve(a.ref)
    n = st.nodes[nid]
    if not n.code.strip():
        die("%s has no code side" % n.title)
    if not load_policy(st.root)["allow_code"]:
        die("the project's policy (%s) does not allow running code"
            % POLICY_FILE)
    text = Path(a.input_file).read_text(encoding="utf-8-sig") if a.input_file \
        else (a.input or "")
    shared, before = {}, dict(SANDBOX_LIMITS)
    SANDBOX_LIMITS["memory_mb"] = a.memory
    try:
        out = run_code(n, text, shared, text, store=st, timeout=a.timeout,
                       sandbox=True)
    except (Exception, CodeTimeout) as e:
        die("%s: %s" % (type(e).__name__, e))
    finally:
        SANDBOX_LIMITS.update(before)
    print(out if out is not None else "(no output)")
    if shared:
        print("vars: %s" % json.dumps(shared, ensure_ascii=False),
              file=sys.stderr)


# --------------------------------------------------------------------- trace
def cmd_trace(a):
    """--impact REF: what a change to a file would reach; --output FILE:
    what a result or a dataset came from; --cost: what runs cost."""
    st = Store(a.project)
    if a.cost:
        a.days = None
        cmd_stats(a)
        return 0
    if a.impact:
        nid = st.resolve(a.impact)
        n = st.nodes[nid]
        print("%s [%s] %s" % (st.number(nid), nid, n.title))
        kids = st.subtree_ids(nid)[1:]
        if kids:
            print("  files under it: %d" % len(kids))
            below = sorted(_paths_using(st, kids))
            if below:
                print("  paths using files under it: " + ", ".join(below))
        for s_, _ in _links_into(st, [nid]):
            print("  linked from: %s [%s] %s" % (st.number(s_), s_,
                                                 st.nodes[s_].title))
        for name in sorted(_paths_using(st, [nid])):
            try:
                flow = st.load_flow(name)
            except FlowError:
                continue
            down, todo = [], [nid]
            while todo:
                cur = todo.pop()
                for e in flow.edges:
                    if e["src"] == cur and e["dst"] not in down:
                        down.append(e["dst"])
                        todo.append(e["dst"])
            cf = st.pdir / (name + ".cases.jsonl")
            cases = len(_read_cases(cf)) if cf.is_file() else 0
            print("  path %s: %d step(s) after it%s%s%s" % (
                name, len(down),
                " (%s)" % ", ".join(st.nodes[d].title for d in down)
                if down else "",
                "; %d evaluation case(s)" % cases if cases else "",
                "; a baseline" if (st.pdir / (name + ".baseline.json"))
                .is_file() else ""))
        for d in _among_deciders(st, nid):
            print("  chosen by AMONG in: %s [%s]" % (st.nodes[d].title, d))
        for fb in st.nodes.values():
            if fb.fallback == nid:
                print("  fallback of: %s [%s]" % (fb.title, fb.id))
        digest = hashlib.sha256(st.node_bytes(nid)).hexdigest()
        for m in sorted(st.root.rglob("*.manifest.json")):
            try:
                srcs = json.loads(m.read_text(encoding="utf-8-sig")).get("from", {})
            except ValueError:
                continue
            if nid in srcs:
                print("  in dataset %s%s" % (
                    m.name.replace(".manifest.json", ""),
                    "" if srcs[nid] == digest else " (made from an older "
                    "version)"))
        rel = sorted((st.root / "releases").glob("*.zip")) \
            if (st.root / "releases").is_dir() else []
        inrel = [r.stem for r in rel if _release_has(r, "files/%s.md" % nid)
                 or _release_has(r, "files/%s/%s" % (nid, HEAD_FILE))]
        if inrel:
            print("  in release(s): %s" % ", ".join(inrel))
        if n.locked:
            print("  locked: unlock it before changing it")
        return
    if a.output:
        f = Path(a.output)
        if not f.is_file():
            die("no such file: %s" % f)
        if f.name.endswith(".manifest.json"):
            man = json.loads(f.read_text(encoding="utf-8-sig"))
            print("%s: %d example(s) from %s, %s" % (
                f.name, man.get("examples", 0), man.get("source"),
                man.get("created")))
            for i, sha in man.get("from", {}).items():
                _trace_file(st, i, sha)
            return
        rec = None
        if f.suffix == ".json" and f.name.startswith("production-"):
            rec = json.loads(f.read_text(encoding="utf-8-sig"))
        else:            # a run's log or result: find its record
            stem = f.name.split(".")[0]
            for r in sorted((st.root / "history").glob("production-*.json")):
                try:
                    d = json.loads(r.read_text(encoding="utf-8-sig"))
                except ValueError:
                    continue
                if Path(d.get("log", "")).stem == stem or \
                        Path(d.get("result", "")).name == f.name:
                    rec = d
        if rec is None:
            die("no run record found for %s" % f)
        print("run of path %s, %s, %s" % (rec.get("name"), rec.get("started"),
                                          rec.get("status")))
        for p in rec.get("steps", []):
            n = st.nodes.get(p["id"])
            now = machine_side(n)[2] if n else ""
            print("  step %s %s [%s] %s: %s%s" % (
                p.get("step"), p.get("number"), p["id"], p.get("title"),
                p.get("status", ""), "" if not n else (
                    "" if now == p.get("machine_sha") else
                    "  (its machine side has changed since)")))
            if n:
                for src in sorted(set(re.findall(r"\[source: ([^\]]+)\]",
                                                 n.reading))):
                    print("      reading from: " + src)
        return
    die("give --impact REF, --output FILE or --cost")


def _trace_file(st, nid, sha):
    n = st.nodes.get(nid)
    if n is None:
        print("  [%s] (no longer in the project)" % nid)
        return
    now = hashlib.sha256(st.node_bytes(nid)).hexdigest()
    print("  %s [%s] %s%s" % (st.number(nid), nid, n.title,
                              "" if now == sha else "  (changed since)"))
    for src in sorted(set(re.findall(r"\[source: ([^\]]+)\]", n.reading))):
        print("      reading from: " + src)


# ---------------------------------------------- release, snapshot, seal
RELEASE_KEEP = ("files", "paths")
RELEASE_EXTRA = ("redact.txt", "prices.json", "mounts.json")


def _project_entries(root):
    """Paths (relative, '/'-separated) that make up the project's state."""
    out = []
    for d in RELEASE_KEEP:
        base = root / d
        if base.is_dir():
            out += [f.relative_to(root).as_posix() for f in sorted(base.rglob(
                "*")) if f.is_file()]
    out += [x for x in RELEASE_EXTRA if (root / x).is_file()]
    return out


def _digest_map(root, entries):
    return {e: hashlib.sha256((root / e).read_bytes()).hexdigest()
            for e in entries}


def _overall(digests):
    return hashlib.sha256("\n".join("%s %s" % (k, digests[k])
                                    for k in sorted(digests)).encode()
                          ).hexdigest()


def _release_has(zpath, entry):
    import zipfile
    try:
        with zipfile.ZipFile(zpath) as z:
            return entry in z.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def _release_read(st, name):
    import zipfile
    if not re.fullmatch(r"[\w.\-]+", name) or name.startswith("."):
        die("a release name is one word")
    f = st.root / "releases" / (name + ".zip")
    if not f.is_file():
        die("no release %s (see: release list)" % name)
    try:
        z = zipfile.ZipFile(f)
    except (OSError, zipfile.BadZipFile) as e:
        die("release %s is not a readable zip file (%s)" % (name, e))
    try:
        man = json.loads(z.read("release.json"))
        files = man["files"]
        if not isinstance(files, dict):
            raise ValueError
    except (KeyError, ValueError, TypeError):
        z.close()
        die("release %s has no readable release.json" % name)
    # a release may come from elsewhere: only the project's own kinds of
    # file, and nothing that would be written outside the project
    names = set(z.namelist())
    bad = [e for e in files if not (
        (re.fullmatch(r"(files|paths)/[\w.\-]+", e) or re.fullmatch(
            r"files/[0-9a-f]{6}/(head\.json|human\.md|machine\.md|"
            r"science\.md)", e)) and ".." not in e
        or e in RELEASE_EXTRA) or e not in names]
    if bad:
        z.close()
        die("release %s is not safe to use: %s" % (name, ", ".join(
            sorted(bad)[:5])))
    return z, man


def make_release(st, name, note=""):
    import zipfile
    if not re.fullmatch(r"[\w.\-]+", name):
        die("a release name is one word (letters, digits, '.', '-', '_')")
    rdir = st.root / "releases"
    rdir.mkdir(exist_ok=True)
    f = rdir / (name + ".zip")
    if f.exists():
        die("release %s already exists" % name)
    entries = _project_entries(st.root)
    digests = _digest_map(st.root, entries)
    man = {"name": name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "by": who(),
           "tawjih": __version__, "note": note, "files": digests,
           "digest": _overall(digests)}
    with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("release.json", json.dumps(man, ensure_ascii=False,
                                              indent=2))
        for e in entries:
            z.write(st.root / e, e)
    return f, man


def cmd_release(a):
    """Snapshots of the whole project (files, paths with their cases and
    baselines, redact.txt, prices.json): create, list, diff, verify,
    rollback."""
    st = Store(a.project)
    act = a.action
    if act == "create":
        if not a.names:
            die("release create NAME [--note TEXT]")
        f, man = make_release(st, a.names[0], a.note or "")
        print("release %s: %d file(s), digest %s… (%s)" % (
            man["name"], len(man["files"]), man["digest"][:16], f))
    elif act == "list":
        rdir = st.root / "releases"
        rels = sorted(rdir.glob("*.zip")) if rdir.is_dir() else []
        for r in rels:
            z, man = _release_read(st, r.stem)
            z.close()
            print("%-24s %s  %d file(s)  %s" % (r.stem, man["created"],
                                                len(man["files"]),
                                                man.get("note", "")))
        if not rels:
            print("no releases yet (release create NAME)")
    elif act == "verify":
        if not a.names:
            die("release verify NAME")
        z, man = _release_read(st, a.names[0])
        with z:
            bad = [e for e, h in man["files"].items()
                   if hashlib.sha256(z.read(e)).hexdigest() != h]
        if bad or _overall(man["files"]) != man["digest"]:
            print("release %s is damaged or was altered: %s" % (
                a.names[0], ", ".join(bad) or "release.json"))
            return 1
        print("release %s is intact (%d file(s))" % (a.names[0],
                                                     len(man["files"])))
    elif act == "diff":
        if not a.names:
            die("release diff A [B]  (B defaults to the current project)")
        za, ma = _release_read(st, a.names[0])
        if len(a.names) > 1:
            zb, mb = _release_read(st, a.names[1])
            fb, read_b = mb["files"], (lambda e: zb.read(e).decode("utf-8"))
        else:
            zb = None
            fb = _digest_map(st.root, _project_entries(st.root))
            read_b = (lambda e: (st.root / e).read_text(encoding="utf-8-sig"))
        fa = ma["files"]
        changed = 0
        for e in sorted(set(fa) | set(fb)):
            if fa.get(e) == fb.get(e):
                continue
            changed += 1
            what = "added" if e not in fa else "removed" if e not in fb \
                else "changed"
            extra = ""
            m = re.fullmatch(r"files/([0-9a-f]{6})/(\w+)\.(md|json)", e)
            if m:
                nid, face = m.group(1), m.group(2)
                title = st.nodes[nid].title if nid in st.nodes else nid
                extra = "  %s: %s" % (title, "header" if face == "head"
                                      else face + " face")
            elif what == "changed" and e.startswith("files/"):
                try:
                    old = Node.parse(za.read(e).decode("utf-8"))
                    new = Node.parse(read_b(e))
                    extra = "  %s: %s" % (new.title, ", ".join(
                        _changes(old, new)) or "header")
                except ValueError:
                    pass
            print("%-8s %s%s" % (what, e, extra))
        za.close()
        if zb:
            zb.close()
        if not changed:
            print("no differences")
    elif act == "rollback":
        if not a.names:
            die("release rollback NAME --yes")
        z, man = _release_read(st, a.names[0])
        with z:
            def entries(nid):
                return [e for e in set(man["files"]) | set(
                    _project_entries(st.root)) if e == "files/%s.md" % nid
                    or e.startswith("files/%s/" % nid)]
            locked = [n for n in st.nodes.values() if n.locked and not n.mount
                      and any(man["files"].get(e) != (_digest_map(
                          st.root, [e])[e] if (st.root / e).is_file()
                          else None) for e in entries(n.id))]
            if locked and not a.force:
                die("locked file(s) would change: %s (unlock them, or "
                    "--force)" % ", ".join(n.title for n in locked))
            if not a.yes:
                print("rollback replaces files/ and paths/ with release %s; "
                      "the current state is saved first as a release. Run "
                      "again with --yes" % a.names[0])
                return
            before, _ = make_release(st, "before-rollback-" + time.strftime(
                "%Y%m%d-%H%M%S"), "automatic, before rolling back to "
                + a.names[0])
            for e in _project_entries(st.root):
                if e.startswith(RELEASE_KEEP) and e not in man["files"]:
                    (st.root / e).unlink()
            for e in man["files"]:
                target = st.root / e
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(e))
        Store(a.project).record_all()
        print("rolled back to %s; the state before is release %s" % (
            a.names[0], before.stem))


def cmd_snapshot(a):
    """A release named by the time, for a quick save point."""
    st = Store(a.project)
    f, man = make_release(st, "snapshot-" + time.strftime("%Y%m%d-%H%M%S"),
                          a.note or "")
    print("snapshot %s: %d file(s) (%s)" % (man["name"], len(man["files"]), f))


def cmd_seal(a):
    """Record the digest of every file of the project in seal.json; with
    --sign also sign it with GnuPG (seal.json.asc)."""
    st = Store(a.project)
    digests = _digest_map(st.root, _project_entries(st.root))
    seal = {"sealed": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tawjih": __version__, "files": digests,
            "digest": _overall(digests)}
    f = st.root / "seal.json"
    f.write_text(json.dumps(seal, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    print("sealed %d file(s): %s…" % (len(digests), seal["digest"][:16]))
    if a.sign:
        gpg = shutil.which("gpg")
        if not gpg:
            die("signing needs GnuPG (gpg), which is not installed; the seal "
                "is written unsigned")
        _run_tool([gpg, "--batch", "--yes", "--armor", "--detach-sign",
                   str(f)], "gpg")
        print("signed: %s.asc" % f.name)


def cmd_verify(a):
    """Compare the project with seal.json (and check its signature when
    seal.json.asc exists)."""
    st = Store(a.project)
    f = st.root / "seal.json"
    if not f.is_file():
        die("no seal.json (make one with: seal)")
    seal = json.loads(f.read_text(encoding="utf-8-sig"))
    bad = 0
    asc = f.with_name("seal.json.asc")
    if asc.is_file():
        gpg = shutil.which("gpg")
        if not gpg:
            print("warning: seal.json.asc exists but gpg is not installed; "
                  "the signature was not checked")
        else:
            r = subprocess.run([gpg, "--batch", "--verify", str(asc), str(f)],
                               capture_output=True, text=True)
            print("signature: %s" % ("good" if r.returncode == 0 else
                                     "BAD: " + r.stderr.strip()[-200:]))
            bad += r.returncode != 0
    if _overall(seal["files"]) != seal["digest"]:
        print("seal.json itself was altered")
        bad += 1
    now = _digest_map(st.root, _project_entries(st.root))
    for e in sorted(set(now) | set(seal["files"])):
        if now.get(e) != seal["files"].get(e):
            bad += 1
            print("%s %s" % ("added  " if e not in seal["files"] else
                             "removed" if e not in now else "changed", e))
    print("matches the seal of %s" % seal["sealed"] if not bad else
          "%d difference(s) from the seal of %s" % (bad, seal["sealed"]))
    return 1 if bad else 0


# ------------------------------------------- inherit, template, scaffold
INHERITABLE = ("model", "expects", "produces", "tags", "split", "weight",
               "retries", "approve")


def cmd_inherit(a):
    """Give the files under REF its values of the chosen fields: only where
    they have none, unless --overwrite; tags are added."""
    st = Store(a.project)
    nid = st.resolve(a.ref)
    src = st.nodes[nid]
    fields = [f.strip() for f in (a.fields or "model,tags,split,weight")
              .split(",") if f.strip()]
    bad = [f for f in fields if f not in INHERITABLE]
    if bad:
        die("cannot inherit %s (fields: %s)" % (", ".join(bad),
                                                ", ".join(INHERITABLE)))
    changed = 0
    for i in st.subtree_ids(nid)[1:]:
        n = st.nodes[i]
        if n.mount or n.locked:
            continue
        before = n.dump()
        for f in fields:
            v = getattr(src, f)
            if f == "tags":
                n.tags = n.tags + [t for t in src.tags if t not in n.tags]
            elif v and (a.overwrite or not getattr(n, f)):
                setattr(n, f, v)
        if n.dump() != before:
            _backup(st, Node.parse(before))
            st.save(n)
            changed += 1
            print("%-10s [%s] %s" % (st.number(i), i, n.title))
    print("%d file(s) changed" % changed)


def _template_dir(st):
    return st.root / "templates"


def cmd_template(a):
    """save NAME REF [--branch]: keep a file (or branch) as a template;
    list; rm NAME."""
    st = Store(a.project)
    tdir = _template_dir(st)
    if a.action == "list":
        ts = sorted(tdir.glob("*.json")) if tdir.is_dir() else []
        for t in ts:
            d = json.loads(t.read_text(encoding="utf-8-sig"))
            print("%-20s %d file(s)  from %s" % (t.stem, len(d["files"]),
                                                  d.get("from", "")))
        if not ts:
            print("no templates yet (template save NAME REF)")
        return
    if a.action == "rm":
        if not a.args:
            die("template rm NAME")
        f = tdir / (a.args[0] + ".json")
        if not f.is_file():
            die("no template %s" % a.args[0])
        f.unlink()
        print("removed template %s" % a.args[0])
        return
    if len(a.args) != 2:
        die("template save NAME REF [--branch]")
    name, ref = a.args
    if not re.fullmatch(r"[\w\-]+", name):
        die("a template name is one word")
    nid = st.resolve(ref)
    ids = st.subtree_ids(nid) if a.branch else [nid]
    files = []
    for i in ids:
        n = st.nodes[i]
        files.append({"parent": ids.index(n.parent) if n.parent in ids
                      else None, "level": n.level - st.nodes[nid].level,
                      "text": n.dump()})
    tdir.mkdir(exist_ok=True)
    (tdir / (name + ".json")).write_text(json.dumps(
        {"from": "%s [%s]" % (st.nodes[nid].title, nid), "files": files},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("template %s: %d file(s)" % (name, len(files)))


def cmd_scaffold(a):
    """New files from a template, under --parent (or as a new tree)."""
    st = Store(a.project)
    f = _template_dir(st) / (a.name + ".json")
    if not f.is_file():
        die("no template %s (see: template list)" % a.name)
    tpl = json.loads(f.read_text(encoding="utf-8-sig"))["files"]
    parent = st.resolve(a.parent) if a.parent else ""
    base = st.nodes[parent].level + 1 if parent else 1
    if base + max(x["level"] for x in tpl) > len(LEVELS):
        die("the template would go deeper than level %d there" % len(LEVELS))
    made = []
    for k, x in enumerate(tpl):
        n = Node.parse(x["text"])
        p = made[x["parent"]].id if x["parent"] is not None else parent
        new = st.new_node(a.title if k == 0 and a.title else n.title,
                          base + x["level"], p)
        for attr in ("human", "instructions", "code", "reading", "examples",
                     "model", "expects", "produces", "retries", "approve",
                     "split", "weight", "each", "built_sha", "examples_sha"):
            setattr(new, attr, getattr(n, attr))
        new.tags = list(n.tags)
        st.save(new)
        made.append(new)
    for n in made:
        print("%-10s [%s] %s" % (st.number(n.id), n.id, n.title))


# ------------------------------------------------- gc, compact, format
def cmd_gc(a):
    """Remove old saved versions from history/ (beyond the newest --keep
    per file, or older than --days) and cached answers older than
    --cache-days. Without --yes, only counts."""
    st = Store(a.project)
    hdir = st.root / "history"
    drop = []
    if hdir.is_dir():
        by_id = {}
        for f in hdir.glob("*.md"):
            m = re.fullmatch(r"([0-9a-f]{6})-(\d{8}-\d{6})(?:-(\d+))?\.md",
                             f.name)
            if m:
                by_id.setdefault(m.group(1), []).append(
                    ((m.group(2), int(m.group(3) or 1)), f))
        limit = time.time() - a.days * 86400 if a.days else None
        for i, vs in by_id.items():
            vs.sort(reverse=True)
            for k, (_, f) in enumerate(vs):
                # the newest version is never dropped for its age alone
                if k >= a.keep or (k and limit and f.stat().st_mtime < limit):
                    if k == 0 and i not in st.nodes:
                        continue        # the only way back for a removed file
                    drop.append(f)
    if a.cache_days is not None and (st.root / "cache").is_dir():
        limit = time.time() - a.cache_days * 86400
        drop += [f for f in (st.root / "cache").glob("*.json")
                 if f.stat().st_mtime < limit]
    size = sum(f.stat().st_size for f in drop)
    print("%s %d file(s), %.1f KB" % ("removed" if a.yes else "would remove",
                                       len(drop), size / 1024))
    if a.yes:
        for f in drop:
            f.unlink()
    elif drop:
        print("run again with --yes")


def cmd_compact(a):
    """Give siblings the orders 1, 2, ... everywhere and rewrite every file
    in the standard form."""
    st = Store(a.project)
    changed = 0
    parents = {""} | {n.id for n in st.nodes.values()}
    for pid in parents:
        for k, n in enumerate(st.children(pid), 1):
            if n.mount:
                continue
            disk = st.load_node(n.id)
            if n.order != k or n.id in st.legacy or disk is None or \
                    disk.dump() != n.dump() or _faces_differ(st, n):
                n.order = k
                st.save(n)
                changed += 1
    print("%d file(s) rewritten" % changed)


def _faces_differ(st, n):
    """True when a face file on disk is not in the standard form."""
    folder = st.file_of(n.id)
    if folder.suffix == ".md":
        return True
    for name, text in n.faces().items():
        f = folder / name
        if not f.is_file() or f.read_text(encoding="utf-8-sig") != text:
            return True
    return False


def _formatted(n):
    """A copy of n with tidy text: no trailing spaces, at most one blank
    line in a row, one blank line between examples, tags without repeats."""
    m = Node.parse(n.dump())
    for side in SECTIONS + EXTRA:
        t = getattr(m, side)
        t = "\n".join(l.rstrip() for l in t.split("\n"))
        t = re.sub(r"\n{3,}", "\n\n", t).strip("\n")
        if side == "examples" and t:
            t = re.sub(r"\n+(?=(?:Q|س|question)\s*:)", "\n\n", t)
        setattr(m, side, t)
    m.title = " ".join(m.title.split())
    seen = []
    for t in m.tags:
        if t not in seen:
            seen.append(t)
    m.tags = seen
    return m


def cmd_format(a):
    """Tidy the text of files (see _formatted); --check only reports the
    files that would change and exits with 1 if there are any."""
    st = Store(a.project)
    ids = st.subtree_ids(st.resolve(a.branch)) if a.branch else \
        st.subtree_ids("")
    todo = []
    for i in ids:
        n = st.nodes[i]
        if n.mount:
            continue
        m = _formatted(n)
        if m.dump() != n.dump():
            todo.append((n, m))
    for n, m in todo:
        if a.check:
            print("would tidy %s [%s] %s" % (st.number(n.id), n.id, n.title))
        elif n.locked:
            print("skipped, locked: %s [%s]" % (n.title, n.id))
        else:
            _backup(st, n)
            st.save(m)
            print("tidied %s [%s] %s" % (st.number(n.id), n.id, n.title))
    if not todo:
        print("every file is tidy")
    return 1 if a.check and todo else 0


# ----------------------------------------------------- archive, glossary
def cmd_archive(a):
    """Move a branch out of the trees into archive/, where nothing reads it;
    --list shows archived branches, --restore ID brings one back."""
    st = Store(a.project)
    adir = st.root / "archive"
    if a.list:
        roots = []
        for f in sorted(adir.glob("*.md")) if adir.is_dir() else []:
            n = Node.parse(f.read_text(encoding="utf-8-sig"))
            if not (adir / (n.parent + ".md")).is_file():
                roots.append(n)
        for n in roots:
            print("[%s] %s" % (n.id, n.title))
        if not roots:
            print("nothing archived")
        return
    if a.restore:
        rid = a.restore
        f = adir / (rid + ".md")
        if not f.is_file():
            die("no archived file %s" % rid)
        arch = {g.stem: Node.parse(g.read_text(encoding="utf-8-sig"))
                for g in adir.glob("*.md")}
        ids, todo = [], [rid]
        while todo:
            cur = todo.pop()
            ids.append(cur)
            todo += [i for i, n in arch.items() if n.parent == cur]
        root = arch[rid]
        if root.parent and root.parent not in st.nodes:
            root.parent = ""
        for i in ids:
            if i in st.nodes:
                die("[%s] exists in the project already" % i)
        for i in ids:
            n = arch[i]
            st.write(n)
            st.nodes[i] = n
            st.record(n)
            (adir / (i + ".md")).unlink()
        _place(st, rid, root.parent)
        print("restored %d file(s): %s [%s] %s" % (len(ids), st.number(rid),
                                                   rid, root.title))
        return
    if not a.ref:
        die("archive REF [--yes], archive --list, archive --restore ID")
    nid = st.resolve(a.ref)
    ids = st.subtree_ids(nid)
    used = _paths_using(st, ids)
    if used:
        die("not archived: used by path(s) %s" % ", ".join(sorted(used)))
    links = _links_into(st, ids, exclude=ids)
    if links:
        die("not archived: linked from %s" % ", ".join(
            "%s [%s]" % (st.nodes[s_].title, s_) for s_, _ in links))
    blocked = [i for i in ids if st.nodes[i].locked or st.nodes[i].mount]
    if blocked:
        die("not archived: locked or mounted files in the branch")
    if not a.yes:
        print("would archive %d file(s) under %s; run again with --yes"
              % (len(ids), st.nodes[nid].title))
        return
    adir.mkdir(exist_ok=True)
    parent = st.nodes[nid].parent
    for i in ids:
        (adir / (i + ".md")).write_text(st.nodes[i].dump(), encoding="utf-8")
        st.remove_file(i)
        del st.nodes[i]
        st.forget(i)
    _renumber(st, parent)
    print("archived %d file(s); bring them back with: archive --restore %s"
          % (len(ids), nid))


def cmd_glossary(a):
    """Terms the human sides define: **bold** words and lines that start
    'term: definition'. --conflicts shows a term defined differently in
    several files."""
    st = Store(a.project)
    ids = st.subtree_ids(st.resolve(a.branch)) if a.branch else \
        st.subtree_ids("")
    terms = {}
    for i in ids:
        n = st.nodes[i]
        for t in re.findall(r"\*\*([^*\n]{2,60})\*\*", n.human):
            terms.setdefault(" ".join(t.split()), {}).setdefault(i, "")
        for line in n.human.split("\n"):
            m = re.match(r"\s*[-*]?\s*([^:\n]{2,40}?)\s*[:：]\s+(\S.{3,})",
                         line)
            if m and len(m.group(1).split()) <= 4 and not re.search(
                    r"https?$", m.group(1)):
                terms.setdefault(" ".join(m.group(1).split()).strip("*"),
                                 {})[i] = m.group(2).strip()
    if not terms:
        print("no terms found (write **term** or 'term: definition' in the "
              "human sides)")
        return
    lines = []
    for t in sorted(terms, key=str.casefold):
        where = terms[t]
        defs = {d for d in where.values() if d}
        if a.conflicts and len(defs) < 2:
            continue
        lines.append("- **%s**: %s" % (t, "; ".join(
            "%s %s%s" % (st.number(i), st.nodes[i].title,
                         (" (%s)" % d[:80]) if d and (a.conflicts or
                                                      len(defs) > 1) else "")
            for i, d in where.items())))
        if len(defs) == 1:
            lines[-1] += " — " + next(iter(defs))[:120]
    text = "# Glossary\n\n" + "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print("glossary of %d term(s) written to %s" % (len(lines), a.out))
    else:
        sys.stdout.write(text)


# ------------------------------------------------------------ doctor
def cmd_doctor(a):
    """Check the project's files without trusting them: unreadable files,
    ids that do not match file names, missing parents, cycles, broken links,
    paths naming missing files, unreadable state, cache and index, leftovers.
    With --fix (or the command repair) mend what can be mended safely."""
    root = Path(a.project)
    if not root.is_dir():
        die("no such folder: %s" % root)
    fix = a.fix or a.cmd == "repair"
    problems, fixed = [], []
    for d in ("files",):
        if not (root / d).is_dir():
            problems.append("folder %s/ is missing" % d)
            if fix:
                (root / d).mkdir(parents=True, exist_ok=True)
                fixed.append("created %s/" % d)
    nodes, files = {}, {}
    fdir = root / "files"
    entries = [f for f in sorted(fdir.iterdir()) if (f.suffix == ".md" and
               f.is_file()) or (f.is_dir() and re.fullmatch(
                   r"[0-9a-f]{6}", f.name))] if fdir.is_dir() else []
    for f in entries:
        single = f.suffix == ".md"
        stem = f.stem if single else f.name
        try:
            n = Node.parse(f.read_text(encoding="utf-8-sig")) if single \
                else Node.from_faces(f)
        except Exception as e:
            problems.append("%s cannot be read: %s" % (f.name, e))
            if fix:
                q = root / "quarantine"
                q.mkdir(exist_ok=True)
                shutil.move(str(f), str(q / f.name))
                fixed.append("moved %s to quarantine/ to be mended by hand"
                             % f.name)
            continue
        if n.id != stem:
            problems.append("%s holds id %s" % (f.name, n.id))
            target = fdir / (n.id + (".md" if single else ""))
            if fix and not target.exists() and not (fdir / n.id).exists():
                f.rename(target)
                fixed.append("renamed %s to %s" % (f.name, target.name))
                f = target
            elif fix:
                continue
        if n.id in nodes:
            problems.append("%s is there twice (a folder and a .md file)"
                            % n.id)
            continue
        nodes[n.id], files[n.id] = n, f

    def save(n):
        hdir = root / "history"
        hdir.mkdir(exist_ok=True)
        f = files[n.id]
        before = f.read_text(encoding="utf-8-sig") if f.suffix == ".md" \
            else Node.from_faces(f).dump()
        (hdir / ("%s-%s.md" % (n.id, time.strftime("%Y%m%d-%H%M%S")))
         ).write_text(before, encoding="utf-8")
        if f.suffix == ".md":
            f.write_text(n.dump(), encoding="utf-8")
        else:
            for name, text in n.faces().items():
                (f / name).write_text(text, encoding="utf-8")
    for n in nodes.values():
        if n.parent and n.parent not in nodes:
            problems.append("%s (%s): its parent %s is missing" % (
                n.id, n.title, n.parent))
            if fix:
                n.parent = ""
                save(n)
                fixed.append("%s became the root of a tree" % n.id)
    for n in list(nodes.values()):
        seen, cur = [], n.id
        while cur in nodes and cur not in seen:
            seen.append(cur)
            cur = nodes[cur].parent
        if cur in seen and cur == n.id:
            problems.append("%s (%s) is in a cycle of parents" % (n.id,
                                                                  n.title))
            if fix:
                n.parent = ""
                save(n)
                fixed.append("broke the cycle at %s" % n.id)
    for n in nodes.values():
        bad = [t for t, _ in n.links if t not in nodes]
        if bad:
            problems.append("%s (%s): links to missing %s" % (
                n.id, n.title, ", ".join(bad)))
            if fix:
                n.links = [(t, ty) for t, ty in n.links if t not in bad]
                save(n)
                fixed.append("removed broken links from %s" % n.id)
        if n.fallback and n.fallback not in nodes:
            problems.append("%s: fallback %s is missing" % (n.id, n.fallback))
    for f in sorted((root / "paths").glob("*.path")) if (root / "paths") \
            .is_dir() else []:
        text = " ".join(l.split("#", 1)[0] for l in f.read_text(
            encoding="utf-8").splitlines())
        missing = sorted({t for t in re.findall(r"\b[0-9a-f]{6}\b", text)
                          if t not in nodes})
        if missing:
            problems.append("path %s names missing file(s) %s" % (
                f.stem, ", ".join(missing)))
    for name, glob_ in (("state.json", None), ("index/embeddings.json", None)):
        f = root / name
        if f.is_file():
            try:
                json.loads(f.read_text(encoding="utf-8-sig"))
            except ValueError:
                problems.append("%s cannot be read" % name)
                if fix:
                    f.unlink()
                    fixed.append("removed %s (it is rebuilt when needed)"
                                 % name)
    bad_cache = []
    for f in (root / "cache").glob("*.json") if (root / "cache").is_dir() \
            else []:
        try:
            json.loads(f.read_text(encoding="utf-8-sig"))["answer"]
        except (ValueError, KeyError, TypeError):
            bad_cache.append(f)
    if bad_cache:
        problems.append("%d cached answer(s) cannot be read" % len(bad_cache))
        if fix:
            for f in bad_cache:
                f.unlink()
            fixed.append("removed %d unreadable cached answer(s)"
                         % len(bad_cache))
    for d in ("work", "stage"):
        if (root / d).is_dir() and any((root / d).iterdir()):
            problems.append("%s/ holds unfinished work (see: %s)" % (
                d, "sync --clean-work" if d == "work" else "stage"))
    for line in problems:
        print("problem: " + line)
    for line in fixed:
        print("fixed:   " + line)
    if not problems:
        print("healthy: %d file(s)" % len(nodes))
    elif not fix:
        print("%d problem(s); mend what can be mended with: repair"
              % len(problems))
    return 1 if problems and not fix else 0


# ------------------------------------- schedule, trigger, pause, cancel
def _sched_file(root):
    return Path(root) / "schedules.json"


def _sched_load(root):
    f = _sched_file(root)
    if not f.is_file():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8-sig"))
    except ValueError:
        die("schedules.json is not readable JSON")


def _sched_save(root, data):
    _sched_file(root).write_text(json.dumps(data, ensure_ascii=False,
                                            indent=2), encoding="utf-8")


def _every_seconds(spec):
    m = re.fullmatch(r"(\d+)\s*([mhd])", spec.strip())
    if not m:
        die("--every is a number with m, h or d, such as 30m, 6h, 1d")
    return int(m.group(1)) * {"m": 60, "h": 3600, "d": 86400}[m.group(2)]


def _due(job, now):
    if job.get("paused"):
        return False
    last = job.get("last_run_ts", 0)
    if job["kind"] == "every":
        return now - last >= job["seconds"]
    if job["kind"] == "at":
        hh, mm = map(int, job["at"].split(":"))
        t = time.localtime(now)
        today = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, hh, mm, 0, 0, 0,
                             -1))
        return now >= today and last < today
    return False


def _change_digest(st, ids):
    return hashlib.sha256("".join(st.nodes[i].content_key() if i in st.nodes
                                  else i for i in ids).encode()).hexdigest()


def cmd_schedule(a):
    """Runs of a path at set times (--every 6h, --at 09:00) or when files
    change (trigger). 'schedule --run-due' runs what is due: call it from
    the system's scheduler (cron, Task Scheduler) every few minutes."""
    root = Path(a.project)
    jobs = _sched_load(root)
    if a.run_due:
        st = Store(a.project)
        now, ran = time.time(), 0
        for name, job in sorted(jobs.items()):
            if job["kind"] == "change":
                if job.get("paused"):
                    continue
                digest = _change_digest(st, job["files"])
                if digest == job.get("seen"):
                    continue
                job["seen"] = digest
            elif not _due(job, now):
                continue
            ran += 1
            text = job["input"]
            if job.get("input_file"):
                try:
                    text = (root / job["input_file"]).read_text(
                        encoding="utf-8")
                except OSError as e:
                    job.update(last_status="failed: %s" % e)
                    continue
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err):
                    flow = st.load_flow(job["path"])
                    run_flow(st, job["path"], text, job["backend"],
                             job.get("allow_code", False), flow,
                             model_override=job.get("model", ""),
                             approve="deny")
                status = "done"
            except (Die, FlowError) as e:
                status = "failed: %s" % getattr(e, "msg", e)
            except Exception as e:
                status = "failed: %s: %s" % (type(e).__name__, e)
            job.update(last_run=time.strftime("%Y-%m-%d %H:%M:%S"),
                       last_run_ts=now, last_status=status[:300])
            print("%s: %s" % (name, status[:200]))
        _sched_save(root, jobs)
        if not ran:
            print("nothing due")
        return
    if a.name and (a.every or a.at):
        st = Store(a.project)
        st.load_flow(a.path) if a.path else die("give --path NAME")
        if a.name in jobs:
            die("%s exists; cancel it first" % a.name)
        job = {"path": a.path, "input": a.input or "",
               "input_file": a.input_file or "", "backend": a.backend,
               "model": a.model or "", "allow_code": a.allow_code,
               "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "by": who()}
        if a.every:
            job.update(kind="every", seconds=_every_seconds(a.every),
                       every=a.every)
        else:
            if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", a.at):
                die("--at is HH:MM")
            job.update(kind="at", at=a.at)
        jobs[a.name] = job
        _sched_save(root, jobs)
        print("scheduled %s: path %s %s" % (a.name, a.path,
                                            "every " + a.every if a.every
                                            else "daily at " + a.at))
        print("for it to run, the system must call this every few minutes:\n"
              "  cron:  */5 * * * * cd %s && %s %s schedule --run-due\n"
              "  Windows Task Scheduler: the same command, repeated every 5 "
              "minutes" % (root.resolve(), sys.executable,
                           Path(sys.argv[0]).resolve()))
        return
    if a.name:
        die("give --every or --at (or use trigger for changes)")
    for name, job in sorted(jobs.items()):
        when = {"every": "every %s" % job.get("every"),
                "at": "daily at %s" % job.get("at"),
                "change": "when %s change" % ", ".join(job.get("files", []))
                }[job["kind"]]
        print("%-16s %-8s path %-14s %s; last: %s %s" % (
            name, "paused" if job.get("paused") else "active", job["path"],
            when, job.get("last_run", "never"), job.get("last_status", "")))
    if not jobs:
        print("nothing scheduled")


def cmd_trigger(a):
    """Run a path when any of the given files changes (checked by
    schedule --run-due)."""
    st = Store(a.project)
    jobs = _sched_load(st.root)
    if a.name in jobs:
        die("%s exists; cancel it first" % a.name)
    st.load_flow(a.path)
    ids = [st.resolve(r) for r in a.on_change.split(",") if r.strip()]
    jobs[a.name] = {"kind": "change", "path": a.path, "files": ids,
                    "seen": _change_digest(st, ids), "input": a.input or "",
                    "input_file": a.input_file or "", "backend": a.backend,
                    "model": a.model or "", "allow_code": a.allow_code,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "by": who()}
    _sched_save(st.root, jobs)
    print("trigger %s: path %s runs when %s change%s (checked by schedule "
          "--run-due)" % (a.name, a.path, ", ".join(
              st.nodes[i].title for i in ids), "s" if len(ids) == 1 else ""))


def cmd_pause(a):
    _sched_flag(a, "pause")


def cmd_resume(a):
    _sched_flag(a, "resume")


def cmd_cancel(a):
    _sched_flag(a, "cancel")


def _sched_flag(a, what):
    root = Path(a.project)
    jobs = _sched_load(root)
    if a.name not in jobs:
        die("no schedule or trigger named %s" % a.name)
    if what == "cancel":
        del jobs[a.name]
    else:
        jobs[a.name]["paused"] = what == "pause"
    _sched_save(root, jobs)
    print("%s: %s" % (a.name, {"pause": "paused", "resume": "active again",
                              "cancel": "cancelled"}[what]))



# ---------------------------------------------------------------- providers
def _save_providers(root, data):
    f = providers_file(root)
    f.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")


def _provider_line(name, cfg, data):
    kind = cfg.get("kind", name)
    env = cfg.get("api_key_env") or {"openai": "TAWJIH_API_KEY",
                                     "anthropic": "ANTHROPIC_API_KEY"}.get(kind)
    bits = [name + (" (default)" if data.get("default") == name else ""),
            "kind " + kind]
    if cfg.get("base_url"):
        bits.append(cfg["base_url"])
    if cfg.get("model"):
        bits.append("model " + cfg["model"])
    if cfg.get("num_ctx"):
        bits.append("num_ctx %s" % cfg["num_ctx"])
    if cfg.get("options"):
        bits.append("options " + json.dumps(cfg["options"],
                                            ensure_ascii=False))
    if env:
        bits.append("key in %s: %s" % (env, "set" if os.environ.get(env)
                                       else "NOT set"))
    pol = dict(DEFAULT_POLICY, **{k: cfg[k] for k in DEFAULT_POLICY
                                  if k in cfg})
    bits.append("retries %s, %s backoff %ss, timeout %ss" % (
        pol["retries"], pol["backoff"], pol["backoff_seconds"],
        pol["timeout"]))
    if data["fallback"].get(name):
        bits.append("then " + " -> ".join(data["fallback"][name]))
    return "  ".join(bits)


def cmd_provider(a):
    """Named providers in providers.json: kind, address, model, the name of
    the environment variable holding the key (never the key), the retry
    policy, the default and fallback chains; test one with a real call."""
    st = Store(a.project)
    data = load_providers(st.root)
    provs = data["providers"]
    if a.action == "list":
        if not provs and not data["fallback"]:
            print("no providers.json; built in: %s (default: %s)" % (
                ", ".join(sorted(BACKENDS)), data["default"] or "dry"))
            return 0
        for name in sorted(set(provs) | set(data["fallback"])):
            print(_provider_line(name, provs.get(name) or {"kind": name},
                                 data))
        if not data["default"]:
            print("default: dry (set one with: provider default NAME)")
        return 0
    if not a.name:
        die("provider %s needs a NAME" % a.action)
    name = a.name
    if a.action == "test" and name == "all":
        names = sorted(provs) or [data["default"] or "dry"]
        bad = 0
        for x in names:
            a.name = x
            bad += cmd_provider(a)
        return 1 if bad else 0
    if a.action in ("add", "set", "policy"):
        existed = name in provs
        if a.action == "policy" and name not in provs:
            die("%s is not in providers.json; add it first" % name)
        if name in BACKENDS and a.kind and a.kind != name:
            die("%s is a built-in kind; give this provider another name"
                % name)
        cfg = dict(provs.get(name) or {})
        kind = a.kind or cfg.get("kind") or (name if name in BACKENDS
                                             else None)
        if kind not in BACKENDS:
            die("give --kind (%s)" % ", ".join(sorted(BACKENDS)))
        cfg["kind"] = kind
        if a.api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*",
                                              a.api_key_env):
            die("--api-key-env takes the NAME of an environment variable "
                "(the key itself is never stored)")
        if a.base_url and not re.match(r"https?://", a.base_url):
            die("--base-url must start with http:// or https://")
        for key, val in (("base_url", a.base_url), ("model", a.model),
                         ("api_key_env", a.api_key_env),
                         ("timeout", a.timeout), ("retries", a.retries),
                         ("backoff", a.backoff),
                         ("backoff_seconds", a.backoff_seconds)):
            if val is not None:
                cfg[key] = val
        if a.retry_status is not None:
            try:
                cfg["retry_status"] = sorted({int(x) for x in
                                              a.retry_status.split(",") if x})
            except ValueError:
                die("--retry-status takes numbers: 429,500,502,503,504")
        if (a.retries or 0) < 0 or (a.timeout is not None and a.timeout <= 0):
            die("retries must be 0 or more, and timeout more than 0")
        if a.num_ctx is not None:
            if a.num_ctx < 0:
                die("--num-ctx takes a number of tokens (0 removes it)")
            if a.num_ctx:
                cfg["num_ctx"] = a.num_ctx
            else:
                cfg.pop("num_ctx", None)
        if a.option:
            opts = dict(cfg.get("options") or {})
            opts.update(key_values(a.option, "--option"))
            for k in [k for k, v in opts.items() if v is None]:
                del opts[k]
            if opts:
                cfg["options"] = opts
            else:
                cfg.pop("options", None)
        if any(re.search(r"(?i)api.?key|secret|passw|auth|bearer|"
                         r"^(access_|refresh_)?token$", k)
               for k in cfg.get("options") or {}):
            die("an option that looks like a key or password is not kept in "
                "providers.json; use --api-key-env")
        provs[name] = cfg
        if not data["default"] and a.action == "add":
            data["default"] = name
        _save_providers(st.root, data)
        print(("saved " if existed else "added ") +
              _provider_line(name, cfg, data))
        return 0
    if a.action == "rm":
        if name not in provs:
            die("%s is not in providers.json" % name)
        used = [k for k, v in data["fallback"].items() if name in v]
        if used and not a.yes:
            die("%s is a fallback of %s; add --yes to remove it from those "
                "chains too" % (name, ", ".join(used)))
        del provs[name]
        data["fallback"].pop(name, None)
        for k in used:
            data["fallback"][k] = [x for x in data["fallback"][k] if x != name]
        if data["default"] == name:
            data["default"] = ""
        _save_providers(st.root, data)
        print("removed %s" % name)
        return 0
    known = set(provs) | set(BACKENDS)
    if a.action == "default":
        if name not in known:
            die("unknown provider '%s'" % name)
        data["default"] = name
        _save_providers(st.root, data)
        print("default provider: %s" % name)
        return 0
    if a.action == "fallback":
        others = list(a.others or [])
        for x in [name] + others:
            if x not in known:
                die("unknown provider '%s'" % x)
        if name in others or len(set(others)) != len(others):
            die("a chain cannot repeat a provider")
        if others:
            data["fallback"][name] = others
            print("%s, then %s" % (name, " -> ".join(others)))
        else:
            data["fallback"].pop(name, None)
            print("%s has no fallback now" % name)
        _save_providers(st.root, data)
        return 0
    if a.action == "test":
        prompt = ('Answer with exactly this JSON and nothing else: '
                  '{"status": "ok"}')
        t0 = time.time()
        try:
            answer, usage = call_model(st.root, name, prompt, a.model or "",
                                       fallback=False)
            got = answer.strip().strip("`").strip()
            if got.lower().startswith("json"):
                got = got[4:].strip()
            try:
                ok = json.loads(got).get("status") == "ok"
            except (ValueError, AttributeError):
                ok = got.lower().rstrip(".") == "ok"
            err = "" if ok else "unexpected answer: " + answer[:200]
        except ProviderError as e:
            ok, usage, err = False, {"attempts": getattr(e, "attempts", 0)
                                     }, str(e)
        except Die as e:              # a setting is missing: no call made
            ok, usage, err = False, {}, e.msg
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "provider": name,
               "model": a.model or (provs.get(name) or {}).get("model", ""),
               "ok": ok, "error": err,
               "seconds": round(time.time() - t0, 3),
               "attempts": usage.get("attempts", 0)}
        (st.root / "runs").mkdir(exist_ok=True)
        with open(st.root / "runs" / "provider-tests.jsonl", "a",
                  encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if a.json:
            print(json.dumps(rec, ensure_ascii=False))
        else:
            print("%s: %s (%.1f s%s)" % (
                name, "ok" if ok else "FAILED", rec["seconds"],
                ", " + err if err else ""))
        return 0 if ok else 1
    die("unknown action %s" % a.action)


# ------------------------------------------------------------------- policy
def cmd_policy(a):
    """tawjih.policy.json: limits the project sets for itself. It only
    restricts; what it allows still needs the usual options."""
    st = Store(a.project)
    f = st.root / POLICY_FILE
    if a.action == "show":
        pol = load_policy(st.root)
        print("%s%s" % (POLICY_FILE, "" if f.is_file() else
                        " (none; these are the defaults)"))
        for k in POLICY_KEYS:
            print("  %-15s %s" % (k, json.dumps(pol[k])))
        return 0
    if a.action == "reset":
        if not f.is_file():
            print("no %s" % POLICY_FILE)
            return 0
        if not a.yes:
            die("reset removes the project's limits; add --yes")
        print("it was: " + " ".join(f.read_text(encoding="utf-8-sig").split()))
        f.unlink()
        print("removed %s; the project has no limits of its own now"
              % POLICY_FILE)
        return 0
    data = json.loads(f.read_text(encoding="utf-8-sig")) if f.is_file() else {}
    load_policy(st.root)                       # refuse a broken file
    given = {"allow_code": a.allow_code, "allow_network": a.allow_network,
             "sandbox": a.sandbox, "human_approval": a.human_approval}
    given = {k: v for k, v in given.items() if v is not None}
    if not given:
        die("give at least one of --allow-code, --allow-network, --sandbox, "
            "--human-approval")
    conv = {"true": True, "false": False}
    for k, v in given.items():
        v = conv.get(v, v)
        if v not in POLICY_KEYS[k]:
            die("%s cannot be %s" % (k, v))
        looser = _looser(k, data.get(k, POLICY_DEFAULT[k]), v)
        if looser and not a.yes:
            die("%s=%s loosens the project's policy; add --yes" % (k, v))
        data[k] = v
    f.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for k in POLICY_KEYS:
        if k in data:
            print("  %-15s %s" % (k, json.dumps(data[k])))
    return 0


_STRICT = {"allow_code": [True, False], "sandbox": [False, True],
           "allow_network": [True, "local", False],
           "human_approval": ["never", "run", "always"]}


def _looser(key, old, new):
    order = _STRICT[key]
    return order.index(new) < order.index(old)

# ----------------------------------------------------------------- whoami
def _identity_field(k, v):
    """A checked value for an identity field, or the reason it is refused."""
    v = (v or "").strip()
    if k == "email" and v and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+",
                                               v):
        return None, "%s does not look like an email address" % v
    if k == "id" and v and not re.fullmatch(r"[\w.\-:]{1,64}", v):
        return None, ("an id is letters, digits and . - _ : (an ORCID such "
                      "as 0000-0002-1825-0097 fits)")
    if k == "name" and len(v) > 100:
        return None, "a name of at most 100 characters"
    return v, None


def _save_identity(me):
    me.setdefault("record", "name")
    if not me.get("id"):
        me["id"] = "u-" + os.urandom(4).hex()
    me.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    f = identity_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(me, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")
    try:
        os.chmod(f, 0o600)
    except OSError:
        pass
    return me


def setup_identity(ask=input, out=None):
    """The questions asked before a person's first use on a terminal: name
    (required; git's user.name is offered), email and id (optional), and
    what records note. Returns the identity saved."""
    out = out or sys.stderr
    me = load_identity()
    offer = me.get("name") or _git_name()
    print("\nWelcome to Tawjih. Before you start, who are you?\n"
          "Your answers stay on this computer, in %s, never in a project.\n"
          "Records of runs, approvals and reviews note your id and name.\n"
          % identity_file(), file=out)

    def question(prompt, k, required=False, default=""):
        while True:
            out.flush()
            v = ask(prompt + (" [%s]" % default if default else "") + ": ")
            v = v.strip() or default
            if required and not v:
                print("  a name is needed", file=out)
                continue
            v, why = _identity_field(k, v)
            if why:
                print("  " + why, file=out)
                continue
            return v

    me["name"] = question("Name", "name", True, offer)
    email = question("Email (optional, Enter to skip)", "email", False,
                     me.get("email", ""))
    if email:
        me["email"] = email
    pid = question("Id (optional, such as an ORCID; Enter for one made "
                   "for you)", "id", False, me.get("id", ""))
    if pid:
        me["id"] = pid
    while True:
        rec = ask("What should records note: name (id and name), id, "
                  "email, none [%s]: " % me.get("record", "name")).strip() \
            or me.get("record", "name")
        if rec in RECORD_WHO:
            me["record"] = rec
            break
        print("  one of: " + ", ".join(RECORD_WHO), file=out)
    me.pop("from", None)
    me = _save_identity(me)
    print("\nYour id is %s. Change any of this with `tawjih whoami setup` "
          "or `tawjih whoami set`.\n" % me["id"], file=out)
    return me


def cmd_whoami(a):
    """Who this person is in records: an id, a name and an email kept in
    the person's own settings, not in the project. Records note as much of
    it as --record allows: name (the id and name, the default), id (the id
    only), email (also the email) or none (nothing)."""
    f = identity_file()
    me = load_identity()
    if a.action == "setup":
        try:
            me = setup_identity()
        except (EOFError, KeyboardInterrupt):
            die("nothing was changed")
    elif a.action == "set":
        if not (me or a.name or a.id or a.record):
            die("give at least --name NAME (and --email E, --id ID if you "
                "like), or --record none to note no one")
        for k in ("name", "email", "id"):
            v = getattr(a, k)
            if v is None:
                continue
            v, why = _identity_field(k, v)
            if why:
                die(why)
            me[k] = v
        if a.record:
            me["record"] = a.record
        me.pop("from", None)
        _save_identity(me)
    elif a.action == "clear":
        if f.is_file():
            f.unlink()
        print("identity cleared; the next record makes a new id (to keep "
              "names out of records instead: whoami set --record none)")
        return 0
    else:
        me = me or first_identity()
    by = who()
    if a.json:
        print(json.dumps({"identity": me, "file": str(f), "records": by},
                         ensure_ascii=False, indent=2))
        return 0
    if me:
        print("id      %s%s" % (me.get("id", ""), "  (made %s)" % me["created"]
                                 if me.get("created") else ""))
        print("name    %s" % (me.get("name") or "-"))
        print("email   %s" % (me.get("email") or "-"))
        print("record  %s" % me.get("record", "name"))
        print("file    %s" % f)
    else:
        print("no identity set (tawjih whoami set --name NAME [--email E] "
              "[--id ID])")
    print("records note: %s" % (json.dumps(by, ensure_ascii=False)
                                if by else "nothing"))
    return 0


__all__ = [
    'cmd_whoami', 'setup_identity',
    '_looser', '_provider_line', '_save_providers', 'cmd_policy', 'cmd_provider',
    'INHERITABLE', 'RELEASE_EXTRA', 'RELEASE_KEEP', '_change_digest',
    '_digest_map', '_due', '_every_seconds', '_formatted', '_overall',
    '_project_entries', '_release_has', '_release_read', '_sched_file',
    '_sched_flag', '_sched_load', '_sched_save', '_staged', '_targets',
    '_template_dir', '_trace_file', 'apply_side', 'cmd_alias',
    'cmd_archive', 'cmd_cancel', 'cmd_compact', 'cmd_doctor',
    'cmd_format', 'cmd_gc', 'cmd_glossary', 'cmd_inherit', 'cmd_lock',
    'cmd_mount', 'cmd_pause', 'cmd_release', 'cmd_resume', 'cmd_review',
    'cmd_sandbox', 'cmd_scaffold', 'cmd_schedule', 'cmd_seal',
    'cmd_snapshot', 'cmd_stage', 'cmd_template', 'cmd_trace',
    'cmd_trigger', 'cmd_unalias', 'cmd_unlock', 'cmd_unmount',
    'cmd_verify', 'make_release',
]
