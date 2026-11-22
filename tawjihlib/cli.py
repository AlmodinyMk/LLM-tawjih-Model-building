"""The command line: argument parsing and main().

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

from .core import (
    Die, die, DESCRIPTION, EXTRA, SECTIONS, SPLITS, __version__,
)
from .providers import (
    POLICY_FILE, backend_kind, default_backend, load_policy,
    load_providers, provider_chain,
)
from .engine import (
    CODE_TIMEOUT, SANDBOX_MEMORY_MB,
)
from .commands import (
    GRAPH_FORMATS, cmd_cache, cmd_check, cmd_confirm, cmd_contract,
    cmd_edit, cmd_generate, cmd_graph, cmd_import, cmd_init, cmd_link, cmd_migrate,
    cmd_new, cmd_open, cmd_path, cmd_produce, cmd_run, cmd_show,
    cmd_tangle, cmd_tree, cmd_trees,
)
from .library import (
    cmd_accept, cmd_deposit, cmd_filter, cmd_library, cmd_read,
)
from .manage import (
    cmd_audit, cmd_clean, cmd_copy, cmd_diff, cmd_log, cmd_merge,
    cmd_move, cmd_refs, cmd_rename, cmd_revert, cmd_rm, cmd_split,
    cmd_status, cmd_sync, cmd_tag, cmd_untag,
)
from .outputs import (
    EMBEDDERS, cmd_export, cmd_index, cmd_ingest, cmd_pack, cmd_redact,
    cmd_search, cmd_serve, cmd_stats, cmd_unpack,
)
from .evaluate import (
    cmd_dataset, cmd_eval,
)
from .governance import (
    INHERITABLE, cmd_alias, cmd_archive, cmd_cancel, cmd_compact,
    cmd_doctor, cmd_format, cmd_gc, cmd_glossary, cmd_inherit, cmd_lock,
    cmd_mount, cmd_pause, cmd_release, cmd_resume, cmd_review,
    cmd_sandbox, cmd_scaffold, cmd_schedule, cmd_seal, cmd_snapshot,
    cmd_stage, cmd_template, cmd_trace, cmd_trigger, cmd_unalias,
    cmd_unlock, cmd_unmount, cmd_verify, cmd_provider, cmd_policy,
)
from .script import (
    translate_script,
)
from .home import TOPICS, cmd_home
from .mcpserver import (
    APPROVAL_MODES, Server, ToolError, serve_stdio,
)


def main():
    for s in (sys.stdout, sys.stderr, sys.stdin):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    a = build_parser().parse_args()
    resolve_backend(a)
    try:
        rc = a.fn(a)
    except BrokenPipeError:          # output piped into head and closed
        try:
            sys.stdout = open(os.devnull, "w")
        except OSError:
            pass
        sys.exit(0)
    if isinstance(rc, int) and rc:
        sys.exit(rc)


def resolve_backend(a):
    """--backend names a provider: a built-in kind or one from
    providers.json; without it, providers.json's default, else dry."""
    root = getattr(a, "project", ".")
    for attr in ("backend", "judge_backend"):
        if not hasattr(a, attr):
            continue
        name = getattr(a, attr)
        if name is None and attr == "backend":
            name = default_backend(root)
        if name is not None:
            provider_chain(root, name, False)      # dies if unknown
        setattr(a, attr, name)
    if getattr(a, "allow_code", False) is True and a.fn is not cmd_policy:
        if not load_policy(root)["allow_code"]:
            die("the project's policy (%s) does not allow --allow-code"
                % POLICY_FILE)
    return a


def cmd_do(a):
    """Management statements (CREATE, EDIT, MOVE, COPY, MERGE, DELETE,
    CHECK, GC, PATH; in Arabic too), translated into commands. Without
    --run it only shows them; DELETE, GC and MERGE also need --yes to
    remove anything, and EDIT stages its proposal unless --yes."""
    text = " ".join(a.statements)
    if a.file:
        text = Path(a.file).read_text(encoding="utf-8-sig") + "\n" + text
    if not text.strip():
        die("give statements, or --file SCRIPT")
    todo, problems = translate_script(
        text, a.backend, a.yes, tuple(load_providers(a.project)["providers"]))
    for line in problems:
        print("problem: " + line, file=sys.stderr)
    if problems:
        die("nothing was run: %d statement(s) not understood" % len(problems))
    parser = build_parser()
    for k, (stmt, argv, destructive) in enumerate(todo, 1):
        print("%d. %s\n   -> tawjih.py %s%s" % (
            k, stmt, " ".join(shlex.quote(x) for x in argv),
            "   (needs --yes)" if destructive and not a.yes else ""))
    if not a.run:
        print("nothing was run; add --run to run these in order")
        return 0
    for k, (stmt, argv, destructive) in enumerate(todo, 1):
        if argv[0] == "merge" and not a.yes:
            print("%d. skipped (MERGE removes a file; add --yes)" % k)
            continue
        print("== %d. %s" % (k, stmt))
        sub = resolve_backend(parser.parse_args(["-p", a.project] + argv))
        try:
            rc = sub.fn(sub)
        except Die:                     # its message is already printed
            die("stopped at statement %d (%s)" % (k, stmt))
        except SystemExit as e:         # a command that exits by itself
            rc = e.code or 0
        if isinstance(rc, int) and rc:
            die("stopped at statement %d (%s): exit status %d" % (k, stmt,
                                                                   rc))
    return 0


def cmd_mcp(a):
    """An MCP server for AI assistants, over standard input and output,
    on this project only. Reading always; --allow-write for new files,
    contracts, links, moves, paths and proposals (which a person applies
    with stage --apply); --allow-run to run paths with the provider fixed
    here; --allow-code to let them run code, always in the sandbox."""
    if backend_kind(a.project, a.backend) == "manual":
        die("the manual provider reads the terminal, which the protocol uses")
    if a.allow_code and not a.allow_run:
        die("--allow-code needs --allow-run")
    least = load_policy(a.project)["human_approval"]
    rank = ["never", "run", "always"]
    if rank.index(a.human_approval) < rank.index(least):
        die("the project's policy (%s) asks for --human-approval %s at "
            "least" % (POLICY_FILE, least))
    server = Server(a.project, build_parser, a.allow_write, a.allow_run,
                    a.allow_code, a.backend, a.model or "", a.human_approval)
    if a.human_approval != "always" and (a.allow_write or a.allow_run):
        print("warning: %s without a person's approval: text in the "
              "project's files could lead the assistant to act on it" % (
                  "writes" if a.human_approval == "run" else
                  "writes and runs"), file=sys.stderr)
    # the protocol gets its own copy of standard output; anything else that
    # writes to standard output (a library, a child program) goes to stderr
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8",
                      newline="\n")
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    print("tawjih MCP server on %s: %s" % (
        Path(a.project).resolve(), ", ".join(sorted(server.allow))),
        file=sys.stderr)
    serve_stdio(server, sys.stdin, proto)
    return 0


def cmd_approve(a):
    """Requests an AI assistant made through the MCP server: without IDs,
    list those waiting; with IDs, show them, and run them with --yes or
    decline them with --no. A request runs with the provider, model and
    code setting the server had when it was made."""
    rdir = Path(a.project) / "requests"
    reqs = {f.stem: f for f in sorted(rdir.glob("r*.json"))} \
        if rdir.is_dir() else {}
    if not a.ids:
        waiting = [(k, json.loads(f.read_text(encoding="utf-8-sig")))
                   for k, f in reqs.items()]
        waiting = [(k, r) for k, r in waiting if r["status"] == "pending"]
        for k, r in waiting:
            print("%s  %s  %s" % (k, r["created"], r["tool"]))
        print("%d request(s) waiting%s" % (len(waiting), "; read one with: "
                                           "approve ID" if waiting else ""))
        return 0
    if a.yes and a.no:
        die("--yes or --no, not both")
    for rid in a.ids:
        if rid not in reqs:
            die("no request %s" % rid)
        f = reqs[rid]
        r = json.loads(f.read_text(encoding="utf-8-sig"))
        print("== %s (%s)\n%s" % (rid, r["status"], r["summary"]))
        if r["status"] != "pending":
            continue
        if a.no:
            r["status"] = "declined"
        elif a.yes:
            server = Server(a.project, build_parser, True, True,
                            r["allow_code"], r["backend"], r["model"],
                            "never")
            try:
                text, code = server.call(r["tool"], r["args"])
            except ToolError as e:
                text, code = str(e), 1
            print(text.rstrip())
            r.update(status="done" if not code else "failed", result=text,
                     decided=time.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            print("run it with: approve %s --yes   (or decline: --no)" % rid)
            continue
        r.setdefault("decided", time.strftime("%Y-%m-%d %H:%M:%S"))
        f.write_text(json.dumps(r, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        print("-> %s" % r["status"])
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="tawjih.py", description=DESCRIPTION)
    p.add_argument("-p", "--project", default=".", help="project folder")
    p.add_argument("--version", action="version",
                   version="tawjih " + __version__)
    sub = p.add_subparsers(dest="cmd")
    p.set_defaults(fn=cmd_home)

    s = sub.add_parser("home", help="the home screen: the project at a "
                       "glance and the commands; home TOPIC explains one")
    s.add_argument("topic", nargs="?", metavar="TOPIC",
                   help="one of: " + ", ".join(TOPICS))
    s.set_defaults(fn=cmd_home)

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
    s.add_argument("--type", action="append", metavar="TYPE",
                   help="only this face: human, machine or science (بشري، "
                   "آلي، علمي); may be given more than once")
    s.set_defaults(fn=cmd_open)

    s = sub.add_parser("tree", help="show trees or a branch as columns: "
                       "Code, No. on tree lines, Title, Sides (H/I/C/R "
                       "filled, · empty, ! = stale)")
    s.add_argument("ref", nargs="?", help="a branch to display (its number "
                   "or id)")
    s.add_argument("--branch", help="the same, as an option")
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
    s.add_argument("--backend", default=None, metavar="PROVIDER")
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
    s.add_argument("--sandbox", action="store_true",
                   help="run code steps in a separate, limited process")
    s.add_argument("--code-timeout", type=float, default=CODE_TIMEOUT,
                   metavar="SECONDS")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("eval", help="run a path on test cases with known "
                       "answers; or init, extract, synthesize, compare, "
                       "diff, baseline, blame, promote, report")
    s.add_argument("args", nargs="+", metavar="[ACTION] NAME",
                   help="a path name, or an action and a path name")
    s.add_argument("--direct", action="store_true", help="ask the model "
                   "each case directly, without the path (for compare)")
    s.add_argument("--judge", metavar="MODEL", help="grade cases with "
                   "match judge by asking this model")
    s.add_argument("--judge-backend", metavar="PROVIDER",
                   help="provider of the judge (default: --backend)")
    s.add_argument("--fail-fast", action="store_true",
                   help="stop at the first failed case and show it")
    s.add_argument("--branch", help="extract, synthesize: only this branch")
    s.add_argument("--split", choices=SPLITS, help="extract: which files "
                   "(default eval)")
    s.add_argument("--count", type=int, default=10,
                   help="synthesize: how many cases to propose")
    s.add_argument("--accept", action="store_true",
                   help="synthesize: add the proposed cases")
    s.add_argument("--save", action="store_true",
                   help="baseline: save the latest evaluation as baseline")
    s.add_argument("--baseline", action="store_true",
                   help="diff: compare with the baseline")
    s.add_argument("--case", help="blame, promote: only this case")
    s.add_argument("--to", metavar="REF", help="promote: the file whose "
                   "examples get the cases")
    s.add_argument("-o", "--out", help="report: output file")
    s.add_argument("--cases", help="JSONL file (default: "
                   "paths/NAME.cases.jsonl)")
    s.add_argument("--backend", default=None, metavar="PROVIDER")
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
    s.add_argument("--sandbox", action="store_true",
                   help="run code steps in a separate, limited process")
    s.add_argument("--code-timeout", type=float, default=CODE_TIMEOUT,
                   metavar="SECONDS")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("dataset", help="question-answer pairs as JSONL for "
                       "fine-tuning tools")
    s.add_argument("-o", "--out", required=True)
    s.add_argument("--source", choices=["eval", "examples", "runs", "steps"],
                   default="eval", help="passed evaluations (default), the "
                   "examples sides, finished runs, or each model step")
    s.add_argument("--branch", help="examples: only this branch")
    s.add_argument("--split", choices=SPLITS, help="examples: which files "
                   "(default train)")
    s.add_argument("--with-rules", action="store_true", help="examples: put "
                   "the file's instructions before each question")
    s.add_argument("--repeat-by-weight", action="store_true",
                   help="examples: repeat each by its file's weight")
    s.add_argument("--dpo", metavar="FILE", help="examples: also write "
                   "preference pairs from BAD: lines")
    s.add_argument("--path", help="only this path")
    s.add_argument("--step", help="with --source steps: only this file id")
    s.add_argument("--include-failed", action="store_true",
                   help="with --source eval: failed cases too")
    s.add_argument("--format", choices=["chat", "prompt"], default="chat")
    s.add_argument("--meta", action="store_true",
                   help="add where each example came from")
    s.set_defaults(fn=cmd_dataset)

    s = sub.add_parser("show", help="print a file's three faces (human, "
                       "machine, science) under its title, code and number")
    s.add_argument("ref")
    s.add_argument("--type", action="append", metavar="TYPE",
                   help="only this face: human, machine or science (بشري، "
                   "آلي، علمي); may be given more than once")
    s.add_argument("--raw", action="store_true",
                   help="the whole file as one text, with its header")
    s.set_defaults(fn=cmd_show)

    s = sub.add_parser("migrate", help="turn files made before 0.15 (one "
                       ".md each) into folders of faces")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_migrate)

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
    s.add_argument("--expects-schema", metavar="FILE", help="a JSON Schema "
                   "for the input: copied to schemas/ and used as "
                   "schema(NAME)")
    s.add_argument("--produces-schema", metavar="FILE",
                   help="a JSON Schema for the output")
    s.add_argument("--model", help="the model this file's step uses")
    s.add_argument("--retries", help="0-9: ask the model again when its "
                   "answer breaks the produces contract")
    s.add_argument("--fallback", metavar="REF", help="a file run instead "
                   "when this step fails (same input)")
    s.add_argument("--approve", help="before, after or both: a person "
                   "approves the step")
    s.add_argument("--split", help="train, eval or ignore: where the "
                   "file's examples go")
    s.add_argument("--weight", help="importance of the file's examples in "
                   "training data (a number, default 1)")
    s.add_argument("--each", help="yes, or the key of a list in the input: "
                   "the step runs once per item (FOR EACH)")
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
    s.add_argument("--examples", action="store_true",
                   help="confirm the examples side instead")
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
                   "TO, AND, IF, ELSE, AMONG, THROUGH, LINK, CALL, with EACH, APPROVE, "
                   "USING, RETRY, UNTIL, PICK (see SPEC.md); ',', '/', '؛' or "
                   "'،' between statements")
    s.set_defaults(fn=cmd_path)

    s = sub.add_parser("run", help="execute a path step by step")
    s.add_argument("name")
    s.add_argument("--input", default="")
    s.add_argument("--input-file")
    s.add_argument("--backend", default=None, metavar="PROVIDER")
    s.add_argument("--retries", type=int, choices=range(10), metavar="N",
                   help="ask a model again up to N times when its answer "
                   "breaks the contract (default: each file's retries line)")
    s.add_argument("--approve", choices=["ask", "auto", "deny"],
                   help="steps marked approve: ask on the terminal "
                   "(default on a terminal), auto, or deny (default "
                   "otherwise)")
    s.add_argument("--sandbox", action="store_true",
                   help="run code steps in a separate, limited process")
    s.add_argument("--max-workers", type=int, default=1, metavar="N",
                   help="run up to N steps whose inputs are ready at the "
                   "same time (default 1: one by one)")
    s.add_argument("--on-failure", choices=["stop", "continue"],
                   default="stop", help="continue: leave a failed step "
                   "out and skip what needs it, instead of stopping")
    s.add_argument("--iterate", type=int, metavar="N",
                   help="run the whole path up to N times, each round on "
                   "the result of the one before (like ITERATE N)")
    s.add_argument("--until", metavar="COND", help="with --iterate: stop "
                   "when the result meets this word, NOT word, or "
                   "key=value")
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
                   "numbers) joined by the words of the path language (TO, AND, "
                   "IF, ELSE, AMONG, THROUGH, LINK, CALL ...); ',' or '/' "
                   "between statements; omit to reuse NAME")
    s.add_argument("--plan", action="store_true",
                   help="show and validate the plan without running it")
    s.add_argument("--input", default="")
    s.add_argument("--input-file")
    s.add_argument("--backend", default=None, metavar="PROVIDER")
    s.add_argument("--retries", type=int, choices=range(10), metavar="N",
                   help="ask a model again up to N times when its answer "
                   "breaks the contract (default: each file's retries line)")
    s.add_argument("--approve", choices=["ask", "auto", "deny"],
                   help="steps marked approve: ask on the terminal "
                   "(default on a terminal), auto, or deny (default "
                   "otherwise)")
    s.add_argument("--sandbox", action="store_true",
                   help="run code steps in a separate, limited process")
    s.add_argument("--max-workers", type=int, default=1, metavar="N",
                   help="run up to N steps whose inputs are ready at the "
                   "same time (default 1: one by one)")
    s.add_argument("--on-failure", choices=["stop", "continue"],
                   default="stop", help="continue: leave a failed step "
                   "out and skip what needs it, instead of stopping")
    s.add_argument("--iterate", type=int, metavar="N",
                   help="run the whole path up to N times, each round on "
                   "the result of the one before (like ITERATE N)")
    s.add_argument("--until", metavar="COND", help="with --iterate: stop "
                   "when the result meets this word, NOT word, or "
                   "key=value")
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
    s.add_argument("--backend", default=None, metavar="PROVIDER")
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
    s.add_argument("--side", choices=SECTIONS + EXTRA, required=True)
    s.add_argument("--ask", required=True, help="what the model should do")
    s.add_argument("--source", help="text file given to the model as material")
    s.add_argument("--backend", default=None, metavar="PROVIDER")
    s.add_argument("--model", help="model name (default: the TAWJIH_MODEL "
                   "environment variable)")
    s.add_argument("--yes", action="store_true", help="save without asking")
    s.add_argument("--stage", action="store_true",
                   help="keep the change in stage/ for review instead")
    s.set_defaults(fn=cmd_edit)

    # governance (0.10)
    def refs(s, many=True):
        s.add_argument("refs", nargs="+" if many else "*", metavar="REF")
        s.add_argument("--branch", action="store_true",
                       help="the whole branch under each REF")
    s = sub.add_parser("lock", aliases=["freeze"], help="lock files: they "
                       "cannot change or be removed until unlocked")
    refs(s)
    s.set_defaults(fn=cmd_lock)
    s = sub.add_parser("unlock", help="unlock files")
    refs(s)
    s.set_defaults(fn=cmd_unlock)

    s = sub.add_parser("alias", help="give a file other names usable as REF; "
                       "without arguments, list aliases")
    s.add_argument("ref", nargs="?")
    s.add_argument("names", nargs="*")
    s.set_defaults(fn=cmd_alias)
    s = sub.add_parser("unalias", help="remove aliases of a file")
    s.add_argument("ref")
    s.add_argument("names", nargs="+")
    s.set_defaults(fn=cmd_unalias)

    s = sub.add_parser("review", help="list files not reviewed by a person "
                       "or changed since; --ok marks them reviewed")
    refs(s, many=False)
    s.add_argument("--ok", action="store_true")
    s.set_defaults(fn=cmd_review)

    s = sub.add_parser("stage", help="changes proposed by edit --stage: list, "
                       "--show, --apply, --drop")
    s.add_argument("ref", nargs="?")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--show", action="store_true")
    g.add_argument("--apply", action="store_true")
    g.add_argument("--drop", action="store_true")
    s.set_defaults(fn=cmd_stage)

    s = sub.add_parser("mount", help="use another project's files read-only; "
                       "without arguments, list mounts")
    s.add_argument("where", nargs="?", help="the other project's folder")
    s.add_argument("--name", help="mount name (default: the folder name)")
    s.set_defaults(fn=cmd_mount)
    s = sub.add_parser("unmount", help="stop using a mounted project")
    s.add_argument("name")
    s.set_defaults(fn=cmd_unmount)

    s = sub.add_parser("sandbox", help="run a file's code side in the "
                       "sandbox on an input")
    s.add_argument("ref")
    s.add_argument("--input", default="")
    s.add_argument("--input-file")
    s.add_argument("--timeout", type=float, default=CODE_TIMEOUT)
    s.add_argument("--memory", type=int, default=SANDBOX_MEMORY_MB,
                   help="memory limit in MB (default %d)" % SANDBOX_MEMORY_MB)
    s.set_defaults(fn=cmd_sandbox)

    s = sub.add_parser("trace", help="--impact REF, --output FILE, --cost")
    s.add_argument("--impact", metavar="REF")
    s.add_argument("--output", metavar="FILE")
    s.add_argument("--cost", action="store_true")
    s.add_argument("--path", help="with --cost: only this path")
    s.set_defaults(fn=cmd_trace)

    s = sub.add_parser("release", help="sealed snapshots of the project: "
                       "create, list, diff, verify, rollback")
    s.add_argument("action", choices=["create", "list", "diff", "verify",
                                      "rollback"])
    s.add_argument("names", nargs="*")
    s.add_argument("--note")
    s.add_argument("--yes", action="store_true")
    s.add_argument("--force", action="store_true",
                   help="rollback: even when locked files would change")
    s.set_defaults(fn=cmd_release)

    s = sub.add_parser("snapshot", help="a release named by the time")
    s.add_argument("--note")
    s.set_defaults(fn=cmd_snapshot)

    s = sub.add_parser("seal", help="record the digest of every file in "
                       "seal.json (--sign: also with GnuPG)")
    s.add_argument("--sign", action="store_true")
    s.set_defaults(fn=cmd_seal)
    s = sub.add_parser("verify", help="compare the project with seal.json")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("inherit", help="give the files under REF its model, "
                       "tags, contracts...")
    s.add_argument("ref")
    s.add_argument("--fields", help="comma separated, from: "
                   + ", ".join(INHERITABLE) + " (default model,tags,split,"
                   "weight)")
    s.add_argument("--overwrite", action="store_true",
                   help="also where the files have their own value")
    s.set_defaults(fn=cmd_inherit)

    s = sub.add_parser("template", help="save NAME REF [--branch], list, "
                       "rm NAME")
    s.add_argument("action", choices=["save", "list", "rm"])
    s.add_argument("args", nargs="*")
    s.add_argument("--branch", action="store_true")
    s.set_defaults(fn=cmd_template)
    s = sub.add_parser("scaffold", help="new files from a template")
    s.add_argument("name")
    s.add_argument("--parent")
    s.add_argument("--title", help="title of the first new file")
    s.set_defaults(fn=cmd_scaffold)

    s = sub.add_parser("gc", help="remove old saved versions and cached "
                       "answers (counts only, without --yes)")
    s.add_argument("--keep", type=int, default=10,
                   help="versions kept per file (default 10)")
    s.add_argument("--days", type=int, help="also drop versions older "
                   "than this")
    s.add_argument("--cache-days", type=int,
                   help="drop cached answers older than this")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_gc)

    s = sub.add_parser("compact", help="orders 1, 2, ... and standard form "
                       "for every file")
    s.set_defaults(fn=cmd_compact)

    s = sub.add_parser("format", aliases=["lint"], help="tidy the text of "
                       "files; --check only reports")
    s.add_argument("--branch")
    s.add_argument("--check", action="store_true")
    s.set_defaults(fn=cmd_format)

    s = sub.add_parser("archive", help="move a branch out of the trees; "
                       "--list, --restore ID")
    s.add_argument("ref", nargs="?")
    s.add_argument("--list", action="store_true")
    s.add_argument("--restore", metavar="ID")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_archive)

    s = sub.add_parser("glossary", help="terms defined in the human sides")
    s.add_argument("--branch")
    s.add_argument("--conflicts", action="store_true",
                   help="only terms defined differently in several files")
    s.add_argument("-o", "--out")
    s.set_defaults(fn=cmd_glossary)

    s = sub.add_parser("doctor", help="check the project's files without "
                       "trusting them")
    s.add_argument("--fix", action="store_true")
    s.set_defaults(fn=cmd_doctor)
    s = sub.add_parser("repair", help="doctor --fix")
    s.add_argument("--fix", action="store_true", help=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_doctor)

    def job(s):
        s.add_argument("--path", help="the path to run")
        s.add_argument("--input", default="")
        s.add_argument("--input-file", help="relative to the project")
        s.add_argument("--backend", default=None, metavar="PROVIDER")
        s.add_argument("--model")
        s.add_argument("--allow-code", action="store_true")
    s = sub.add_parser("schedule", help="run a path --every 6h or --at "
                       "HH:MM; list; --run-due runs what is due")
    s.add_argument("name", nargs="?")
    s.add_argument("--every")
    s.add_argument("--at")
    s.add_argument("--run-due", action="store_true")
    job(s)
    s.set_defaults(fn=cmd_schedule)
    s = sub.add_parser("trigger", help="run a path when files change")
    s.add_argument("name")
    s.add_argument("--on-change", required=True, metavar="REF,REF")
    job(s)
    s.set_defaults(fn=cmd_trigger)
    for nm, fn in (("pause", cmd_pause), ("resume", cmd_resume),
                   ("cancel", cmd_cancel)):
        s = sub.add_parser(nm, help="%s a schedule or trigger" % nm)
        s.add_argument("name")
        s.set_defaults(fn=fn)

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
    s.add_argument("--backend", default=None, metavar="PROVIDER")
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

    s = sub.add_parser("mcp", help="an MCP server for AI assistants on "
                       "this project (standard input and output)")
    s.add_argument("--allow-write", action="store_true",
                   help="new files, contracts, links, moves, paths, and "
                   "proposals for sides (applied by a person)")
    s.add_argument("--allow-run", action="store_true",
                   help="run and evaluate paths")
    s.add_argument("--allow-code", action="store_true",
                   help="let runs execute code steps (always sandboxed)")
    s.add_argument("--backend", default=None, metavar="PROVIDER",
                   help="the provider for runs (fixed for the session)")
    s.add_argument("--model", help="the model for runs")
    s.add_argument("--human-approval", choices=APPROVAL_MODES,
                   default="always", help="what a person approves: always "
                   "(writes and runs, the default), run (runs only), never")
    s.set_defaults(fn=cmd_mcp)

    s = sub.add_parser("approve", help="requests from an AI assistant "
                       "(MCP) waiting for a person: list, read, run, decline")
    s.add_argument("ids", nargs="*", metavar="ID")
    s.add_argument("--yes", action="store_true", help="run them")
    s.add_argument("--no", action="store_true", help="decline them")
    s.set_defaults(fn=cmd_approve)

    s = sub.add_parser("do", help="management statements: CREATE, EDIT, "
                       "MOVE, COPY, MERGE, DELETE, CHECK, GC, PATH (Arabic "
                       "too); shown unless --run")
    s.add_argument("statements", nargs="*")
    s.add_argument("--file", help="a script of statements, one per line")
    s.add_argument("--run", action="store_true", help="run them in order")
    s.add_argument("--yes", action="store_true",
                   help="let DELETE, GC and MERGE remove, and EDIT save")
    s.add_argument("--backend", default=None, metavar="PROVIDER",
                   help="for EDIT, unless it says USING a provider")
    s.set_defaults(fn=cmd_do)

    s = sub.add_parser("provider", help="named providers (providers.json): "
                       "address, model, key variable, retries, default, "
                       "fallback chain, and a test call")
    s.add_argument("action", choices=["list", "add", "set", "policy", "rm",
                                      "default", "fallback", "test"])
    s.add_argument("name", nargs="?", help="provider name (test: or all)")
    s.add_argument("others", nargs="*", metavar="FALLBACK",
                   help="fallback: the providers to try after NAME, in "
                   "order (none clears the chain)")
    s.add_argument("--kind", help="dry, manual, ollama, openai (any "
                   "OpenAI-compatible server) or anthropic")
    s.add_argument("--base-url", help="the server's address")
    s.add_argument("--model", help="model (test: this model)")
    s.add_argument("--api-key-env", metavar="VARIABLE", help="the NAME of "
                   "the environment variable holding the key; the key is "
                   "never written to a file")
    s.add_argument("--timeout", type=float, help="seconds per call")
    s.add_argument("--retries", type=int, help="tries after the first, on "
                   "a lost connection or a retry status")
    s.add_argument("--retry-status", metavar="CODES",
                   help="HTTP statuses tried again (429,500,502,503,504)")
    s.add_argument("--backoff", choices=["exponential", "fixed"])
    s.add_argument("--backoff-seconds", type=float)
    s.add_argument("--json", action="store_true", help="test: one JSON line")
    s.add_argument("--yes", action="store_true",
                   help="rm: also remove it from fallback chains")
    s.set_defaults(fn=cmd_provider)

    s = sub.add_parser("policy", help="the project's own limits "
                       "(tawjih.policy.json); it only restricts")
    s.add_argument("action", choices=["show", "set", "reset"])
    s.add_argument("--allow-code", choices=["true", "false"])
    s.add_argument("--allow-network", choices=["true", "false", "local"])
    s.add_argument("--sandbox", choices=["true", "false"])
    s.add_argument("--human-approval", choices=["always", "run", "never"],
                   help="the least approval an MCP server may use")
    s.add_argument("--yes", action="store_true",
                   help="allow loosening a limit, or removing the file")
    s.set_defaults(fn=cmd_policy)
    return p


__all__ = [
    'build_parser', 'cmd_approve', 'cmd_do', 'cmd_mcp', 'main',
    'resolve_backend',
]
