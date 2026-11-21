"""The evaluation family and the dataset compiler.

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import contextlib
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path

from .core import (
    Die, Store, __version__, _json_in, _norm_q, die, parse_examples,
)
from .providers import (
    backend_kind, call_model, use_model,
)
from .engine import (
    _stamped, run_flow,
)
from .commands import (
    _backup, _flow_from_args, strip_fence,
)
from .outputs import (
    _by_time,
)


# --------------------------------------------------------------------- eval
JUDGE_PROMPT = """You are grading an answer against an expected answer.
Reply with PASS or FAIL on the first line, then one sentence giving the
reason. PASS only if the answer agrees with the expected answer on every
point that matters; wording may differ.

# Question
%s

# Expected answer
%s

# Answer to grade
%s"""


def _case_passes(case, output, judge=None):
    how = case.get("match", "json" if isinstance(case.get("expect"), dict)
                   else "contains")
    exp = case.get("expect")
    if how == "judge":
        if judge is None:
            return None, "needs a judge (--judge MODEL)"
        backend, model, root = judge
        try:
            verdict, _ = call_model(root, backend, JUDGE_PROMPT % (
                case["input"], exp if isinstance(exp, str) else json.dumps(
                    exp, ensure_ascii=False), output), model)
        except Exception as e:
            return False, "the judge failed: %s" % e
        first = verdict.strip().split("\n", 1)
        word = first[0].strip().strip("*#: ").upper()
        why = "judge: " + (first[1].strip() if len(first) > 1 else
                           first[0].strip())[:200]
        if word.startswith("PASS"):
            return True, why
        if word.startswith("FAIL"):
            return False, why
        return False, "the judge did not answer PASS or FAIL: " + verdict[:120]
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
    return False, ("unknown match '%s' (contains, not-contains, equals, "
                   "regex, json, judge)" % how)


EVAL_ACTIONS = ("init", "extract", "synthesize", "compare", "diff",
                "baseline", "blame", "promote", "report")


def _cases_file(st, name, given=None):
    return Path(given) if given else st.pdir / (name + ".cases.jsonl")


def _read_cases(cfile):
    cases = []
    for k, line in enumerate(cfile.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip() and not line.lstrip().startswith("#"):
            try:
                c = json.loads(line)
                if "input" not in c or "expect" not in c:
                    raise ValueError("needs input and expect")
            except ValueError as e:
                die("%s line %d: %s" % (cfile, k, e))
            c.setdefault("name", "case %d" % (len(cases) + 1))
            cases.append(c)
    return cases


def _eval_records(st, name, mode=None):
    """Evaluation records of a path, oldest first."""
    out = []
    for f in sorted(st.rdir.glob("eval-%s-*.json" % name), key=_by_time) if \
            st.rdir.is_dir() else []:
        try:
            r = json.loads(f.read_text(encoding="utf-8-sig"))
        except ValueError:
            continue
        if r.get("path") == name and (mode is None or
                                      r.get("mode", "path") == mode):
            r["_file"] = str(f)
            out.append(r)
    return out


def _latest(st, name, mode=None):
    recs = _eval_records(st, name, mode)
    if not recs:
        die("no %sevaluation of %s yet (run: eval %s%s)" % (
            mode + " " if mode else "", name, name,
            " --direct" if mode == "direct" else ""))
    return recs[-1]


def _load_record(st, ref, name):
    if ref in ("baseline",):
        f = st.pdir / (name + ".baseline.json")
        if not f.is_file():
            die("no baseline for %s (save one with: eval baseline %s --save)"
                % (name, name))
        return json.loads(f.read_text(encoding="utf-8-sig"))
    if ref in ("path", "direct"):
        return _latest(st, name, ref)
    f = Path(ref)
    if not f.is_file():
        die("no evaluation record %s (give a file, or path, direct, "
            "baseline)" % ref)
    return json.loads(f.read_text(encoding="utf-8-sig"))


def _changes_between(old, new):
    """Cases that passed before and fail now, and the reverse, by name."""
    before = {c["name"]: c["pass"] for c in old["cases"]}
    worse = [c for c in new["cases"] if before.get(c["name"]) and not c["pass"]]
    better = [c for c in new["cases"]
              if before.get(c["name"]) is False and c["pass"]]
    return worse, better


def cmd_eval(a):
    """Run a path on test cases with known answers and report how many
    pass, or (with a first word) prepare, compare and trace evaluations:
    init, extract, synthesize, compare, diff, baseline, blame, promote,
    report."""
    args = list(a.args)
    if args and args[0] in EVAL_ACTIONS:
        action = args.pop(0)
        return globals()["_eval_" + action](a, args)
    if len(args) != 1:
        die("give one path name (or an action: %s)" % ", ".join(EVAL_ACTIONS))
    a.name = args[0]
    st = Store(a.project)
    flow = None if a.direct else _flow_from_args(st, a.name, [])
    cfile = _cases_file(st, a.name, a.cases)
    if not cfile.is_file():
        die("no test cases: write %s (eval init %s makes a template), one "
            "JSON object per line, such as\n"
            '  {"input": "Is Z6 cyclic?", "expect": ["cyclic", "1", "5"]}'
            % (cfile, a.name))
    cases = _read_cases(cfile)
    if not cases:
        die("%s has no cases" % cfile)
    if flow is not None:
        coded = [i for i in flow.nodes if st.nodes[i].code.strip()]
        if coded and not a.allow_code:
            die("the path has code; pass --allow-code after reading it")
    use_model(a)
    judge = None
    if a.judge:
        judge = (a.judge_backend or a.backend, a.judge, st.root)
    results, passed, skipped = [], 0, 0
    for c in cases:
        err = io.StringIO()
        usage, t0, record = {}, time.monotonic(), ""
        try:
            with contextlib.redirect_stderr(err):
                if flow is None:          # the model asked directly
                    out, usage = call_model(st.root, a.backend, c["input"],
                                            getattr(a, "model", None) or
                                            os.environ.get("TAWJIH_MODEL", ""))
                else:
                    out = run_flow(st, a.name, c["input"], a.backend,
                                   a.allow_code, flow, resume=a.resume,
                                   model_override=getattr(a, "model", None)
                                   or "", code_timeout=a.code_timeout,
                                   retries=a.retries,
                                   approve=a.approve or "deny",
                                   sandbox=getattr(a, "sandbox", False))
            ok, why = _case_passes(c, out, judge)
        except Die as e:
            out, ok, why = "", False, "the run failed: %s" % getattr(
                e, "msg", e.code)
        except Exception as e:                  # a fault in the program
            out, ok, why = "", False, "the run failed: %s: %s" % (
                type(e).__name__, e)
        seconds = round(time.monotonic() - t0, 3)
        lines = err.getvalue().splitlines()
        log = next((l.split(": ", 1)[1] for l in lines
                    if l.startswith("log: ")), "")
        record = next((l.split(": ", 1)[1] for l in lines
                       if l.startswith("record: ")), "")
        if record and not usage:
            try:
                usage = json.loads(Path(record).read_text(
                    encoding="utf-8")).get("usage", {})
            except (OSError, ValueError):
                usage = {}
        if ok is None:
            skipped += 1
        else:
            passed += ok
        results.append({"name": c["name"], "input": c["input"],
                        "expect": c["expect"], "match": c.get("match", ""),
                        "output": out, "pass": ok, "why": why, "log": log,
                        "record": record, "seconds": seconds,
                        "tokens_in": usage.get("in", 0),
                        "tokens_out": usage.get("out", 0)})
        print("%-4s %-28s %s" % ({True: "PASS", False: "FAIL", None: "SKIP"}
                                 [ok], c["name"][:28], why[:90]))
        if a.fail_fast and ok is False:
            print("\n--- input\n%s\n--- expected\n%s\n--- output\n%s" % (
                c["input"], json.dumps(c["expect"], ensure_ascii=False)
                if not isinstance(c["expect"], str) else c["expect"], out))
            if log:
                print("--- log: %s" % log)
            break
    total = len(results) - skipped
    rec = _stamped(st.rdir, "eval-" + a.name, ".json")
    data = {"tawjih": __version__, "path": a.name,
            "mode": "direct" if flow is None else "path",
            "backend": a.backend, "model": os.environ.get("TAWJIH_MODEL", ""),
            "judge": a.judge or "", "passed": passed, "total": total,
            "skipped": skipped, "cases": results}
    rec.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("%d of %d passed (%.0f%%)%s; record: %s" % (
        passed, total, 100 * passed / total if total else 0,
        ", %d skipped" % skipped if skipped else "", rec))
    regress = False
    base = st.pdir / (a.name + ".baseline.json")
    if base.is_file() and flow is not None:
        worse, better = _changes_between(json.loads(base.read_text(
            encoding="utf-8")), data)
        for c in worse:
            print("REGRESSION: %s passed in the baseline and fails now" % c["name"])
        for c in better:
            print("fixed since the baseline: %s" % c["name"])
        regress = bool(worse)
    return 0 if passed == total and not regress else 1


def _eval_init(a, args):
    st = Store(a.project)
    if len(args) != 1:
        die("eval init NAME")
    f = _cases_file(st, args[0], a.cases)
    if f.exists():
        die("%s already exists" % f)
    st.pdir.mkdir(exist_ok=True)
    f.write_text(
        "# One case per line: input, expect, and how to compare (match).\n"
        "# match: contains (default; expect may be a list), not-contains,\n"
        "# equals, regex, json (expect is an object of keys and values),\n"
        "# judge (a model compares meaning; run with --judge MODEL).\n"
        '# {"name": "first", "input": "a question", "expect": ["a word", '
        '"another"]}\n'
        '# {"input": "a question", "expect": {"key": 1}, "match": "json"}\n',
        encoding="utf-8")
    print("written %s; add one case per line, then: eval %s" % (f, args[0]))


def _eval_extract(a, args):
    """Cases from the examples of files marked split: eval (or --split)."""
    st = Store(a.project)
    if len(args) != 1:
        die("eval extract NAME [--branch REF] [--split eval]")
    split = a.split or "eval"
    if split == "train":
        print("warning: cases taken from training files measure memory, "
              "not skill", file=sys.stderr)
    ids = st.subtree_ids(st.resolve(a.branch)) if a.branch else \
        st.subtree_ids("")
    f = _cases_file(st, args[0], a.cases)
    have = set()
    if f.is_file():
        have = {_norm_q(c["input"]) for c in _read_cases(f)}
    new = []
    for i in ids:
        n = st.nodes[i]
        if (n.split or "train") != split:
            continue
        for k, ex in enumerate(parse_examples(n.examples)[0], 1):
            if _norm_q(ex["q"]) in have:
                continue
            have.add(_norm_q(ex["q"]))
            case = {"name": "%s #%d" % (n.title[:20], k), "input": ex["q"],
                    "from": i}
            if ex["e"]:
                case.update(expect=[x.strip() for x in ex["e"].split(";")
                                    if x.strip()])
            else:
                case.update(expect=ex["a"], match="judge")
            new.append(case)
    if not new:
        print("no new cases (files with split: %s and examples)" % split)
        return
    st.pdir.mkdir(exist_ok=True)
    with f.open("a", encoding="utf-8") as fh:
        for c in new:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    judged = sum(1 for c in new if c.get("match") == "judge")
    print("%d case(s) added to %s%s" % (len(new), f, (
        "; %d without an E: line will be graded by a judge (--judge MODEL)"
        % judged) if judged else ""))


def _eval_synthesize(a, args):
    """A model proposes hard and edge cases from the files' human sides and
    examples; they are written for review, not used until --accept."""
    st = Store(a.project)
    if len(args) != 1:
        die("eval synthesize NAME --backend B [--branch REF] [--count N]")
    name = args[0]
    prop = st.pdir / (name + ".cases.proposed.jsonl")
    if a.accept:
        if not prop.is_file():
            die("no proposed cases for %s" % name)
        cases = _read_cases(prop)
        f = _cases_file(st, name, a.cases)
        with f.open("a", encoding="utf-8") as fh:
            for c in cases:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        prop.unlink()
        print("%d case(s) added to %s" % (len(cases), f))
        return
    if backend_kind(st.root, a.backend) == "dry":
        die("choose a --backend for the model that writes the cases")
    ids = st.subtree_ids(st.resolve(a.branch)) if a.branch else (
        list(_flow_from_args(st, name, []).nodes)
        if (st.pdir / (name + ".path")).is_file() else st.subtree_ids(""))
    material = []
    for i in ids:
        n = st.nodes[i]
        if n.split == "eval":
            continue                  # never show the held-out answers
        if n.human.strip() or n.examples.strip():
            material.append("## %s\n%s\n%s" % (n.title, n.human.strip(),
                                               n.examples.strip()))
    if not material:
        die("the files have no human sides or examples to work from")
    prompt = (
        "Write %d test cases for a program that answers questions on the "
        "subject below. Prefer hard cases: edge cases, common mistakes, "
        "questions that need every rule. Reply with JSON Lines only, one "
        'object per line: {"input": "...", "expect": ["short text a correct '
        'answer must contain", ...]}. Write the questions in the language of '
        "the material.\n\n# Material\n%s" % (a.count, "\n\n".join(material)))
    use_model(a)
    answer, _ = call_model(st.root, a.backend, prompt,
                           getattr(a, "model", None) or
                           os.environ.get("TAWJIH_MODEL", ""))
    good, bad = [], 0
    for line in strip_fence(answer).splitlines():
        try:
            c = json.loads(line)
            if not isinstance(c, dict) or "input" not in c or "expect" not in c:
                raise ValueError
            c.setdefault("name", "synthesized %d" % (len(good) + 1))
            c["synthesized"] = True
            good.append(c)
        except ValueError:
            bad += line.strip() != ""
    if not good:
        die("the model wrote no usable cases")
    prop.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n"
                            for c in good), encoding="utf-8")
    print("%d case(s) proposed in %s%s. Read and correct them: a model's "
          "expected answers can be wrong. Then: eval synthesize %s --accept"
          % (len(good), prop, " (%d unreadable line(s) left out)" % bad
             if bad else "", name))


def _summary(r):
    t = r.get("total") or 0
    toks = sum(c.get("tokens_in", 0) + c.get("tokens_out", 0)
               for c in r["cases"])
    secs = sum(c.get("seconds", 0) for c in r["cases"])
    return "%d/%d (%.0f%%), %d tokens, %.1f s" % (
        r.get("passed", 0), t, 100 * r.get("passed", 0) / t if t else 0,
        toks, secs)


def _eval_compare(a, args):
    """Two evaluations case by case: by default the latest run of the path
    against the latest run of the model asked directly (eval NAME
    --direct). Records may also be given as files, or as path, direct or
    baseline."""
    st = Store(a.project)
    if not args:
        die("eval compare NAME [A B]")
    name = args[0]
    ra = _load_record(st, args[1] if len(args) > 1 else "path", name)
    rb = _load_record(st, args[2] if len(args) > 2 else "direct", name)
    la = Path(args[1]).stem[-12:] if len(args) > 1 else "path"
    lb = Path(args[2]).stem[-12:] if len(args) > 2 else "direct"
    print("%-30s %-12s %-12s" % ("case", la, lb))
    other = {c["name"]: c for c in rb["cases"]}
    both = only_a = only_b = 0
    for c in ra["cases"]:
        d = other.get(c["name"])
        mark = {True: "pass", False: "FAIL", None: "skip"}
        print("%-30s %-12s %-12s" % (c["name"][:30], mark[c["pass"]],
                                     mark[d["pass"]] if d else "-"))
        if d:
            both += bool(c["pass"] and d["pass"])
            only_a += bool(c["pass"] and not d["pass"])
            only_b += bool(d["pass"] and not c["pass"])
    print("%s: %s\n%s: %s" % (la, _summary(ra), lb, _summary(rb)))
    print("passed by both %d, only by %s %d, only by %s %d" % (
        both, la, only_a, lb, only_b))
    print("note: a few cases decide nothing; differences of one or two cases "
          "are within chance")


def _eval_diff(a, args):
    """What changed between the latest evaluation and the one before it
    (or the baseline with --baseline)."""
    st = Store(a.project)
    if len(args) != 1:
        die("eval diff NAME [--baseline]")
    recs = _eval_records(st, args[0], "path")
    if not recs:
        die("no evaluation of %s yet" % args[0])
    new = recs[-1]
    if a.baseline:
        old = _load_record(st, "baseline", args[0])
    elif len(recs) < 2:
        die("only one evaluation so far")
    else:
        old = recs[-2]
    worse, better = _changes_between(old, new)
    for c in worse:
        print("now failing: %s (%s)" % (c["name"], c["why"]))
    for c in better:
        print("now passing: %s" % c["name"])
    print("%s -> %s" % (_summary(old), _summary(new)))
    if not (worse or better):
        print("no case changed")
    return 1 if worse else 0


def _eval_baseline(a, args):
    st = Store(a.project)
    if len(args) != 1:
        die("eval baseline NAME [--save]")
    f = st.pdir / (args[0] + ".baseline.json")
    if a.save:
        rec = _latest(st, args[0], "path")
        rec.pop("_file", None)
        f.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        print("baseline of %s saved: %s. From now on eval reports any case "
              "that passed here and fails later" % (args[0], _summary(rec)))
        return
    if not f.is_file():
        print("no baseline (save the latest evaluation with: eval baseline "
              "%s --save)" % args[0])
        return
    print("baseline: " + _summary(json.loads(f.read_text(encoding="utf-8-sig"))))


def _eval_blame(a, args):
    """For each failed case of the latest evaluation, the steps of its run:
    which one failed, or where the expected text appeared and was lost.
    It points to files to look at; it cannot prove which one is wrong."""
    st = Store(a.project)
    if len(args) != 1:
        die("eval blame NAME [--case NAME]")
    rec = _latest(st, args[0], "path")
    failed = [c for c in rec["cases"] if c["pass"] is False
              and (not a.case or c["name"] == a.case)]
    if not failed:
        print("no failed case in the latest evaluation")
        return
    for c in failed:
        print("== %s: %s" % (c["name"], c["why"]))
        try:
            prod = json.loads(Path(c["record"]).read_text(encoding="utf-8-sig"))
            state = json.loads(Path(prod["state"]).read_text(
                encoding="utf-8")).get("steps", {})
        except (OSError, ValueError, KeyError, TypeError):
            print("  the run record is not available")
            continue
        exps = [str(x) for x in (c["expect"] if isinstance(c["expect"], list)
                                 else [c["expect"]])] \
            if not isinstance(c["expect"], dict) else []
        first = None
        for p in prod.get("steps", []):
            out = state.get(p["id"], "")
            has = bool(exps) and all(x.casefold() in out.casefold()
                                     for x in exps)
            if has and first is None:
                first = p
            print("  %-10s [%s] %-28s %-8s%s%s" % (
                p.get("number", ""), p["id"], p.get("title", "")[:28],
                p.get("status", ""), "  cached" if p.get("cached") else "",
                "  has the expected text" if has else ""))
        if prod.get("status") == "failed":
            bad = next((p for p in prod["steps"] if p.get("status") ==
                        "failed"), None)
            if bad:
                print("  -> look at %s [%s] %s: the run stopped there (%s)" % (
                    bad.get("number", ""), bad["id"], bad.get("title", ""),
                    prod.get("error", "")[:120]))
        elif exps and first is not None:
            print("  -> the expected text appeared at %s [%s] %s and was lost "
                  "after it: look at the steps that follow" % (
                      first.get("number", ""), first["id"],
                      first.get("title", "")))
        elif exps:
            print("  -> no step produced the expected text: look at the "
                  "steps meant to produce it, and at their instructions")
        else:
            print("  -> look at the last step that ran and the steps that "
                  "fed it")


def _eval_promote(a, args):
    """Turn failed cases of the latest evaluation, with their expected
    answers, into examples of a file. They leave the cases file (kept in
    paths/NAME.promoted.jsonl), since a case trained on no longer tests."""
    st = Store(a.project)
    if len(args) != 1 or not a.to:
        die("eval promote NAME --to REF [--case NAME]")
    name = args[0]
    nid = st.resolve(a.to)
    n = st.nodes[nid]
    if n.split == "eval":
        die("%s is an evaluation file; promote into a training file" % n.title)
    rec = _latest(st, name, "path")
    failed = [c for c in rec["cases"] if c["pass"] is False
              and (not a.case or c["name"] == a.case)]
    if not failed:
        print("no failed case to promote")
        return
    blocks = []
    for c in failed:
        exp = c["expect"]
        if isinstance(exp, dict):
            ans = json.dumps(exp, ensure_ascii=False)
            e = ""
        elif isinstance(exp, list):
            ans, e = "; ".join(map(str, exp)), "; ".join(map(str, exp))
        else:
            ans, e = str(exp), ""
        block = "Q: %s\nA: %s" % (c["input"], ans)
        if e:
            block += "\nE: " + e
        if c["output"].strip():
            block += "\nBAD: " + c["output"].strip()
        blocks.append(block)
    _backup(st, n)
    n.examples = "\n\n".join(x for x in [n.examples.strip("\n")] + blocks if x)
    st.save(n)
    cfile = _cases_file(st, name, a.cases)
    names = {c["name"] for c in failed}
    if cfile.is_file():
        keep, moved = [], []
        for line in cfile.read_text(encoding="utf-8-sig").splitlines():
            try:
                c = json.loads(line) if line.strip() and not \
                    line.lstrip().startswith("#") else None
            except ValueError:
                c = None
            if c is not None and c.get("name") in names:
                moved.append(line)
            else:
                keep.append(line)
        cfile.write_text("\n".join(keep) + "\n", encoding="utf-8")
        with (st.pdir / (name + ".promoted.jsonl")).open("a",
                                                        encoding="utf-8") as fh:
            fh.write("".join(l + "\n" for l in moved))
    print("%d case(s) became examples of %s [%s], with the wrong output as "
          "BAD:. They were moved out of the cases; write new cases to keep "
          "testing that skill." % (len(failed), n.title, nid))
    print("note: the A: lines are the expected texts; rewrite them as full "
          "answers before training")


def _eval_report(a, args):
    """A Markdown report: accuracy, time, tokens, failed cases, and the
    comparison with the direct model and the baseline when they exist."""
    st = Store(a.project)
    if len(args) != 1:
        die("eval report NAME [-o FILE]")
    name = args[0]
    rec = _latest(st, name, "path")
    secs = sorted(c.get("seconds", 0) for c in rec["cases"])
    lines = ["# Evaluation of %s" % name, "",
             "Model %s through %s; Tawjih %s." % (
                 rec.get("model") or "-", rec.get("backend"),
                 rec.get("tawjih")), "",
             "| | |", "|---|---|",
             "| Accuracy | %s |" % _summary(rec).split(",")[0],
             "| Time per case | median %.2f s, longest %.2f s |" % (
                 secs[len(secs) // 2] if secs else 0, secs[-1] if secs else 0),
             "| Tokens | %d in, %d out |" % (
                 sum(c.get("tokens_in", 0) for c in rec["cases"]),
                 sum(c.get("tokens_out", 0) for c in rec["cases"]))]
    others = [("direct", "The model asked directly")]
    if (st.pdir / (name + ".baseline.json")).is_file():
        others.append(("baseline", "Baseline"))
    for key, label in others:
        try:
            r = _load_record(st, key, name)
        except Die:
            continue
        lines.append("| %s | %s |" % (label, _summary(r)))
    lines += ["", "## Cases", "", "| Case | Result | Why |", "|---|---|---|"]
    for c in rec["cases"]:
        lines.append("| %s | %s | %s |" % (
            c["name"].replace("|", "/"),
            {True: "pass", False: "**fail**", None: "skip"}[c["pass"]],
            (c["why"] or "").replace("|", "/")[:120]))
    lines += ["", "A small number of cases is weak evidence; read the failed "
              "outputs before drawing conclusions."]
    text = "\n".join(lines) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print("report written to %s" % a.out)
    else:
        sys.stdout.write(text)


# ------------------------------------------------------------------ dataset
def cmd_dataset(a):
    """Write question-answer pairs as JSONL for fine-tuning tools (Unsloth,
    axolotl and the like): from passed evaluations (default), from finished
    runs, or each model step's prompt and answer (--steps). Read them
    before training: Tawjih does not judge their quality."""
    st = Store(a.project)
    rows, sources, prefs = [], {}, []
    if a.source == "examples":
        ids = st.subtree_ids(st.resolve(a.branch)) if a.branch else \
            st.subtree_ids("")
        for i in ids:
            n = st.nodes[i]
            if (n.split or "train") != (a.split or "train"):
                continue
            exs, probs = parse_examples(n.examples)
            if probs:
                die("%s [%s] has unreadable examples (%s); run check" % (
                    n.title, i, probs[0]))
            if not exs:
                continue
            w = float(n.weight or 1)
            if w == 0:          # weight 0 keeps a file out of training data
                continue
            sources[i] = hashlib.sha256(st.file_of(i).read_bytes()).hexdigest()
            for ex in exs:
                prompt = ex["q"]
                if a.with_rules and n.instructions.strip():
                    prompt = n.instructions.strip() + "\n\n" + ex["q"]
                for _ in range(max(1, round(w)) if a.repeat_by_weight else 1):
                    rows.append((prompt, ex["a"], {"file": i, "weight": w}))
                for bad in ex["bad"]:
                    if ex["a"]:
                        prefs.append({"prompt": prompt, "chosen": ex["a"],
                                      "rejected": bad})
    elif a.source == "eval":
        for f in sorted(st.rdir.glob("eval-*.json")) if st.rdir.is_dir() else []:
            data = json.loads(f.read_text(encoding="utf-8-sig"))
            if a.path and data.get("path") != a.path:
                continue
            if data.get("mode") == "direct":
                continue
            sources[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
            for c in data["cases"]:
                if c["pass"] or a.include_failed:
                    rows.append((c["input"], c["output"], {
                        "path": data["path"], "case": c["name"],
                        "pass": c["pass"]}))
    elif a.source == "runs":
        hdir = st.root / "history"
        for f in sorted(hdir.glob("production-*.json")) if hdir.is_dir() else []:
            rec = json.loads(f.read_text(encoding="utf-8-sig"))
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
            c = json.loads(f.read_text(encoding="utf-8-sig"))
            if "prompt" in c and (not a.step or c.get("step") == a.step):
                rows.append((c["prompt"], c["answer"], {
                    "step": c.get("step", ""), "model": c.get("model", "")}))
    seen, out = set(), []
    for q, ans, meta in rows:
        key = (q, ans)
        if (key in seen and not a.repeat_by_weight) or not ans.strip():
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
    if a.dpo:
        Path(a.dpo).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n"
                                       for x in prefs), encoding="utf-8")
        print("%d preference pair(s) (prompt, chosen, rejected) written to %s"
              % (len(prefs), a.dpo))
    # the seal: what this data was made from, to trace a trained model back
    man = Path(a.out).with_name(Path(a.out).name + ".manifest.json")
    man.write_text(json.dumps({
        "tawjih": __version__, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": a.source, "split": a.split or "train",
        "examples": len(out), "preference_pairs": len(prefs),
        "data_sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(),
        "from": sources}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("%d example(s) written to %s (from %s). Read them before training."
          % (len(out), a.out, a.source))
    if not out:
        hint = {"examples": "no file of that split has examples",
                "eval": "run eval first", "runs": "runs made before 0.7 do "
                "not keep their input", "steps": "answers cached before 0.7 "
                "do not keep their prompt"}[a.source]
        print("note: nothing found (%s)" % hint)


__all__ = [
    'EVAL_ACTIONS', 'JUDGE_PROMPT', '_case_passes', '_cases_file',
    '_changes_between', '_eval_baseline', '_eval_blame', '_eval_compare',
    '_eval_diff', '_eval_extract', '_eval_init', '_eval_promote',
    '_eval_records', '_eval_report', '_eval_synthesize', '_latest',
    '_load_record', '_read_cases', '_summary', 'cmd_dataset', 'cmd_eval',
]
