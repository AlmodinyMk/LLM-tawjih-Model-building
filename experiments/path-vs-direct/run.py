#!/usr/bin/env python3
"""Does the path beat the model asked directly? A measurement to run on
your own computer with Ollama.

For each model, the same quadratic equations go three ways:

  path    the example's path: the model turns the equation into a, b, c;
          Python solves it and checks the roots; the model explains;
  direct  the question alone, with one line saying how to write the
          answer (x = value), so that both are scored alike;
  steps   the same, asked to work step by step first.

Each case is asked --repeat times. The report gives the share of answers
that are right, with a 95% interval, the cases where the path and the
model differ, a McNemar test on them, and the time and tokens spent.

    python experiments/path-vs-direct/run.py --models qwen2.5:3b
    python experiments/path-vs-direct/run.py --models qwen2.5:1.5b,llama3.2:3b \\
        --cases 60 --repeat 3

Standard library only. Part of Tawjih. Copyright (c) 2026 Mohammed Khalid
Almodiny. SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import argparse
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXAMPLE = REPO / "examples" / "quadratic"
TAWJIH = REPO / "tawjih.py"
PATH = "branch"
ARMS = ("path", "direct", "steps")
FORMAT = {"ar": "اكتب الجواب النهائي لكل جذر بالصيغة x = القيمة.",
          "en": "Write the final answer for each root as x = value."}
STEPS = {"ar": "حل المسألة خطوة بخطوة قبل الجواب النهائي.",
         "en": "Work it out step by step before the final answer."}


# -------------------------------------------------------------------- cases
def expect_for(roots):
    """The example's own test: 'x = r' (also x₁, x_1, x1) for each root."""
    parts = []
    for r in sorted(set(roots)):
        num = ("[-−]\\s?%d" % -r) if r < 0 else ("\\+?%d" % r)
        parts.append("(?=.*x\\s*(?:[₁₂]|_?[12])?\\s*=\\s*%s(?![\\d.]))" % num)
    return "(?s)" + "".join(parts)


def _term(coef, power, first):
    if coef == 0:
        return ""
    sign = "−" if coef < 0 else ("" if first else "+")
    mag = abs(coef)
    var = {2: "x²", 1: "x", 0: ""}[power]
    body = ("" if mag == 1 and power else str(mag)) + var
    return (sign + body) if first else " %s %s" % (sign, body)


def poly(a, b, c):
    out = ""
    for coef, p in ((a, 2), (b, 1), (c, 0)):
        out += _term(coef, p, not out)
    return out or "0"


def make_cases(count, seed=7):
    """The example's 20 cases, then generated ones with integer roots:
    larger roots and leading coefficients, terms on both sides."""
    cases = []
    for line in (EXAMPLE / "paths" / "branch.cases.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.startswith("{"):
            c = json.loads(line)
            c["lang"] = "ar" if any("؀" <= ch <= "ۿ"
                                    for ch in c["input"]) else "en"
            c["source"] = "example"
            cases.append(c)
    rnd = random.Random(seed)
    k = 0
    while len(cases) < count:
        k += 1
        r1 = rnd.choice([x for x in range(-15, 16) if x])
        r2 = r1 if rnd.random() < 0.1 else rnd.choice(
            [x for x in range(-15, 16) if x and x != r1])
        a = rnd.choice([1, 1, 2, 3, 4, 5, 6, -1, -2, -3])
        b, c = -a * (r1 + r2), a * r1 * r2
        form = rnd.randrange(4)
        if form == 0:
            eq = "%s = 0" % poly(a, b, c)
        elif form == 1:
            eq = "%s = %s" % (poly(a, b, 0), poly(0, 0, -c))
        elif form == 2:
            eq = "%s = %s" % (poly(a, 0, 0), poly(0, -b, -c))
        else:
            p, q, s = rnd.randint(1, 4), rnd.randint(-9, 9), rnd.randint(-20, 20)
            eq = "%s = %s" % (poly(a + p, b + q, c + s), poly(p, q, s))
        lang = "ar" if k % 2 else "en"
        one = r1 == r2
        text = rnd.choice(
            ["حل المعادلة %s واذكر %s." % (eq, "قيمة x" if one else
                                           "قيمتي x"),
             "أوجد %s المعادلة %s." % ("جذر" if one else "جذري", eq)]
            if lang == "ar" else
            ["Solve %s and give %s." % (eq, "the value of x" if one else
                                       "both values of x"),
             "Find the %s of %s." % ("root" if one else "roots", eq)])
        cases.append({"name": "g%03d" % k, "input": text,
                      "expect": expect_for([r1, r2]), "match": "regex",
                      "roots": sorted({r1, r2}), "lang": lang,
                      "source": "generated", "form": form, "a": a})
    return cases[:count]


def arm_cases(cases, arm):
    out = []
    for c in cases:
        c = dict(c)
        if arm == "direct":
            c["input"] = c["input"] + "\n" + FORMAT[c["lang"]]
        elif arm == "steps":
            c["input"] = "%s\n%s %s" % (c["input"], STEPS[c["lang"]],
                                        FORMAT[c["lang"]])
        out.append(c)
    return out


# -------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, mid - half), min(1.0, mid + half))


def bootstrap(values, rounds=2000, seed=11):
    """A 95% interval of the mean, resampling cases (runs of one case are
    not independent of each other)."""
    if not values:
        return (0.0, 0.0)
    rnd = random.Random(seed)
    n = len(values)
    means = sorted(sum(rnd.choice(values) for _ in range(n)) / n
                   for _ in range(rounds))
    return (means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1])


def mcnemar(only_a, only_b):
    """Two-sided exact p of McNemar's test on the discordant cases."""
    n = only_a + only_b
    if not n:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


# --------------------------------------------------------------- the runs
def ollama_models(host):
    with urllib.request.urlopen(host.rstrip("/") + "/api/tags",
                                timeout=10) as r:
        data = json.loads(r.read().decode("utf-8"))
    return {m.get("name"): m for m in data.get("models", [])}


def find_model(models, name):
    for n in (name, name + ":latest"):
        if n in models:
            return models[n]
    return None


def tawjih(project, *args, quiet=False):
    env = dict(os.environ, TAWJIH_NO_SETUP="1")
    argv = [sys.executable, str(TAWJIH), "-p", str(project)] + list(args)
    if quiet:
        return subprocess.run(argv, env=env, capture_output=True, text=True)
    return subprocess.run(argv, env=env)


def run_arm(project, cases_file, arm, model, repeat):
    before = set((project / "runs").glob("eval-%s-*.json" % PATH))
    args = ["eval", PATH, "--cases", str(cases_file), "--backend", "local",
            "--model", model, "--repeat", str(repeat)]
    args += ["--direct"] if arm != "path" else ["--allow-code"]
    tawjih(project, *args)
    new = sorted(set((project / "runs").glob("eval-%s-*.json" % PATH)) -
                 before, key=lambda f: f.stat().st_mtime)
    if not new:
        sys.exit("no evaluation record was written for %s / %s" % (model,
                                                                    arm))
    return json.loads(new[-1].read_text(encoding="utf-8"))


def case_rows(record, repeat):
    rows = {}
    for c in record["cases"]:
        runs = c.get("runs") or [c]
        rows[c["name"]] = {
            "passes": sum(bool(r["pass"]) for r in runs), "runs": len(runs),
            "seconds": c["seconds"] / len(runs),
            "tokens": (c["tokens_in"] + c["tokens_out"]) / len(runs),
            "why": c["why"], "output": (c["output"] or "")[:2000]}
    return rows


def summarise(results, cases, repeat):
    lines = []
    for model, arms in results.items():
        lines.append("\n## %s\n" % model)
        lines.append("| way | right answers | 95% interval | cases right "
                     "every time | seconds / case | tokens / case |")
        lines.append("|---|---|---|---|---|---|")
        rate = {}
        for arm in ARMS:
            rows = arms.get(arm)
            if not rows:
                continue
            k = sum(r["passes"] for r in rows.values())
            n = sum(r["runs"] for r in rows.values())
            per = [r["passes"] / r["runs"] for r in rows.values()]
            rate[arm] = {c: r["passes"] / r["runs"] for c, r in rows.items()}
            lo, hi = bootstrap(per)
            always = sum(r["passes"] == r["runs"] for r in rows.values())
            lines.append("| %s | %d / %d (%.0f%%) | %.0f%% – %.0f%% | %d / %d "
                         "| %.1f | %.0f |" % (
                             arm, k, n, 100 * k / n if n else 0, 100 * lo,
                             100 * hi, always, len(rows),
                             sum(r["seconds"] for r in rows.values()) /
                             len(rows),
                             sum(r["tokens"] for r in rows.values()) /
                             len(rows)))
        for other in ("direct", "steps"):
            if "path" not in rate or other not in rate:
                continue
            a, b = rate["path"], rate[other]
            diffs = [a[c] - b[c] for c in a if c in b]
            lo, hi = bootstrap(diffs)
            maj_a = {c for c in a if a[c] >= 0.5}
            maj_b = {c for c in b if b[c] >= 0.5}
            only_a, only_b = len(maj_a - maj_b), len(maj_b - maj_a)
            lines.append("\npath − %s: %+.0f points (95%%: %+.0f to %+.0f); "
                         "right only by the path in %d cases, only by %s in "
                         "%d; McNemar p = %.3g" % (
                             other, 100 * sum(diffs) / len(diffs), 100 * lo,
                             100 * hi, only_a, other, only_b,
                             mcnemar(only_a, only_b)))
        by = {}
        for c in cases:
            key = "%s, %s" % (c["source"], c["lang"])
            by.setdefault(key, []).append(c["name"])
        lines.append("\n| group | " + " | ".join(a for a in ARMS if a in rate)
                     + " |")
        lines.append("|---|" + "---|" * len(rate))
        for key, names in sorted(by.items()):
            lines.append("| %s (%d) | " % (key, len(names)) + " | ".join(
                "%.0f%%" % (100 * sum(rate[a][n] for n in names) /
                            len(names)) for a in ARMS if a in rate) + " |")
    return "\n".join(lines)


def environment(host, models, chosen):
    env = {"tawjih": tawjih(EXAMPLE, "--version", quiet=True).stdout.strip()
           .replace("tawjih ", ""),
           "python": sys.version.split()[0],
           "system": platform.platform(), "machine": platform.machine(),
           "processor": platform.processor() or "", "cpus": os.cpu_count(),
           "ollama": host, "models": {}}
    for m in chosen:
        info = find_model(models, m) or {}
        env["models"][m] = {"digest": info.get("digest", ""),
                            "size": info.get("size"),
                            "details": info.get("details", {})}
    if shutil.which("nvidia-smi"):
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                            "--format=csv,noheader"], capture_output=True,
                           text=True)
        env["gpus"] = r.stdout.strip().splitlines()
    return env


def say(text):
    """Print, replacing what the console cannot show (the files keep it)."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, "replace").decode(enc))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--models", default="qwen2.5:3b",
                   help="Ollama models, separated by commas (qwen2.5:3b)")
    p.add_argument("--cases", type=int, default=60, help="how many "
                   "equations: the example's 20, then generated ones (60)")
    p.add_argument("--repeat", type=int, default=3,
                   help="times each case is asked in each way (3)")
    p.add_argument("--arms", default=",".join(ARMS),
                   help="which ways (path,direct,steps)")
    p.add_argument("--host", default=os.environ.get(
        "OLLAMA_HOST", "http://localhost:11434"))
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--temperature", type=float,
                   help="sent to the model (by default the model's own)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results",
                   help="the folder for the report (results)")
    a = p.parse_args(argv)
    for f in (sys.stdout, sys.stderr):    # Arabic and − on any console
        try:
            f.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    host = a.host if a.host.startswith("http") else "http://" + a.host
    chosen = [m.strip() for m in a.models.split(",") if m.strip()]
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    if not set(arms) <= set(ARMS) or not arms:
        sys.exit("--arms takes path, direct and steps")
    try:
        models = ollama_models(host)
    except OSError as e:
        sys.exit("Ollama does not answer at %s (%s): start it with "
                 "'ollama serve'" % (host, e))
    missing = [m for m in chosen if not find_model(models, m)]
    if missing:
        sys.exit("pull these first:\n" + "\n".join("  ollama pull " + m
                                                   for m in missing))
    out = Path(a.out).resolve()
    work = out / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    cases = make_cases(a.cases, a.seed)
    calls = len(cases) * a.repeat * len(chosen) * sum(
        {"path": 2, "direct": 1, "steps": 1}[x] for x in arms)
    print("%d cases x %d ways x %d repeats x %d models: about %d model "
          "calls (a few seconds each on a GPU, often longer on a CPU)" % (
              len(cases), len(arms), a.repeat, len(chosen), calls))
    results = {}
    t0 = time.time()
    for m in chosen:
        project = work / m.replace(":", "_").replace("/", "_")
        shutil.copytree(EXAMPLE, project, ignore=shutil.ignore_patterns(
            "runs", "history", "cache"))
        prov = ["provider", "add", "local", "--kind", "ollama",
                "--base-url", host, "--num-ctx", str(a.num_ctx)]
        if a.temperature is not None:
            prov += ["--option", "temperature=%s" % a.temperature]
        r = tawjih(project, *prov, quiet=True)
        if r.returncode:
            sys.exit(r.stderr)
        results[m] = {}
        for arm in arms:
            f = project / ("cases-%s.jsonl" % arm)
            f.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n"
                                 for c in arm_cases(cases, arm)),
                         encoding="utf-8")
            print("\n=== %s / %s" % (m, arm), flush=True)
            rec = run_arm(project, f, arm, m, a.repeat)
            results[m][arm] = case_rows(rec, a.repeat)
    env = environment(host, models, chosen)
    env["minutes"] = round((time.time() - t0) / 60, 1)
    settings = {"cases": len(cases), "repeat": a.repeat, "arms": arms,
                "seed": a.seed, "num_ctx": a.num_ctx,
                "temperature": a.temperature}
    (out / "results.json").write_text(json.dumps(
        {"settings": settings, "environment": env, "cases": cases,
         "results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    text = ("# Path or direct: quadratic equations\n\n"
            "%d cases (the example's 20 and %d generated), each asked %d "
            "times in each way; %s; Tawjih %s, %s, %.1f minutes.\n\n"
            "path: the model gives a, b, c, Python solves and checks, the "
            "model explains. direct: the question and how to write the "
            "answer. steps: the same, step by step.\n" % (
                len(cases), len(cases) - sum(c["source"] == "example"
                                             for c in cases),
                a.repeat, ", ".join(chosen), env["tawjih"],
                env["system"], env["minutes"]) +
            summarise(results, cases, a.repeat) +
            "\n\nA case counts as right when the answer says x = r for each "
            "root r (the example's own test). The intervals resample "
            "cases; McNemar compares the cases each way gets right in most "
            "of its repeats.\n")
    (out / "summary.md").write_text(text, encoding="utf-8")
    say("\n" + text)
    print("written: %s and %s; every run's log and record are in %s" % (
        out / "summary.md", out / "results.json", work))
    return 0


if __name__ == "__main__":
    sys.exit(main())
