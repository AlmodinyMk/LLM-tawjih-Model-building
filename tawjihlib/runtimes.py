"""runtimes - where long jobs (training, merging, building models) run.

A runtime is this computer (local), another computer reached with ssh, or
any service driven by commands you write (command), such as a cloud tool's
command line. Runtimes belong to the person, not to a project: they are kept
in ~/.config/tawjih/runtimes.json (or the file TAWJIH_RUNTIMES names), so a
project copied from elsewhere cannot bring commands that would run on this
computer. No key or password is ever stored: ssh uses your own keys, and a
command runtime's commands use whatever your tools already have.

A job is a folder models/jobs/ID/ in the project: run.sh, the files it needs,
and job.json, its record. run.sh writes log.txt while it runs, and `exit`
with its exit status when it ends; results go to out/ inside the folder and
are collected from there.

Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

KINDS = ("local", "ssh", "command")
TEMPLATES = ("submit", "status", "logs", "cancel", "collect", "check")
LOCAL = {"kind": "local"}


from .core import Die, die, user_config_dir, who
from .providers import load_policy


RuntimeError_ = Die          # what a runtime or job problem raises


def fail(msg):
    die(msg)


# ----------------------------------------------------------------- settings
def runtimes_file():
    env = os.environ.get("TAWJIH_RUNTIMES")
    if env:
        return Path(env)
    return user_config_dir() / "runtimes.json"


def load():
    f = runtimes_file()
    data = {}
    if f.is_file():
        try:
            data = json.loads(f.read_text(encoding="utf-8-sig"))
        except ValueError as e:
            fail("%s is not readable JSON (%s)" % (f, e))
    data.setdefault("runtimes", {})
    data.setdefault("default", "local")
    return data


def save(data):
    f = runtimes_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                 encoding="utf-8")


def get(name=None):
    """(name, settings) of a runtime; 'local' always exists."""
    data = load()
    name = name or data["default"] or "local"
    if name == "local" and name not in data["runtimes"]:
        return name, dict(LOCAL)
    if name not in data["runtimes"]:
        fail("no runtime '%s' (runtime list shows them; runtime add "
             "creates one)" % name)
    return name, dict(data["runtimes"][name])


SECRET = re.compile(r"(sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|"
                    r"(api[_-]?key|token|password|secret)\s*=\s*\S{8,})",
                    re.I)


def check_settings(cfg):
    kind = cfg.get("kind")
    if kind not in KINDS:
        fail("kind must be one of: %s" % ", ".join(KINDS))
    if kind == "ssh" and not re.fullmatch(r"[\w.\-]+@?[\w.\-]*",
                                          cfg.get("host", "")):
        fail("an ssh runtime needs --host user@machine")
    if kind == "command" and not cfg.get("submit"):
        fail("a command runtime needs --submit 'COMMAND {job}'")
    for k in TEMPLATES + ("python", "workdir", "host"):
        if SECRET.search(str(cfg.get(k, ""))):
            fail("%s looks like it holds a key or password; keep those in "
                 "your tools' own settings or environment variables" % k)


def is_local(cfg):
    return cfg.get("kind") == "local" or (
        cfg.get("kind") == "ssh" and cfg.get("host", "").split("@")[-1] in
        ("localhost", "127.0.0.1", "::1"))


def allowed(root, name, cfg, yes):
    """Refuse what the project's policy forbids; ask for --yes before data
    leaves this computer (and may cost money)."""
    net = load_policy(root)["allow_network"]
    if not is_local(cfg):
        if net is not True:
            fail("the project's policy allows no runtime outside this "
                 "computer; %s is one" % name)
        if not yes:
            fail("runtime %s is outside this computer: the job's files are "
                 "sent there and it may cost money. Check the plan above, "
                 "then add --yes" % name)


# --------------------------------------------------------------------- jobs
def jobs_dir(root):
    return Path(root) / "models" / "jobs"


def make_job(root, name, runtime, command, files=None, texts=None,
             collect_to=None):
    """A job folder: files copied in (bundle path -> source), texts written
    (bundle path -> text), and run.sh running command (a list, run inside
    the folder). Returns the folder."""
    jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    jdir = jobs_dir(root) / jid
    jdir.mkdir(parents=True)
    for rel, src in (files or {}).items():
        target = jdir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, target)
    for rel, text in (texts or {}).items():
        target = jdir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    (jdir / "out").mkdir(exist_ok=True)
    (jdir / "run.sh").write_text(
        "#!/bin/sh\n# written by Tawjih: job %s (%s)\n"
        "cd \"$(dirname \"$0\")\" || exit 1\n"
        "echo $$ > pid\n"
        "rm -f exit\n"
        "%s > log.txt 2>&1\n"
        "echo $? > exit\n" % (jid, name, " ".join(shlex.quote(str(c))
                                                  for c in command)),
        encoding="utf-8")
    record(jdir, {"id": jid, "name": name, "runtime": runtime,
                  "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "by": who(),
                  "command": [str(c) for c in command],
                  "collect_to": str(collect_to) if collect_to else "",
                  "status": "made"})
    return jdir


def record(jdir, update=None):
    f = Path(jdir) / "job.json"
    data = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else {}
    if update:
        data.update(update)
        f.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")
    return data


def find_job(root, jid):
    d = jobs_dir(root)
    hits = [x for x in sorted(d.iterdir()) if x.name.startswith(jid)] \
        if d.is_dir() and jid else []
    if not hits and jid == "last" and d.is_dir():
        started = [x for x in d.iterdir() if record(x).get("status")
                   not in (None, "made")]
        hits = sorted(started, key=lambda x: record(x).get("started", ""))[-1:]
    if len(hits) != 1:
        fail(("no job %s" % jid if jid != "last" else "no job has been "
              "started yet") if not hits else
             "%s names %d jobs; give more of its id" % (jid, len(hits)))
    return hits[0]


def _fill(template, jdir, cfg, rec):
    return template.format(job=shlex.quote(str(jdir)), id=rec["id"],
                           name=shlex.quote(rec["name"]),
                           workdir=cfg.get("workdir", "tawjih-jobs"))


def _remote(cfg, rec):
    return "%s/%s" % (cfg.get("workdir", "tawjih-jobs").rstrip("/"),
                      rec["id"])


def _sh(cmd, capture=True):
    """Run a shell command line (a runtime's own command)."""
    r = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _run(argv, cwd=None, stdin=None):
    """ssh and scp as argument lists, never through this computer's shell
    (cmd.exe on Windows would take the remote &&, > and $(...) for its
    own): the remote command reaches the remote shell as written."""
    try:
        r = subprocess.run(argv, cwd=cwd, input=stdin, capture_output=True,
                           text=True, stdin=None if stdin is not None
                           else subprocess.DEVNULL)
    except OSError as e:
        return 127, "cannot run %s: %s" % (argv[0], e)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def _ssh(cfg, command):
    return _run(["ssh", cfg["host"], command])


def _ssh_steps(jdir, cfg, rec):
    """(argv, folder to run it in) that send the job and start it. scp runs
    inside the job folder with relative names, so no local path (with a
    drive letter) is ever taken for a host."""
    host, rdir = cfg["host"], _remote(cfg, rec)
    names = sorted(x.name for x in Path(jdir).iterdir())
    return [(["ssh", host, "mkdir -p %s" % shlex.quote(rdir)], None),
            (["scp", "-r"] + names + ["%s:%s/" % (host, rdir)], str(jdir)),
            (["ssh", host, "cd %s && nohup sh run.sh > /dev/null 2>&1 &"
              % shlex.quote(rdir)], None)]


def plan(jdir, name, cfg):
    """The commands that would start the job, as text."""
    rec = record(jdir)
    kind = cfg["kind"]
    if kind == "local":
        return ["sh %s" % shlex.quote(str(jdir / "run.sh"))]
    if kind == "ssh":
        return [("(in the job folder) " if cwd else "") + " ".join(
            shlex.quote(x) for x in argv)
            for argv, cwd in _ssh_steps(jdir, cfg, rec)]
    return [_fill(cfg["submit"], jdir, cfg, rec)]


def submit(root, jdir, name, cfg, wait=True):
    """Start the job on its runtime. Local jobs run here and are waited for
    (unless wait is False); others are started and checked with status."""
    rec = record(jdir)
    kind = cfg["kind"]
    if kind == "local":
        record(jdir, {"status": "running", "started":
                      time.strftime("%Y-%m-%d %H:%M:%S")})
        if wait:           # here, without sh, so it works on Windows too
            with open(jdir / "log.txt", "w", encoding="utf-8") as log:
                try:
                    code = subprocess.run(rec["command"], cwd=jdir,
                                          stdout=log,
                                          stderr=subprocess.STDOUT
                                          ).returncode
                except OSError as e:
                    log.write("cannot start %s: %s\n" % (rec["command"][0],
                                                         e))
                    code = 127
            (jdir / "exit").write_text("%d\n" % code)
            record(jdir, {"status": "done" if code == 0 else "failed",
                          "exit": code})
            if code:
                print((jdir / "log.txt").read_text(encoding="utf-8",
                                                   errors="replace")[-2000:],
                      file=sys.stderr)
            return code
        subprocess.Popen(["sh", str(jdir / "run.sh")],
                         start_new_session=True)
        return 0
    if kind == "ssh":
        steps = [(argv, cwd) for argv, cwd in _ssh_steps(jdir, cfg, rec)]
    else:
        steps = [(line, None) for line in plan(jdir, name, cfg)]
    for cmd, cwd in steps:
        code, out = _run(cmd, cwd) if isinstance(cmd, list) else _sh(cmd)
        if code != 0:
            record(jdir, {"status": "failed to start"})
            fail("could not start the job on %s: %s\n%s" % (
                name, cmd if isinstance(cmd, str) else " ".join(cmd),
                out.strip()))
    record(jdir, {"status": "submitted", "started":
                  time.strftime("%Y-%m-%d %H:%M:%S")})
    return 0


def status(jdir, cfg):
    rec = record(jdir)
    kind = cfg["kind"]
    if kind == "local":
        e = jdir / "exit"
        if e.is_file():
            code = int(e.read_text().strip() or 1)
            return "done" if code == 0 else "failed (exit %d)" % code
        return "running" if (jdir / "pid").is_file() else rec.get(
            "status", "made")
    if kind == "ssh":
        rdir = _remote(cfg, rec)
        q = shlex.quote(rdir)
        code, out = _ssh(cfg, "cat %s/exit 2>/dev/null || (test -f %s/pid "
                         "&& kill -0 $(cat %s/pid) 2>/dev/null && echo "
                         "running) || echo unknown" % (q, q, q))
        out = out.strip().splitlines()[-1] if out.strip() else "unknown"
        return "done" if out == "0" else "running" if out == "running" else \
            "failed (exit %s)" % out if out.isdigit() else out
    if not cfg.get("status"):
        return "unknown (the runtime has no status command)"
    code, out = _sh(_fill(cfg["status"], jdir, cfg, rec))
    last = (out.strip().splitlines() or ["unknown"])[-1].strip()
    if code != 0:
        return "running or unknown (%s)" % last[:80]
    return "done" if last == "0" else "failed (exit %s)" % last \
        if last.isdigit() else last


def logs(jdir, cfg):
    rec = record(jdir)
    kind = cfg["kind"]
    if kind == "local":
        f = jdir / "log.txt"
        return f.read_text(encoding="utf-8", errors="replace") \
            if f.is_file() else ""
    if kind == "ssh":
        return _ssh(cfg, "cat %s/log.txt" % shlex.quote(
            _remote(cfg, rec)))[1]
    if not cfg.get("logs"):
        return "(the runtime has no logs command)"
    return _sh(_fill(cfg["logs"], jdir, cfg, rec))[1]


def cancel(jdir, cfg):
    rec = record(jdir)
    kind = cfg["kind"]
    if kind == "local":
        pid = jdir / "pid"
        if pid.is_file():
            try:
                os.kill(int(pid.read_text().strip()), 15)
            except (OSError, ValueError):
                pass
    elif kind == "ssh":
        rdir = _remote(cfg, rec)
        _ssh(cfg, "kill $(cat %s/pid)" % shlex.quote(rdir))
    elif cfg.get("cancel"):
        _sh(_fill(cfg["cancel"], jdir, cfg, rec))
    else:
        fail("the runtime has no cancel command")
    record(jdir, {"status": "cancelled"})


def collect(jdir, cfg):
    """Bring out/ back into the job folder, then to where the job's results
    belong (collect_to). Returns that place."""
    rec = record(jdir)
    kind = cfg["kind"]
    if kind == "ssh":
        (jdir / "out").mkdir(exist_ok=True)
        rdir = "%s:%s" % (cfg["host"], _remote(cfg, rec))
        code, out = _run(["scp", "-r", rdir + "/out/.", "."],
                         str(jdir / "out"))
        if code != 0:
            fail("could not collect: %s" % out.strip())
        _run(["scp", rdir + "/log.txt", "log.txt"], str(jdir))
    elif kind == "command":
        if not cfg.get("collect"):
            fail("the runtime has no collect command")
        code, out = _sh(_fill(cfg["collect"], jdir, cfg, rec))
        if code != 0:
            fail("could not collect: %s" % out.strip())
    target = rec.get("collect_to")
    if target:
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        for f in (jdir / "out").iterdir():
            dest = target / f.name
            if f.is_dir():
                shutil.copytree(f, dest, dirs_exist_ok=True)
            else:
                shutil.copy(f, dest)
    record(jdir, {"status": "collected", "collected":
                  time.strftime("%Y-%m-%d %H:%M:%S")})
    return target or jdir / "out"


PROBE = r"""
import importlib, json, os, platform, shutil, sys
out = {"python": sys.version.split()[0], "system": platform.platform(),
       "cpus": os.cpu_count(), "modules": {}}
for m in ("torch", "transformers", "peft", "trl", "datasets"):
    try:
        out["modules"][m] = getattr(importlib.import_module(m),
                                    "__version__", "installed")
    except Exception:
        out["modules"][m] = None
try:
    import torch
    out["gpus"] = [torch.cuda.get_device_name(i)
                   for i in range(torch.cuda.device_count())]
    out["mps"] = bool(getattr(torch.backends, "mps", None)
                      and torch.backends.mps.is_available())
except Exception:
    out["gpus"], out["mps"] = [], False
du = shutil.disk_usage(".")
out["free_gb"] = round(du.free / 1e9, 1)
print(json.dumps(out))
"""


def check(name, cfg):
    """What the runtime has: Python, libraries, GPUs, free disk."""
    kind = cfg["kind"]
    py = cfg.get("python") or (sys.executable if kind == "local"
                               else "python3")
    if kind == "local":
        r = subprocess.run([py, "-c", PROBE], capture_output=True,
                           text=True, timeout=300)
        out = r.stdout
    elif kind == "ssh":
        r = subprocess.run(["ssh", cfg["host"], py, "-"], input=PROBE,
                           capture_output=True, text=True, timeout=300)
        out = r.stdout
    else:
        if not cfg.get("check"):
            return None
        out = _sh(_fill(cfg["check"], ".", cfg, {"id": "check",
                                                 "name": name}))[1]
    try:
        return json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        fail("%s did not answer the check (%s)" % (name, out.strip()[:300]))


def finish(root, jdir, cfg):
    """Collect a job's results and note them in models/registry.json."""
    where = collect(jdir, cfg)
    rec = record(jdir)
    print("results of job %s in %s" % (jdir.name, where))
    if rec.get("kind") in ("train", "merge", "pipeline") and \
            rec.get("adapter"):
        f = Path(root) / "models" / "registry.json"
        reg = json.loads(f.read_text(encoding="utf-8")) if f.is_file() \
            else {"adapters": {}, "bindings": {}}
        entry = reg.setdefault("adapters", {}).setdefault(rec["adapter"], {})
        if rec["kind"] in ("train", "pipeline"):
            entry["trained"] = True
            for key in ("path", "base", "rank", "peft", "merged", "hub"):
                if rec.get(key) is not None:
                    entry[key] = rec[key]
        else:
            try:
                entry["merged"] = Path(rec["collect_to"]).resolve(
                    ).relative_to(Path(root).resolve()).as_posix()
            except ValueError:
                entry["merged"] = rec["collect_to"]
        entry.setdefault("jobs", []).append(jdir.name)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(reg, ensure_ascii=False, indent=2) + "\n",
                     encoding="utf-8")


def cmd_runtime(a):
    """Where long jobs run (kept in the person's settings, not the
    project), and the project's jobs."""
    act, args = a.action, a.args
    data = load()
    if act == "list":
        names = ["local"] + sorted(n for n in data["runtimes"]
                                   if n != "local")
        for n in names:
            cfg = data["runtimes"].get(n, LOCAL)
            print("%-14s %-8s %s%s" % (
                n, cfg["kind"], cfg.get("host") or cfg.get("submit", "")
                or "this computer", "  (default)" if n == (
                    data["default"] or "local") else ""))
        print("settings: %s" % runtimes_file())
        return 0
    if act == "add":
        if len(args) != 1 or not re.fullmatch(r"[\w\-]+", args[0]):
            fail("runtime add NAME --kind local|ssh|command ...")
        cfg = {"kind": a.kind}
        for k in ("host", "python", "workdir") + TEMPLATES:
            v = getattr(a, "rt_python" if k == "python" else k, None)
            if v:
                cfg[k] = v
        check_settings(cfg)
        data["runtimes"][args[0]] = cfg
        save(data)
        print("runtime %s (%s) saved in %s" % (args[0], a.kind,
                                               runtimes_file()))
        return 0
    if act == "rm":
        if not args or args[0] not in data["runtimes"]:
            fail("no runtime %s" % (args[0] if args else ""))
        del data["runtimes"][args[0]]
        if data["default"] == args[0]:
            data["default"] = "local"
        save(data)
        print("removed %s" % args[0])
        return 0
    if act == "default":
        if not args:
            print("default runtime: %s" % (data["default"] or "local"))
            return 0
        get(args[0])
        data["default"] = args[0]
        save(data)
        print("default runtime: %s" % args[0])
        return 0
    if act == "check":
        name, cfg = get(args[0] if args else None)
        info = check(name, cfg)
        if info is None:
            print("%s has no check command" % name)
            return 0
        print("%s: Python %s, %s, %s CPU(s), %s GB free" % (
            name, info["python"], info.get("system", ""),
            info.get("cpus", "?"), info.get("free_gb", "?")))
        for m, v in info["modules"].items():
            print("  %-13s %s" % (m, v or "-"))
        print("GPU: %s" % (", ".join(info["gpus"]) if info.get("gpus") else
                           "Apple MPS" if info.get("mps") else "none found"))
        return 0
    if act == "jobs":
        d = jobs_dir(a.project)
        rows = sorted(d.iterdir()) if d.is_dir() else []
        for jd in rows:
            rec = record(jd)
            print("%-22s %-10s %-14s %s" % (jd.name, rec.get("runtime", ""),
                                             rec.get("status", ""),
                                             rec.get("name", "")))
        if not rows:
            print("no jobs yet")
        return 0
    if act in ("status", "logs", "cancel", "collect"):
        jdir = find_job(a.project, args[0] if args else "last")
        rec = record(jdir)
        name, cfg = get(rec["runtime"])
        if act == "status":
            st = status(jdir, cfg)
            record(jdir, {"last_status": st})
            print("%s: %s" % (jdir.name, st))
        elif act == "logs":
            print(logs(jdir, cfg), end="")
        elif act == "cancel":
            cancel(jdir, cfg)
            print("cancelled %s" % jdir.name)
        else:
            finish(a.project, jdir, cfg)
        return 0
    fail("unknown action %s" % act)



__all__ = ['cmd_runtime', 'runtimes_file']
