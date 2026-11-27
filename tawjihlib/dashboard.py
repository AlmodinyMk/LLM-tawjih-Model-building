"""tawjih dashboard (لوحة): the board of `tawjih top` in a window of its
own on this computer, drawn with Tkinter from Python's standard library: no
browser, no server, no network. Arabic (right to left) or English; the
processor, memory and each GPU's free memory; the project's model servers,
its runs, jobs and files; the chosen run or server; and the servers'
actions, with who did them. Everything on it is read from the project and
the system. Its keys act as in `top`: s start, x stop, r restart, t tune,
f flush, d the batch command.

Tk on Linux (X11) neither joins Arabic letters nor writes them right to
left, so on X11 the board does both itself (shape() and visual()); on
Windows and macOS the system does it.

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import json
import queue
import threading
import time
from pathlib import Path

from .core import Store, die, who
from .monitor import (
    Project, System, gpu_model, server_action, server_use, watched,
)

LOG_LINES = 14


def _steps(row):
    out = []
    for p in (row.get("record") or {}).get("steps") or []:
        secs = p.get("seconds")
        if secs is None and p.get("status") == "running" and \
                p.get("started_at"):
            secs = time.time() - p["started_at"]
        if secs is None:
            secs = (p.get("usage") or {}).get("seconds")
        out.append({"step": p.get("step"), "id": p.get("id", ""),
                    "number": p.get("number", ""), "title": p.get("title",
                                                                 ""),
                    "status": p.get("status", "waiting"),
                    "seconds": round(secs, 2) if secs else None})
    return out


def _row(r, procs):
    """A row of the board as plain JSON: no records, no settings that are
    not shown."""
    if r["kind"] == "server":
        cpu, rss = server_use(r, procs)
        cfg = r["cfg"]
        sets = ["%s=%s" % (k, cfg[k]) for k in ("tp", "pp", "context", "mem",
                                                 "kv_cache") if cfg.get(k)]
        sets += ["%s=%s" % (k, v if isinstance(v, str) else json.dumps(v))
                 for k, v in (cfg.get("args") or {}).items()]
        return {"kind": "server", "name": r["name"], "status": r["status"],
                "pid": r["pid"], "engine": r["engine"], "port": r["port"],
                "health": r["health"], "seconds": r["seconds"],
                "gpu_mib": r["gpu_mib"], "cpu": cpu, "rss": rss,
                "gpus": cfg.get("gpus") or "", "model": cfg.get("model") or "",
                "settings": sets, "command": r["command"],
                "tail": r["tail"][-12:]}
    if r["kind"] == "job":
        return {"kind": "job", "name": r["name"], "id": r.get("id"),
                "status": r["status"], "runtime": r.get("runtime", ""),
                "pct": r.get("pct"), "seconds": r.get("seconds"),
                "started": r.get("started", ""),
                "command": r.get("command", ""), "tail": r.get("tail", [])[-12:]}
    p = procs.get(r.get("pid")) or {}
    return {"kind": "run", "name": r["name"], "pid": r.get("pid"),
            "status": r["status"], "live": r.get("live", False),
            "done": r.get("done"), "total": r.get("steps"),
            "seconds": r.get("seconds"), "calls": r.get("calls"),
            "in": r.get("in"), "out": r.get("out"),
            "retries": r.get("retries"), "fallbacks": r.get("fallbacks"),
            "cache": r.get("cache"), "backend": r.get("backend") or "",
            "model": r.get("model") or "", "trees": r.get("trees") or [],
            "cpu": p.get("cpu"), "rss": p.get("rss"), "steps": _steps(r)}


def _files(store, stats):
    """The files of the trees in tree order, with their runs."""
    if store is None:
        return []
    out = []

    def walk(pid, depth):
        for n in store.children(pid):
            s = stats.get(n.id) or {}
            runs = s.get("runs", 0)
            out.append({"number": store.number(n.id), "title": n.title,
                        "depth": depth, "runs": runs,
                        "ok": round(100.0 * (runs - s.get("fails", 0)) / runs)
                        if runs else None,
                        "avg": round(s.get("seconds", 0) / runs, 2)
                        if runs else None,
                        "running": s.get("running", False),
                        "machine": bool((n.instructions or "").strip() or
                                        (getattr(n, "code", "") or "")
                                        .strip())})
            walk(n.id, depth + 1)
    try:
        walk("", 0)
    except (AttributeError, KeyError, TypeError):
        return out
    return out


def _log(root, notes):
    """The last actions on the servers, from history/, with the board's
    own messages."""
    rows = []
    d = Path(root) / "history"
    for f in sorted(d.glob("servers-*.jsonl"))[-2:] if d.is_dir() else []:
        for line in f.read_text(encoding="utf-8").splitlines()[-LOG_LINES:]:
            try:
                x = json.loads(line)
            except ValueError:
                continue
            by = x.get("by") or {}
            rows.append({"time": x.get("time", ""), "action": x.get("action"),
                         "server": x.get("server"), "by": by.get("name")
                         or by.get("id") or "",
                         "detail": " ".join(
                             " ".join(map(str, x[k])) if isinstance(
                                 x[k], list) else str(x[k])
                             for k in ("status", "path", "pid", "changes",
                                       "step")
                             if x.get(k) not in (None, "", []))})
    rows += notes
    rows.sort(key=lambda x: x["time"])
    return rows[-LOG_LINES:][::-1]


class Board:
    """What the page shows, sampled at most once a second for all
    viewers, and the actions its keys ask for."""

    def __init__(self, root, min_free=8000, read_only=False):
        self.root = str(Path(root).resolve())
        self.min_free, self.read_only = min_free, read_only
        self.project, self.system = Project(self.root), System(self.root)
        self.lock = threading.Lock()
        self.state, self.t = None, 0.0
        self.notes = []

    def snapshot(self):
        with self.lock:
            if self.state is not None and time.monotonic() - self.t < 1.0:
                return self.state
            snap = self.project.snapshot()
            s = self.system.sample(watched(snap))
            procs = s.get("procs") or {}
            gpus = [{"index": g.get("index", i), "name": g["name"],
                     "util": g["util"], "total_mib": round(g["total"] /
                                                           2 ** 20),
                     "used_mib": round(g["used"] / 2 ** 20),
                     "free_mib": round(g["free"] / 2 ** 20)
                     if g.get("free") is not None else None}
                    for i, g in enumerate(s.get("gpu") or [])]
            self.state = {
                "project": self.root, "time": time.strftime("%H:%M:%S"),
                "read_only": self.read_only, "min_free": self.min_free,
                "system": {"name": s.get("name"), "cores": s.get("count"),
                           "cpu": s.get("cpu"), "per_core": s.get("cores"),
                           "load": s.get("load"), "uptime": s.get("uptime"),
                           "mem": s.get("mem"), "disk": s.get("disk"),
                           "net": s.get("net"), "history": s.get("history"),
                           "gpus": gpus,
                           "free_gpus": [g["index"] for g in gpus
                                         if g["free_mib"] is not None and
                                         g["free_mib"] >= self.min_free]},
                "rows": [_row(r, procs) for r in snap["rows"]],
                "files": _files(self.project.tree(), snap["stats"]),
                "log": _log(self.root, self.notes)}
            self.t = time.monotonic()
            return self.state

    def act(self, name, key, text=""):
        if self.read_only:
            return 403, "this board was started with --read-only"
        if key not in ("s", "x", "r", "t", "f", "d"):
            return 400, "unknown key %s" % key
        rows = [r for r in self.project.servers() if r["name"] == name]
        if not rows:
            return 404, "no server %s" % name
        msg = server_action(self.root, rows[0], key, lambda q: text)
        by = who() or {}
        with self.lock:
            self.notes.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "action": "board " + key, "server": name,
                               "by": by.get("name") or by.get("id") or "",
                               "detail": msg})
            self.notes = self.notes[-LOG_LINES:]
            self.state = None                      # read everything again
        return (200 if not msg.startswith("error") else 409), msg


# ----------------------------------------------------------------- Arabic
# The contextual forms of Arabic letters (Unicode's presentation forms B):
# isolated, final, initial, medial; a letter with two forms joins only to
# the letter before it.
FORMS = {
    "ء": (0xFE80,), "آ": (0xFE81, 0xFE82),
    "أ": (0xFE83, 0xFE84), "ؤ": (0xFE85, 0xFE86),
    "إ": (0xFE87, 0xFE88), "ئ": (0xFE89, 0xFE8A, 0xFE8B, 0xFE8C),
    "ا": (0xFE8D, 0xFE8E), "ب": (0xFE8F, 0xFE90, 0xFE91, 0xFE92),
    "ة": (0xFE93, 0xFE94), "ت": (0xFE95, 0xFE96, 0xFE97, 0xFE98),
    "ث": (0xFE99, 0xFE9A, 0xFE9B, 0xFE9C),
    "ج": (0xFE9D, 0xFE9E, 0xFE9F, 0xFEA0),
    "ح": (0xFEA1, 0xFEA2, 0xFEA3, 0xFEA4),
    "خ": (0xFEA5, 0xFEA6, 0xFEA7, 0xFEA8),
    "د": (0xFEA9, 0xFEAA), "ذ": (0xFEAB, 0xFEAC),
    "ر": (0xFEAD, 0xFEAE), "ز": (0xFEAF, 0xFEB0),
    "س": (0xFEB1, 0xFEB2, 0xFEB3, 0xFEB4),
    "ش": (0xFEB5, 0xFEB6, 0xFEB7, 0xFEB8),
    "ص": (0xFEB9, 0xFEBA, 0xFEBB, 0xFEBC),
    "ض": (0xFEBD, 0xFEBE, 0xFEBF, 0xFEC0),
    "ط": (0xFEC1, 0xFEC2, 0xFEC3, 0xFEC4),
    "ظ": (0xFEC5, 0xFEC6, 0xFEC7, 0xFEC8),
    "ع": (0xFEC9, 0xFECA, 0xFECB, 0xFECC),
    "غ": (0xFECD, 0xFECE, 0xFECF, 0xFED0),
    "ف": (0xFED1, 0xFED2, 0xFED3, 0xFED4),
    "ق": (0xFED5, 0xFED6, 0xFED7, 0xFED8),
    "ك": (0xFED9, 0xFEDA, 0xFEDB, 0xFEDC),
    "ل": (0xFEDD, 0xFEDE, 0xFEDF, 0xFEE0),
    "م": (0xFEE1, 0xFEE2, 0xFEE3, 0xFEE4),
    "ن": (0xFEE5, 0xFEE6, 0xFEE7, 0xFEE8),
    "ه": (0xFEE9, 0xFEEA, 0xFEEB, 0xFEEC),
    "و": (0xFEED, 0xFEEE), "ى": (0xFEEF, 0xFEF0),
    "ي": (0xFEF1, 0xFEF2, 0xFEF3, 0xFEF4),
    "ـ": (0x0640, 0x0640, 0x0640, 0x0640),          # tatweel
}
LAM_ALEF = {"آ": (0xFEF5, 0xFEF6), "أ": (0xFEF7, 0xFEF8),
            "إ": (0xFEF9, 0xFEFA), "ا": (0xFEFB, 0xFEFC)}
MIRROR = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{",
          "<": ">", ">": "<", "«": "»", "»": "«"}


def _mark(ch):
    """Arabic vowel marks and the like: they ride on the letter before."""
    return "ً" <= ch <= "ٟ" or ch == "ٰ"


def _arabic(ch):
    return "؀" <= ch <= "ۿ" or "ﭐ" <= ch <= "﻿"


def shape(text):
    """Arabic letters in their joined forms (logical order kept)."""
    out, i, n = [], 0, len(text)

    def letter(j, step):
        """The nearest letter before (step -1) or after (+1), past marks."""
        j += step
        while 0 <= j < n and _mark(text[j]):
            j += step
        return text[j] if 0 <= j < n else ""
    while i < n:
        ch = text[i]
        forms = FORMS.get(ch)
        if not forms:
            out.append(ch)
            i += 1
            continue
        prev = letter(i, -1)
        joins_before = len(FORMS.get(prev, ())) == 4
        if ch == "ل":                         # lam + alef: one sign
            j = i + 1
            while j < n and _mark(text[j]):
                j += 1
            if j < n and text[j] in LAM_ALEF:
                lig = LAM_ALEF[text[j]]
                out.append(chr(lig[1] if joins_before else lig[0]))
                out.extend(text[i + 1:j])
                i = j + 1
                continue
        joins_after = len(forms) == 4 and len(FORMS.get(letter(i, 1),
                                                         ())) >= 2
        if len(forms) == 1:
            k = 0
        elif len(forms) == 2:
            k = 1 if joins_before else 0
        else:
            k = (3 if joins_after else 1) if joins_before else (
                2 if joins_after else 0)
        out.append(chr(forms[k]))
        i += 1
    return "".join(out)


def visual(text, rtl=None):
    """A line as it is drawn by a system that writes left to right: Arabic
    runs reversed (marks kept on their letters, brackets mirrored), the
    other runs as they are, and the runs in right-to-left order when the
    line is Arabic (rtl None: the first strong letter decides)."""
    if not any(_arabic(c) for c in text):
        return text
    text = shape(text)
    if rtl is None:
        first = next((c for c in text if _arabic(c) or c.isalpha()), "")
        rtl = _arabic(first)
    runs = []                                   # [is_arabic, [clusters]]
    for ch in text:
        if _mark(ch) and runs:
            runs[-1][1][-1] += ch
            continue
        # strong: Arabic (True) or another letter or digit (False);
        # anything else is neutral (None) and takes a side below
        kind = True if _arabic(ch) else False if ch.isalnum() else None
        if runs and runs[-1][0] is kind:
            runs[-1][1].append(ch)
        else:
            runs.append([kind, [ch]])
    # a neutral run between two runs of one kind belongs to that kind;
    # otherwise to the line's direction
    for k, r in enumerate(runs):
        if r[0] is None:
            before = runs[k - 1][0] if k else None
            after = runs[k + 1][0] if k + 1 < len(runs) else None
            r[0] = before if before == after and before is not None else rtl
    merged = []
    for kind, cl in runs:
        if merged and merged[-1][0] == kind:
            merged[-1][1].extend(cl)
        else:
            merged.append([kind, list(cl)])
    pieces = []
    for kind, cl in merged:
        if kind:
            pieces.append("".join(MIRROR.get(c, c) for c in reversed(cl)))
        else:
            pieces.append("".join(cl))
    if rtl:
        pieces.reverse()
    return "".join(pieces)


# ----------------------------------------------------------------- window
WORDS = {
    "ar": {"title": "توجيه · لوحة التحكم", "sys": "العتاد والذاكرة",
           "srv": "الخوادم", "files": "الأشجار", "det": "التفاصيل",
           "runs": "التشغيلات", "log": "سجل الأفعال",
           "free": "للاختيار التلقائي (حرة {1} MiB فأكثر): {0}", "none": "لا شيء",
           "noGpu": "لا بطاقة: nvidia-smi غير موجود",
           "noSrv": "لا خوادم بعد: tawjih server add NAME --engine vllm",
           "choose": "اختر صفاً", "needSrv": "اختر خادماً أولاً",
           "ro": "للعرض فقط", "ask": "KEY=VALUE ... (مثل mem=0.9):",
           "sure": "هل تريد {0} الخادم {1}؟", "lang": "English",
           "keys": "↑↓ تحديد   s تشغيل   x إيقاف   r إعادة   t ضبط   "
                   "f تفريغ   d توزيع   +/- التحديث   L اللغة   q خروج",
           "files_cols": ("الرقم", "العنوان", "مرات", "نجاح", "المتوسط"),
           "runs_cols": ("PID", "المسار / الخادم", "الحالة", "الخطوة",
                         "الزمن", "نداءات", "معالج/ذاكرة",
                         "المزود/المحرك"),
           "log_cols": ("الوقت", "الفعل", "الخادم", "من", "التفاصيل"),
           "st": {"running": "يعمل", "done": "اكتمل", "failed": "فشل",
                  "stopped": "متوقف", "partial": "جزئي",
                  "waiting": "ينتظر", "skipped": "تُخطّي", "made": "أُعدّ",
                  "submitted": "أُرسل", "collected": "جُمع"},
           "l": {"status": "الحالة", "up": "يعمل منذ", "engine": "المحرك",
                 "port": "المنفذ", "gpus": "البطاقات", "health": "الصحة",
                 "model": "النموذج", "settings": "الإعدادات",
                 "gpuMem": "ذاكرة البطاقات", "elapsed": "المدة",
                 "steps": "الخطوات", "calls": "الاستدعاءات",
                 "tokens": "الرموز", "backend": "المزود",
                 "runtime": "بيئة التنفيذ", "x": "إيقاف",
                 "r": "إعادة تشغيل"}},
    "en": {"title": "tawjih · dashboard", "sys": "system & gpu",
           "srv": "servers", "files": "trees", "det": "details",
           "runs": "runs", "log": "actions",
           "free": "free for auto: {0}  ≥ {1} MiB", "none": "none",
           "noGpu": "no GPU: nvidia-smi is not here",
           "noSrv": "no servers yet: tawjih server add NAME --engine vllm",
           "choose": "choose a row", "needSrv": "choose a server first",
           "ro": "read only", "ask": "KEY=VALUE ... (mem=0.9):",
           "sure": "{0} the server {1}?", "lang": "العربية",
           "keys": "↑↓ select   s start   x stop   r restart   t tune   "
                   "f flush   d batch   +/- refresh   L language   q quit",
           "files_cols": ("No.", "Title", "Runs", "OK", "Avg"),
           "runs_cols": ("PID", "Path / server", "Status", "Step", "Time",
                         "Calls", "CPU / RAM", "Backend / engine"),
           "log_cols": ("Time", "Action", "Server", "By", "Detail"),
           "st": {}, "l": {"status": "Status", "up": "Up",
                           "engine": "Engine", "port": "Port",
                           "gpus": "GPUs", "health": "Health",
                           "model": "Model", "settings": "Settings",
                           "gpuMem": "GPU memory", "elapsed": "Elapsed",
                           "steps": "Steps", "calls": "Calls",
                           "tokens": "Tokens", "backend": "Backend",
                           "runtime": "Runtime", "x": "stop",
                           "r": "restart"}},
}
# the columns' widths in pixels; the widest stretches with the window
WIDTHS = {"files_cols": (110, 230, 50, 55, 65),
          "runs_cols": (55, 135, 90, 65, 65, 70, 115, 105),
          "log_cols": (75, 85, 80, 80, 300)}
COLOURS = {"bg": "#16161e", "panel": "#1a1b26", "line": "#292e42",
           "frame": "#414868", "text": "#c0caf5", "sub": "#a9b1d6",
           "dim": "#565f89", "green": "#9ece6a", "cyan": "#7dcfff",
           "blue": "#7aa2f7", "pink": "#f7768e", "yellow": "#e0af68",
           "mauve": "#bb9af7"}
ICON = {"running": "▶", "done": "✓", "failed": "✗", "stopped": "■",
        "partial": "◐", "skipped": "–", "waiting": "·", "submitted": "◔",
        "collected": "✓"}


def _bytes(n):
    if n is None:
        return "—"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return ("%.0f %s" if unit == "B" else "%.1f %s") % (n, unit)
        n /= 1024.0


def _secs(s):
    if s is None:
        return ""
    s = float(s)
    if s < 60:
        return "%.1fs" % s
    if s < 3600:
        return "%dm%02ds" % (s // 60, s % 60)
    return "%dh%02dm" % (s // 3600, s % 3600 // 60)


def row_cells(r, lang):
    """The cells of a row of the runs table, in reading order."""
    st = WORDS[lang]["st"].get(r["status"], r["status"])
    state = "%s %s" % (ICON.get(r["status"], "·"), st)
    if r["kind"] == "server":
        use = " ".join(x for x in (
            "%.0f%%" % r["cpu"] if r.get("cpu") is not None else "",
            _bytes(r["rss"]) if r.get("rss") else "") if x)
        return (str(r["pid"] or "srv"), r["name"], state, ":%s" % r["port"],
                _secs(r["seconds"]), str(r["health"] or "—"), use,
                r["engine"])
    if r["kind"] == "job":
        return ("job", r["name"], state, "%.0f%%" % r["pct"]
                if r.get("pct") is not None else "", _secs(r["seconds"]),
                "", "", r["runtime"])
    use = " ".join(x for x in (
        "%.0f%%" % r["cpu"] if r.get("cpu") is not None else "",
        _bytes(r["rss"]) if r.get("rss") else "") if x)
    return (str(r["pid"] or ""), r["name"], state, "%s/%s" % (
        r["done"], r["total"]), _secs(r["seconds"]), str(r["calls"] or 0),
        use, r["backend"])


def detail_lines(r, lang):
    """The chosen row as (label, value) pairs and then plain lines."""
    w = WORDS[lang]["l"]
    if r is None:
        return [], [WORDS[lang]["choose"]]
    if r["kind"] == "server":
        pairs = [(w["status"], WORDS[lang]["st"].get(r["status"],
                                                    r["status"])),
                 (w["up"], _secs(r["seconds"]) or "—"),
                 (w["engine"], r["engine"]), (w["port"], str(r["port"])),
                 (w["gpus"], r["gpus"] or "—"),
                 (w["health"], str(r["health"] or "—")),
                 (w["model"], r["model"] or "—")]
        if r.get("cpu") is not None:
            pairs.append(("CPU", "%.1f%%" % r["cpu"]))
        if r.get("rss"):
            pairs.append(("RAM", _bytes(r["rss"])))
        if r.get("gpu_mib") is not None:
            pairs.append((w["gpuMem"], "%d MiB" % r["gpu_mib"]))
        if r["settings"]:
            pairs.append((w["settings"], " ".join(r["settings"])))
        return pairs, ["$ " + r["command"]] + r["tail"]
    if r["kind"] == "job":
        return ([(w["runtime"], r["runtime"]),
                 ("%", "%.0f%%" % r["pct"] if r.get("pct") is not None
                  else "—")], ["$ " + r["command"]] + r["tail"])
    pairs = [(w["elapsed"], _secs(r["seconds"])),
             (w["steps"], "%s/%s" % (r["done"], r["total"])),
             (w["calls"], str(r["calls"] or 0)),
             (w["tokens"], "%s / %s" % (r["in"] or 0, r["out"] or 0)),
             (w["backend"], r["backend"] or "—"),
             (w["model"], r["model"] or "—")]
    if r.get("cpu") is not None:
        pairs.append(("CPU", "%.1f%%" % r["cpu"]))
    if r.get("rss"):
        pairs.append(("RAM", _bytes(r["rss"])))
    lines = ["%s %s  %-10s %s  %s" % (
        ICON.get(p["status"], "·"), p["step"], p["number"], p["title"],
        _secs(p["seconds"])) for p in r["steps"]]
    return pairs, lines


class Window:
    """The board in a Tk window. Data is read in a thread, so a slow server
    never freezes the window; the window only draws."""

    def __init__(self, board, lang="ar", every=2.0, tk=None):
        if tk is None:
            try:
                import tkinter as tk
            except ImportError:
                die("the dashboard needs Tkinter, which this Python does "
                    "not have: on Linux install python3-tk (Debian, Ubuntu) "
                    "or python3-tkinter (Fedora); Python from python.org "
                    "has it on Windows and macOS. `tawjih top` shows the "
                    "same board in the terminal")
        from tkinter import messagebox, simpledialog, ttk
        self.tk, self.ttk = tk, ttk
        self.ask_yes, self.ask_text = messagebox.askyesno, \
            simpledialog.askstring
        self.board, self.lang, self.every = board, lang, every
        self.state, self.sel_key, self.msg = None, None, ""
        self.q = queue.Queue()
        self.busy, self.acting = False, None
        self.root = tk.Tk()
        self.x11 = self.root.tk.call("tk", "windowingsystem") == "x11"
        self.root.configure(bg=COLOURS["bg"])
        self.root.geometry("1360x820")
        self._style()
        self._build()
        self._bind()
        self.relabel()
        self.root.after(50, self.poll)
        self.root.after(80, self.fetch)

    # ------------------------------------------------------------ helpers
    def t(self, text, rtl=None):
        """Text as Tk must be given it: on X11 Arabic is joined and turned
        here; elsewhere the system does it."""
        if self.x11:
            return visual(text, rtl)
        return text

    def w(self, key):
        return WORDS[self.lang][key]

    def _style(self):
        c = COLOURS
        s = self.ttk.Style(self.root)
        try:
            s.theme_use("clam")
        except self.tk.TclError:
            pass        # another theme: the colours below still apply
        s.configure("T.Treeview", background=c["panel"],
                    fieldbackground=c["panel"], foreground=c["sub"],
                    rowheight=22, borderwidth=0, font=("DejaVu Sans Mono",
                                                       10))
        s.configure("T.Treeview.Heading", background=c["bg"],
                    foreground=c["dim"], relief="flat",
                    font=("DejaVu Sans Mono", 9))
        s.map("T.Treeview", background=[("selected", c["line"])],
              foreground=[("selected", c["text"])])

    def _frame(self, colour):
        c = COLOURS
        f = self.tk.LabelFrame(self.root, bg=c["panel"], fg=colour,
                               highlightthickness=0, bd=1, relief="solid",
                               font=("DejaVu Sans Mono", 10, "bold"),
                               labelanchor="ne" if self.lang == "ar"
                               else "nw")
        f.configure(highlightbackground=colour)
        return f

    def _tree(self, parent, cols):
        tv = self.ttk.Treeview(parent, columns=[str(i) for i in
                                                range(len(cols))],
                               show="headings", style="T.Treeview",
                               selectmode="browse")
        tv.pack(fill="both", expand=True, padx=4, pady=4)
        for name, colour in (("ok", "green"), ("bad", "pink"),
                             ("warn", "yellow"), ("srv", "mauve"),
                             ("dim", "dim")):
            tv.tag_configure(name, foreground=COLOURS[colour])
        return tv

    def _build(self):
        c, tk = COLOURS, self.tk
        self.f_sys = self._frame(c["green"])
        self.sys = tk.Canvas(self.f_sys, bg=c["panel"], highlightthickness=0,
                             height=150)
        self.sys.pack(fill="both", expand=True, padx=4, pady=4)
        self.f_srv = self._frame(c["cyan"])
        self.srv = tk.Canvas(self.f_srv, bg=c["panel"], highlightthickness=0,
                             height=60)
        self.srv.pack(fill="both", expand=True, padx=4, pady=4)
        self.f_files = self._frame(c["blue"])
        self.files = self._tree(self.f_files, WORDS["en"]["files_cols"])
        self.f_det = self._frame(c["cyan"])
        self.det = tk.Text(self.f_det, bg=c["panel"], fg=c["text"], bd=0,
                           height=9, wrap="none", highlightthickness=0,
                           font=("DejaVu Sans Mono", 10))
        self.det.pack(fill="both", expand=True, padx=6, pady=4)
        for name, colour in (("lbl", "dim"), ("val", "text"),
                             ("ok", "green"), ("sub", "sub")):
            self.det.tag_configure(name, foreground=c[colour])
        self.f_runs = self._frame(c["pink"])
        self.runs = self._tree(self.f_runs, WORDS["en"]["runs_cols"])
        self.f_log = self._frame(c["yellow"])
        self.log = self._tree(self.f_log, WORDS["en"]["log_cols"])
        self.foot = tk.Frame(self.root, bg=c["bg"])
        self.keys = tk.Label(self.foot, bg=c["bg"], fg=c["dim"],
                             font=("DejaVu Sans Mono", 10))
        self.msgl = tk.Label(self.foot, bg=c["bg"], fg=c["yellow"],
                             font=("DejaVu Sans Mono", 10))
        self.langb = tk.Button(self.foot, bg=c["line"], fg=c["cyan"],
                               activebackground=c["frame"], bd=0,
                               command=self.switch, font=("DejaVu Sans Mono",
                                                          10))
        self.runs.bind("<<TreeviewSelect>>", self.chose)
        self.srv.bind("<Button-1>", self.chip)

    def layout(self):
        """Two columns: system, servers and trees beside details, runs and
        actions; their sides swap in Arabic."""
        for f in (self.f_sys, self.f_srv, self.f_files, self.f_det,
                  self.f_runs, self.f_log, self.foot):
            f.grid_forget()
        a, b = (1, 0) if self.lang == "ar" else (0, 1)
        self.root.grid_columnconfigure(a, weight=5, uniform="c")
        self.root.grid_columnconfigure(b, weight=7, uniform="c")
        # rows shared so that a tall system panel (many cores and GPUs)
        # leaves the details their own height and gives the runs the rest
        for r, wt in ((0, 0), (1, 2), (2, 0), (3, 2), (4, 1), (5, 0)):
            self.root.grid_rowconfigure(r, weight=wt)
        pad = dict(padx=6, pady=5, sticky="nsew")
        self.f_sys.grid(row=0, column=a, rowspan=2, **pad)
        self.f_srv.grid(row=2, column=a, **pad)
        self.f_files.grid(row=3, column=a, rowspan=2, **pad)
        self.f_det.grid(row=0, column=b, **pad)
        self.f_runs.grid(row=1, column=b, rowspan=3, **pad)
        self.f_log.grid(row=4, column=b, **pad)
        self.foot.grid(row=5, column=0, columnspan=2, sticky="ew", padx=6)
        side, other = ("right", "left") if self.lang == "ar" else \
            ("left", "right")
        for wdg in (self.keys, self.msgl, self.langb):
            wdg.pack_forget()
        self.keys.pack(side=side)
        self.langb.pack(side=other, padx=4)
        self.msgl.pack(side=other, padx=8)

    def relabel(self):
        ar = self.lang == "ar"
        self.root.title(self.w("title"))
        for f, key in ((self.f_sys, "sys"), (self.f_srv, "srv"),
                       (self.f_files, "files"), (self.f_det, "det"),
                       (self.f_runs, "runs"), (self.f_log, "log")):
            f.configure(text=" %s " % self.t(self.w(key), ar),
                        labelanchor="ne" if ar else "nw")
        for tv, key in ((self.files, "files_cols"),
                        (self.runs, "runs_cols"), (self.log, "log_cols")):
            names = self.w(key)
            n = len(names)
            order = list(range(n))[::-1] if ar else list(range(n))
            tv.configure(displaycolumns=[str(i) for i in order])
            widths = WIDTHS[key]
            for i, name in enumerate(names):
                tv.heading(str(i), text=self.t(name, ar),
                           anchor="e" if ar else "w")
                tv.column(str(i), anchor="e" if ar else "w",
                          width=widths[i], minwidth=40,
                          stretch=i == widths.index(max(widths)))
        self.keys.configure(text=self.t(self.w("keys"), ar))
        self.langb.configure(text="[L] " + self.t(self.w("lang")))
        self.layout()
        self.draw()

    def _bind(self):
        r = self.root
        r.bind("<Down>", lambda e: self.move(1))
        r.bind("<Up>", lambda e: self.move(-1))
        for k in "sxrtfd":
            r.bind(k, lambda e, k=k: self.act(k))
        r.bind("l", lambda e: self.switch())
        r.bind("L", lambda e: self.switch())
        r.bind("q", lambda e: r.destroy())
        r.bind("+", lambda e: self.speed(-0.5))
        r.bind("=", lambda e: self.speed(-0.5))
        r.bind("-", lambda e: self.speed(0.5))

    # --------------------------------------------------------------- data
    def fetch(self):
        """Read the board in a thread, then again after the interval."""
        if not self.busy:
            self.busy = True
            # The thread holds the board and the queue, never the window:
            # Tk must only be freed in the thread that made it.
            board, q = self.board, self.q

            def work():
                try:
                    q.put(("state", board.snapshot()))
                except Exception as e:                  # shown, not raised
                    q.put(("msg", "error: %s" % e))
            threading.Thread(target=work, daemon=True).start()
        self.root.after(int(self.every * 1000), self.fetch)

    def poll(self):
        try:
            while True:
                kind, value = self.q.get_nowait()
                if kind == "state":
                    self.busy = False
                    self.state = value
                    self.draw()
                else:
                    self.say(value)
        except queue.Empty:
            pass        # nothing new: look again shortly
        self.root.after(100, self.poll)

    def say(self, text):
        self.msg = text
        self.msgl.configure(text=self.t(text))

    # ---------------------------------------------------------- drawing
    def draw(self):
        if self.state is None:
            return
        self.draw_sys()
        self.draw_servers()
        self.fill(self.files, [
            (("  " * f["depth"] + f["number"], f["title"],
              str(f["runs"] or "·"), "" if f["ok"] is None else
              "%d%%" % f["ok"], _secs(f["avg"])),
             "ok" if f["running"] else "" if f["machine"] else "dim",
             f["number"]) for f in self.state["files"]])
        rows = self.state["rows"]
        self.fill(self.runs, [
            (row_cells(r, self.lang), "srv" if r["kind"] == "server" else
             "ok" if r["status"] == "running" else "bad"
             if r["status"] == "failed" else "",
             "%s:%s:%s" % (r["kind"], r["name"], r.get("pid") or
                           r.get("id") or ""))
            for r in rows])
        self.fill(self.log, [
            ((x["time"][11:] or x["time"], x["action"], x["server"],
              x["by"], x["detail"]), "", "%d" % i)
            for i, x in enumerate(self.state["log"])])
        keys = [self._key(r) for r in rows]
        if self.sel_key not in keys and keys:
            self.sel_key = keys[0]
        if self.sel_key in keys:
            iid = self.sel_key
            if self.runs.selection() != (iid,):
                self.runs.selection_set(iid)
        self.draw_detail()

    def _key(self, r):
        return "%s:%s:%s" % (r["kind"], r["name"], r.get("pid") or
                             r.get("id") or "")

    def fill(self, tv, items):
        ar = self.lang == "ar"
        old = set(tv.get_children())
        new = []
        for k, (cells, tag, iid) in enumerate(items):
            iid = iid or str(k)
            while iid in new:
                iid += "+"
            new.append(iid)
            vals = [self.t(str(c), ar if j else None) for j, c in
                    enumerate(cells)]
            if iid in old:
                tv.item(iid, values=vals, tags=(tag,) if tag else ())
                tv.move(iid, "", k)
            else:
                tv.insert("", k, iid=iid, values=vals,
                          tags=(tag,) if tag else ())
        for iid in old - set(new):
            tv.delete(iid)

    def chosen(self):
        if not self.state:
            return None
        for r in self.state["rows"]:
            if self._key(r) == self.sel_key:
                return r
        return None

    def draw_detail(self):
        ar = self.lang == "ar"
        pairs, lines = detail_lines(self.chosen(), self.lang)
        d = self.det
        d.configure(state="normal")
        d.delete("1.0", "end")
        d.tag_configure("dir", justify="right" if ar else "left")
        # the pairs a few to a line, so none runs off the panel
        room = max(40, d.winfo_width() // 8 - 4)
        rows, cur = [], ""
        for label, value in pairs:
            part = "%s: %s" % (label, value)
            if cur and len(cur) + 3 + len(part) > room:
                rows.append(cur)
                cur = part
            else:
                cur = cur + "   " + part if cur else part
        if cur:
            rows.append(cur)
        for row in rows:
            d.insert("end", self.t(row, ar) + "\n", ("val", "dir"))
        d.insert("end", "\n")
        for t in lines:            # commands and logs read left to right
            d.insert("end", self.t(t) + "\n", ("sub",))
        d.configure(state="disabled")

    def bar(self, cv, x, y, w, pct, colour=None, n=24):
        """A meter of small squares, green to red or one colour."""
        pct = max(0.0, min(100.0, pct or 0.0))
        lit = int(round(pct / 100.0 * n))
        step = w / float(n)
        for i in range(n):
            if i < lit:
                fill = colour or _grade(i / float(max(1, n - 1)))
            else:
                fill = COLOURS["line"]
            cv.create_rectangle(x + i * step, y, x + i * step + step - 2,
                                y + 10, fill=fill, width=0)

    def draw_sys(self):
        cv, c, s = self.sys, COLOURS, self.state["system"]
        ar = self.lang == "ar"
        cv.delete("all")
        width = max(300, cv.winfo_width())
        font = ("DejaVu Sans Mono", 10)
        lx, bx, bw = 10, 110, width - 330
        vx = width - 10

        def line(y, label, pct, value, colour=None, lab_colour=None):
            cv.create_text(lx, y + 5, text=label, anchor="w", font=font,
                           fill=lab_colour or c["sub"])
            self.bar(cv, bx, y, bw, pct, colour)
            cv.create_text(vx, y + 5, text=value, anchor="e", font=font,
                           fill=c["sub"])
        def grid(y, items, colour=None):
            """Many small bars: one column up to four, then two or four."""
            n = len(items)
            cols = 1 if n <= 4 else 2 if n <= 16 else 4
            rows = (n + cols - 1) // cols
            cw = (width - 20) // cols
            for k, (label, pct, value) in enumerate(items):
                cx, cy = lx + (k // rows) * cw, y + (k % rows) * 20
                if cols == 1:
                    line(cy, label, pct, value, colour)
                    continue
                cv.create_text(cx, cy + 5, text=label, anchor="w",
                               font=font, fill=c["sub"])
                self.bar(cv, cx + 60, cy, cw - 120, pct, colour)
                cv.create_text(cx + cw - 10, cy + 5, text=value,
                               anchor="e", font=font, fill=c["sub"])
            return y + rows * 20

        def pc(v):
            return "—" if v is None else "%.0f%%" % v
        y = 6
        kinds = {}
        for g in s.get("gpus") or []:
            kinds.setdefault(gpu_model(g["name"]), []).append(str(g["index"]))
        what = [s.get("name") or "CPU"] + [
            "%s× %s (%s)" % (len(ix), k, ",".join(ix)) if len(ix) > 1 else
            "%s (%s)" % (k, ix[0]) for k, ix in kinds.items()]
        cv.create_text(lx, y + 5, text="  ·  ".join(what), anchor="w",
                       font=font, fill=c["text"])
        y += 22
        line(y, "CPU %s" % (s.get("cores") or ""), s.get("cpu"),
             pc(s.get("cpu")) + (
                 "  LAV %s" % " ".join("%.2f" % x for x in s["load"])
                 if s.get("load") else ""), lab_colour=c["cyan"])
        y += 20
        y = grid(y, [("CPU %d" % i, v, pc(v))
                     for i, v in enumerate(s.get("per_core") or [])])
        m = s.get("mem") or {}
        if m.get("total"):
            p = 100.0 * m["used"] / m["total"]
            line(y, "RAM", p, "%s / %s  %.0f%%" % (_bytes(m["used"]),
                                                   _bytes(m["total"]), p),
                 c["pink"] if p > 85 else None, c["mauve"])
            y += 20
        if s.get("disk"):
            k = s["disk"]
            line(y, "Disk", 100.0 * k["used"] / k["total"],
                 "%s free" % _bytes(k["free"]))
            y += 20
        y += 6
        gpus = s.get("gpus") or []
        if not gpus:
            cv.create_text(vx if ar else lx, y + 5, text=self.t(
                self.w("noGpu"), ar), anchor="e" if ar else "w", font=font,
                fill=c["dim"])
            y += 20
        for g in gpus:          # memory: used, and free (red below the limit)
            low = g["free_mib"] is not None and \
                g["free_mib"] < self.state["min_free"]
            pct = 100.0 * g["used_mib"] / max(1, g["total_mib"])
            line(y, "VRAM %d" % g["index"], pct,
                 "%s free / %s" % (
                     "—" if g["free_mib"] is None else
                     _bytes(g["free_mib"] * 2 ** 20),
                     _bytes(g["total_mib"] * 2 ** 20)),
                 c["pink"] if low else None, c["pink"] if low else None)
            y += 20
        if gpus:                # how busy each GPU is
            y = grid(y + 6, [("GPU %d" % g["index"], g["util"],
                              "%s  %s" % (gpu_model(g["name"]), pc(g["util"])))
                             for g in gpus])
        if gpus:
            free = ",".join(str(i) for i in s["free_gpus"]) or \
                self.w("none")
            cv.create_text(vx if ar else lx, y + 5, text=self.t(
                self.w("free").format(free, self.state["min_free"]), ar),
                anchor="e" if ar else "w", font=font,
                fill=c["green"] if s["free_gpus"] else c["pink"])
            y += 20
        cv.configure(height=y + 6)

    def draw_servers(self):
        cv, c = self.srv, COLOURS
        ar = self.lang == "ar"
        cv.delete("all")
        self.chips = []
        srv = [r for r in self.state["rows"] if r["kind"] == "server"]
        font = ("DejaVu Sans Mono", 10)
        width = max(300, cv.winfo_width())
        if not srv:
            cv.create_text(width - 8 if ar else 8, 14, text=self.t(
                self.w("noSrv"), ar), anchor="e" if ar else "w", font=font,
                fill=c["dim"])
            return
        x, y = (width - 8, 6) if ar else (8, 6)
        for r in srv:
            colour = (c["green"] if r["health"] == 200 else c["yellow"]) \
                if r["pid"] else c["dim"]
            label = "● %s  %s:%s" % (r["name"], r["engine"], r["port"])
            item = cv.create_text(0, 0, text=label, anchor="w", font=font,
                                  fill=colour)
            x0, y0, x1, y1 = cv.bbox(item)
            wid = x1 - x0 + 14
            if (ar and x - wid < 8) or (not ar and x + wid > width - 8):
                x, y = (width - 8 if ar else 8), y + 26
            left = x - wid if ar else x
            cv.create_rectangle(left, y, left + wid, y + 20,
                                fill=c["line"], width=0)
            cv.coords(item, left + 7, y + 10)
            cv.tag_raise(item)
            self.chips.append((left, y, left + wid, y + 20, self._key(r)))
            x = left - 6 if ar else left + wid + 6
        cv.configure(height=y + 26)

    # ------------------------------------------------------------ actions
    def chose(self, _e=None):
        sel = self.runs.selection()
        if sel:
            self.sel_key = sel[0]
            self.draw_detail()

    def chip(self, e):
        for x0, y0, x1, y1, key in getattr(self, "chips", []):
            if x0 <= e.x <= x1 and y0 <= e.y <= y1:
                self.sel_key = key
                self.draw()

    def move(self, step):
        kids = list(self.runs.get_children())
        if not kids:
            return
        i = kids.index(self.sel_key) if self.sel_key in kids else 0
        self.sel_key = kids[max(0, min(len(kids) - 1, i + step))]
        self.runs.selection_set(self.sel_key)
        self.runs.see(self.sel_key)
        self.draw_detail()

    def speed(self, delta):
        self.every = max(0.5, min(10.0, self.every + delta))
        self.say("%dms" % int(self.every * 1000))

    def switch(self):
        self.lang = "en" if self.lang == "ar" else "ar"
        self.relabel()

    def act(self, key):
        r = self.chosen()
        if self.board.read_only:
            return self.say(self.w("ro"))
        if not r or r["kind"] != "server":
            return self.say(self.w("needSrv"))
        text = ""
        if key in ("x", "r") and not self.ask_yes(
                "tawjih", self.w("sure").format(self.w("l")[key],
                                                r["name"]), parent=self.root):
            return None
        if key == "t":
            text = self.ask_text("tawjih", "%s: %s" % (r["name"],
                                                      self.w("ask")),
                                 parent=self.root) or ""
            if not text:
                return None
        self.say("…")

        board, q, name = self.board, self.q, r["name"]   # not the window

        def work():
            code, msg = board.act(name, key, text)
            q.put(("msg", msg))
        # kept so a caller can wait: the action swaps sys.stdout while it
        # runs (server commands print), which is the whole process's
        self.acting = threading.Thread(target=work, daemon=True)
        self.acting.start()
        return None

    def run(self):
        self.root.mainloop()


def _grade(x):
    a, b, c = (158, 206, 106), (224, 175, 104), (247, 118, 142)
    lo, hi, t = (a, b, x / 0.5) if x < 0.5 else (b, c, (x - 0.5) / 0.5)
    return "#%02x%02x%02x" % tuple(int(lo[i] + (hi[i] - lo[i]) * t)
                                   for i in range(3))


def cmd_dashboard(a):
    """The board of `tawjih top` in a window of its own on this computer."""
    st = Store(a.project)
    board = Board(st.root, a.min_free, a.read_only)
    Window(board, a.lang or "ar", a.interval).run()
    return 0


def add_dashboard_parser(sub):
    s = sub.add_parser("dashboard", aliases=["لوحة"], help="the board of top "
                       "in a window of its own on this computer (Tkinter), "
                       "in Arabic or English, with no browser or network")
    s.add_argument("--lang", choices=["ar", "en"], help="the board's "
                   "language: ar (Arabic, right to left, the default) or en "
                   "(English); L switches it")
    s.add_argument("--min-free", type=int, default=8000, metavar="MIB",
                   help="a GPU with less free memory is shown in red")
    s.add_argument("--interval", type=float, default=2.0, metavar="S",
                   help="seconds between refreshes (+ and - change it)")
    s.add_argument("--read-only", action="store_true",
                   help="show only: the keys that act on servers are off")
    s.set_defaults(fn=cmd_dashboard)
    return s


__all__ = ['Board', 'Window', 'add_dashboard_parser', 'cmd_dashboard',
           'detail_lines', 'row_cells', 'shape', 'visual']
