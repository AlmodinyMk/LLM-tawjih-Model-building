"""tawjih top (مراقبة): a live board, like btop, of the runs of paths over
the files of the trees beside the computer's performance: CPU, memory,
disk, network and GPU. It reads history/live/ (written by runs as
they go), history/ (finished runs), models/jobs/ (runtime jobs) and the
system, and changes nothing.

Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

from .commands import _fancy, display_width, tree_lines
from .core import Store
from .home import C, Paint

M = dict(C, red=(243, 139, 168), teal=(148, 226, 213), sky=(137, 220, 235),
         mauve=(203, 166, 247), pink=(245, 194, 231), base=(30, 30, 46))
FRAMES = {"cpu": M["mauve"], "mem": M["green"], "trees": M["blue"],
          "run": M["teal"], "runs": M["red"]}
ICONS = {"running": ("▶", ">", "green"), "done": ("✓", "+", "green"),
         "failed": ("✗", "x", "red"), "skipped": ("–", "-", "dim"),
         "waiting": ("·", ".", "dim"), "stopped": ("■", "S", "peach"),
         "partial": ("◐", "~", "yellow"), "made": ("·", ".", "dim"),
         "submitted": ("◔", "o", "sky"), "collected": ("✓", "+", "teal"),
         "failed to start": ("✗", "x", "red")}


# ------------------------------------------------------------------ formats
def fmt_bytes(n):
    if n is None:
        return "—"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return ("%d %s" % (n, unit)) if unit == "B" else \
                ("%.2f %s" % (n, unit) if n < 10 else "%.1f %s" % (n, unit))
        n /= 1024.0


def fmt_secs(s):
    if s is None:
        return "—"
    if s < 60:
        return "%.1fs" % s
    s = int(s)
    if s < 3600:
        return "%d:%02d" % (s // 60, s % 60)
    return "%d:%02d:%02d" % (s // 3600, s // 60 % 60, s % 60)


def fmt_count(n):
    if n is None:
        return "—"
    if n >= 1e6:
        return "%.1fM" % (n / 1e6)
    if n >= 1e3:
        return "%.1fk" % (n / 1e3)
    return "%d" % n


# ------------------------------------------------------------------ system
def alive(pid):
    """Whether a process still runs."""
    if not pid:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259          # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except PermissionError:
        return True
    except (OSError, ValueError):
        return False
    return True


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


class System:
    """Samples of the computer: each call of sample() gives rates since the
    call before."""

    def __init__(self, root="."):
        self.root = str(root)
        self.prev_cpu = self.prev_net = None
        self.prev_t = None
        self.prev_proc = {}
        self.hist = deque(maxlen=600)
        self.gpu, self.gpu_t = None, 0.0
        self.has_smi = bool(shutil.which("nvidia-smi"))
        self.name = self._cpu_name()

    @staticmethod
    def _cpu_name():
        for line in _read("/proc/cpuinfo").splitlines():
            if line.startswith("model name"):
                name = line.split(":", 1)[1].strip()
                name = re.sub(r"\(R\)|\(TM\)|CPU|@.*", " ", name)
                return " ".join(name.split())
        import platform
        return platform.processor() or platform.machine()

    def _cpu_times(self):
        """[(busy, total)] for all cores together, then each core."""
        text = _read("/proc/stat")
        if text:
            out = []
            for line in text.splitlines():
                if not line.startswith("cpu"):
                    break
                v = [int(x) for x in line.split()[1:9]]
                idle = v[3] + (v[4] if len(v) > 4 else 0)
                out.append((sum(v) - idle, sum(v)))
            return out
        if os.name == "nt":
            import ctypes
            idle, kern, user = (ctypes.c_ulonglong() for _ in range(3))
            if ctypes.windll.kernel32.GetSystemTimes(
                    ctypes.byref(idle), ctypes.byref(kern),
                    ctypes.byref(user)):
                total = kern.value + user.value
                return [(total - idle.value, total)]
        return None

    @staticmethod
    def _memory():
        text = _read("/proc/meminfo")
        if text:
            kb = {}
            for line in text.splitlines():
                k, _, v = line.partition(":")
                try:
                    kb[k] = int(v.split()[0]) * 1024
                except (IndexError, ValueError):
                    pass
            total = kb.get("MemTotal", 0)
            avail = kb.get("MemAvailable", kb.get("MemFree", 0))
            return {"total": total, "used": total - avail, "avail": avail,
                    "cached": kb.get("Cached", 0),
                    "swap_total": kb.get("SwapTotal", 0),
                    "swap_used": kb.get("SwapTotal", 0) -
                    kb.get("SwapFree", 0)}
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong)
                            ] + [(n, ctypes.c_ulonglong) for n in (
                                "total", "avail", "ptotal", "pavail",
                                "vtotal", "vavail", "xavail")]
            ms = MS()
            ms.len = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return {"total": ms.total, "used": ms.total - ms.avail,
                        "avail": ms.avail, "cached": None,
                        "swap_total": None, "swap_used": None}
        return None

    @staticmethod
    def _uptime():
        text = _read("/proc/uptime")
        if text:
            return float(text.split()[0])
        if os.name == "nt":
            import ctypes
            f = ctypes.windll.kernel32.GetTickCount64
            f.restype = ctypes.c_ulonglong
            return f() / 1000.0
        return None

    @staticmethod
    def _net():
        text = _read("/proc/net/dev")
        if not text:
            return None
        rx = tx = 0
        for line in text.splitlines()[2:]:
            name, _, rest = line.partition(":")
            if name.strip() == "lo":
                continue
            v = rest.split()
            if len(v) >= 9:
                rx += int(v[0])
                tx += int(v[8])
        return rx, tx

    def _gpus(self, now):
        if not self.has_smi:
            return None
        if now - self.gpu_t >= 2.0:
            self.gpu_t = now
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,utilization.gpu,"
                     "memory.used,memory.total", "--format=csv,noheader,"
                     "nounits"], capture_output=True, text=True,
                    timeout=3).stdout
                self.gpu = []
                for line in out.strip().splitlines():
                    name, util, used, total = [x.strip()
                                               for x in line.split(",")]
                    self.gpu.append({"name": name, "util": float(util),
                                     "used": float(used) * 2 ** 20,
                                     "total": float(total) * 2 ** 20})
            except (OSError, ValueError, subprocess.SubprocessError):
                self.gpu = None
        return self.gpu

    @staticmethod
    def _proc(pid):
        """(CPU ticks, resident bytes) of a process, on Linux."""
        stat = _read("/proc/%d/stat" % pid)
        if not stat:
            return None
        rest = stat.rsplit(")", 1)[1].split()
        rss = _read("/proc/%d/statm" % pid).split()
        try:
            return (int(rest[11]) + int(rest[12]),
                    int(rss[1]) * os.sysconf("SC_PAGE_SIZE"))
        except (IndexError, ValueError, OSError, AttributeError):
            return None

    def sample(self, pids=()):
        now = time.monotonic()
        dt = (now - self.prev_t) if self.prev_t else None
        cpu = self._cpu_times()
        pct = []
        if cpu and self.prev_cpu and len(cpu) == len(self.prev_cpu):
            for (b, t), (pb, pt) in zip(cpu, self.prev_cpu):
                pct.append(max(0.0, min(100.0, 100.0 * (b - pb) /
                                        max(1, t - pt))))
        load = os.getloadavg() if hasattr(os, "getloadavg") else None
        if not pct and not cpu and load:          # no counters: from load
            pct = [min(100.0, 100.0 * load[0] / (os.cpu_count() or 1))]
        self.prev_cpu = cpu or self.prev_cpu
        total = pct[0] if pct else None
        if total is not None:
            self.hist.append(total)
        net, rates = self._net(), None
        if net and self.prev_net and dt:
            rates = ((net[0] - self.prev_net[0]) / dt,
                     (net[1] - self.prev_net[1]) / dt)
        self.prev_net = net
        procs = {}
        tck = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        for pid in pids:
            got = self._proc(pid)
            if got is None:
                continue
            prev = self.prev_proc.get(pid)
            procs[pid] = {"rss": got[1], "cpu": (
                100.0 * (got[0] - prev) / tck / dt) if prev is not None
                and dt else None}
            self.prev_proc[pid] = got[0]
        try:
            du = shutil.disk_usage(self.root)
            disk = {"total": du.total, "used": du.used, "free": du.free}
        except OSError:
            disk = None
        self.prev_t = now
        return {"cpu": total, "cores": pct[1:], "name": self.name,
                "count": os.cpu_count(), "load": load,
                "uptime": self._uptime(), "mem": self._memory(),
                "disk": disk, "net": {"rx": rates[0], "tx": rates[1],
                                      "rx_total": net[0],
                                      "tx_total": net[1]} if rates else (
                    {"rx": None, "tx": None, "rx_total": net[0],
                     "tx_total": net[1]} if net else None),
                "gpu": self._gpus(now), "procs": procs,
                "history": list(self.hist)}


# ----------------------------------------------------------------- project
def _when(text):
    try:
        return time.mktime(time.strptime(text, "%Y-%m-%d %H:%M:%S"))
    except (TypeError, ValueError):
        return None


def summary(rec, live=False):
    """One line's worth of a run record (finished or live)."""
    steps = rec.get("steps") or []
    status = rec.get("status", "?")
    if live and status == "running" and not alive(rec.get("pid")):
        status = "stopped"
    now = time.time()
    if rec.get("seconds") is not None:
        secs = rec["seconds"]
    elif rec.get("t0"):
        secs = now - rec["t0"]
    else:
        a, b = _when(rec.get("started")), _when(rec.get("finished"))
        secs = (b - a) if a and b else rec.get("usage", {}).get("seconds")
    cur = [p for p in steps if p.get("status") == "running"]
    done = sum(1 for p in steps if p.get("status") in
               ("done", "failed", "skipped"))
    u = rec.get("usage") or {}
    rnd = rec.get("round")
    return {"kind": "run", "pid": rec.get("pid"), "name": rec.get("name", "?"),
            "status": status, "live": live, "steps": len(steps),
            "done": done, "current": cur, "seconds": secs,
            "calls": u.get("calls", 0), "in": u.get("in", 0),
            "out": u.get("out", 0), "retries": u.get("retries", 0),
            "fallbacks": u.get("fallbacks", 0),
            "cache": rec.get("from_cache", 0),
            "backend": rec.get("backend", ""), "model": rec.get("model", ""),
            "round": rnd, "started": rec.get("started", ""),
            "trees": [t.get("title", "") for t in rec.get("trees", [])],
            "handoffs": rec.get("handoffs", 0), "record": rec}


def _job_progress(jdir):
    """The last line of a job's log, and a percentage if it shows one."""
    f = jdir / "log.txt"
    try:
        with open(f, "rb") as fh:
            fh.seek(max(0, f.stat().st_size - 4096))
            tail = fh.read().decode("utf-8", "replace")
    except OSError:
        return "", None, []
    lines = [x.strip() for x in re.split(r"[\r\n]+", tail) if x.strip()]
    last = lines[-1] if lines else ""
    pct = None
    for line in reversed(lines):
        m = re.search(r"(\d{1,3})%\|", line)
        if m:
            pct = float(m.group(1))
            break
    return last, pct, lines[-12:]


class Project:
    """The project's runs, jobs and files, read again on each refresh (JSON
    files only when they changed)."""

    def __init__(self, root, keep=30):
        self.root = Path(root)
        self.keep = keep
        self.cache = {}
        self.store, self.store_t = None, 0.0
        self.error = ""

    def _json(self, f):
        try:
            mt = f.stat().st_mtime_ns
        except OSError:
            return None
        hit = self.cache.get(f)
        if hit and hit[0] == mt:
            return hit[1]
        try:
            data = json.loads(f.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            return hit[1] if hit else None
        self.cache[f] = (mt, data)
        return data

    def tree(self):
        """The Store, loaded again at most every 10 seconds."""
        now = time.monotonic()
        if self.store is None or now - self.store_t > 10:
            self.store_t = now
            import contextlib
            import io
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    self.store = Store(self.root)
                self.error = ""
            except (Exception, SystemExit):
                self.store = None
                self.error = ""
        return self.store

    def live(self):
        d = self.root / "history" / "live"
        out = []
        for f in sorted(d.glob("*.json")) if d.is_dir() else []:
            rec = self._json(f)
            if isinstance(rec, dict):
                out.append(summary(rec, live=True))
        return sorted(out, key=lambda r: -(r["record"].get("t0") or 0))

    def records(self, limit=300):
        d = self.root / "history"
        if not d.is_dir():
            return []
        files = sorted(d.glob("production-*.json"),
                       key=lambda f: f.stat().st_mtime_ns)[-limit:]
        out = []
        for f in files:
            rec = self._json(f)
            if isinstance(rec, dict):
                out.append(rec)
        return out

    def jobs(self, limit=8):
        d = self.root / "models" / "jobs"
        if not d.is_dir():
            return []
        out = []
        for jdir in sorted(x for x in d.iterdir() if x.is_dir())[-limit:]:
            rec = self._json(jdir / "job.json") or {}
            status = rec.get("status", "made")
            if status == "running" and not (jdir / "exit").is_file():
                pid = _read(jdir / "pid").strip()
                if pid.isdigit() and not alive(int(pid)):
                    status = "stopped"
            elif status in ("running", "submitted") and \
                    (jdir / "exit").is_file():
                code = _read(jdir / "exit").strip()
                status = "done" if code == "0" else "failed"
            last, pct, tail = _job_progress(jdir)
            t0 = _when(rec.get("started"))
            secs = (time.time() - t0) if t0 and status == "running" else None
            out.append({"kind": "job", "pid": None, "id": rec.get("id",
                                                                   jdir.name),
                        "name": rec.get("name", jdir.name), "status": status,
                        "runtime": rec.get("runtime", ""), "pct": pct,
                        "last": last, "tail": tail, "seconds": secs,
                        "started": rec.get("started", ""),
                        "command": " ".join(rec.get("command", [])),
                        "record": rec})
        return list(reversed(out))

    def snapshot(self):
        recs = self.records()
        live = self.live()
        recent = [summary(r) for r in reversed(recs[-self.keep:])]
        stats = {}
        for rec in recs + [x["record"] for x in live]:
            for p in rec.get("steps") or []:
                st = p.get("status")
                s = stats.setdefault(p.get("id"), {
                    "runs": 0, "fails": 0, "seconds": 0.0, "last": "",
                    "running": False})
                if st == "running" and rec.get("pid") and \
                        alive(rec.get("pid")):
                    s["running"] = True
                if st not in ("done", "failed"):
                    continue
                s["runs"] += 1
                s["fails"] += st == "failed"
                s["seconds"] += p.get("seconds") or (
                    p.get("usage") or {}).get("seconds", 0)
                s["last"] = st
        jobs = self.jobs()
        rows = live + [j for j in jobs if j["status"] in (
            "running", "submitted")] + recent + [
            j for j in jobs if j["status"] not in ("running", "submitted")]
        return {"rows": rows, "live": live, "recent": recent, "jobs": jobs,
                "stats": stats}


# ---------------------------------------------------------------- drawing
class Draw:
    """Panels drawn as lines of exactly the given display width."""

    def __init__(self, paint, fancy):
        self.p, self.fancy = paint, fancy
        self.ch = dict(zip("tl tr bl br h v".split(),
                           "╭ ╮ ╰ ╯ ─ │".split() if fancy else
                           "+ + + + - |".split()))

    def fit(self, segs, w, bg=None):
        """Segments (text or (text, colour[, bold])) cut or padded to w."""
        out, used = [], 0
        for seg in segs:
            if isinstance(seg, str):
                text, fg, bold = seg, None, False
            else:
                text, fg, bold = (tuple(seg) + (None, False))[:3]
            buf, full = "", False
            for ch in text:
                cw = display_width(ch)
                if used + cw > w:
                    full = True
                    break
                buf += ch
                used += cw
            if full and buf and w > 1:
                used -= display_width(buf[-1])
                buf = buf[:-1]
                if used < w:
                    buf += "…"
                    used += 1
            if buf:
                out.append(self.p(buf, fg=M.get(fg, fg), bg=bg, bold=bold))
            if full:
                break
        return "".join(out) + self.p(" " * max(0, w - used), bg=bg)

    def box(self, name, lines, w, h, left=(), right=(), bottom=(),
            colour=None):
        """A rounded box with a title on its top line: lines are lists of
        segments; returns h strings of width w."""
        c = self.ch
        fc = colour or M["frame"]
        f = (lambda t: self.p(t, fg=fc))

        def label(segs):
            return [(" ", None)] + list(segs) + [(" ", None)]
        title = label([(name, "text", True)] + ([("  ", None)] + list(left)
                                                if left else []))
        rt = label(right) if right else []
        tw = sum(display_width(s[0]) for s in title)
        rw = sum(display_width(s[0]) for s in rt)
        fill = w - 3 - tw - rw - (1 if rt else 0)
        if fill < 1:
            rt, rw, fill = [], 0, max(0, w - 3 - tw)
        top = f(c["tl"] + c["h"]) + self.fit(title, min(tw, w - 3)) + \
            f(c["h"] * fill) + (self.fit(rt, rw) + f(c["h"]) if rt else "") \
            + f(c["tr"])
        out = [top]
        inner = w - 2
        for i in range(h - 2):
            segs = lines[i] if i < len(lines) else []
            if isinstance(segs, tuple) and len(segs) == 2 and \
                    segs[0] == "BG":
                body = self.fit(segs[1][1], inner, bg=segs[1][0])
            else:
                body = self.fit(segs, inner)
            out.append(f(c["v"]) + body + f(c["v"]))
        if bottom:
            bt = label(bottom)
            bw = min(sum(display_width(s[0]) for s in bt), w - 4)
            out.append(f(c["bl"] + c["h"]) + self.fit(bt, bw) +
                       f(c["h"] * (w - 3 - bw) + c["br"]))
        else:
            out.append(f(c["bl"] + c["h"] * (w - 2) + c["br"]))
        return out[:h]

    def meter(self, pct, w, solid=None):
        """A bar of small squares coloured green to red (or one colour)."""
        if w <= 0:
            return []
        pct = 0.0 if pct is None else max(0.0, min(100.0, pct))
        n = int(round(pct / 100.0 * w))
        on, off = ("■", "■" if self.p.mode else "·") if self.fancy \
            else ("#", ".")
        segs = []
        for i in range(w):
            if i < n:
                segs.append((on, solid or grade(i / max(1, w - 1))))
            else:
                segs.append((off, "badge"))
        return segs

    def graph(self, values, w, h):
        """A filled graph of the last values (0 to 100) in braille dots,
        two samples per column; rows of segments."""
        if w <= 0 or h <= 0:
            return []
        per = 2 if self.fancy else 1
        vals = list(values)[-w * per:]
        vals = [None] * (w * per - len(vals)) + vals
        dots = h * 4 if self.fancy else h
        lv = [None if v is None else int(round(v / 100.0 * dots))
              for v in vals]
        if lv and lv[-1] == 0 and vals[-1]:
            lv[-1] = 1
        rows = []
        left = (0x40, 0x04, 0x02, 0x01)
        right = (0x80, 0x20, 0x10, 0x08)
        for r in range(h):
            colour = grade(1 - r / max(1, h - 1))
            line = ""
            for c in range(w):
                if self.fancy:
                    bits, base = 0, (h - 1 - r) * 4
                    for k, bitset in ((0, left), (1, right)):
                        v = lv[2 * c + k]
                        if v is None:
                            continue
                        for d in range(max(0, min(4, v - base))):
                            bits |= bitset[d]
                    line += chr(0x2800 + bits) if bits else " "
                else:
                    v = lv[c]
                    line += "#" if v is not None and v > h - 1 - r else " "
            rows.append([(line, colour)])
        return rows


def grade(x):
    """green, then yellow, then red, for 0 to 1."""
    x = max(0.0, min(1.0, x))
    a, b, c = M["green"], M["yellow"], M["red"]
    if x < 0.5:
        t, lo, hi = x / 0.5, a, b
    else:
        t, lo, hi = (x - 0.5) / 0.5, b, c
    return tuple(int(lo[i] + (hi[i] - lo[i]) * t) for i in range(3))


def icon(status, fancy):
    i = ICONS.get(status, ("?", "?", "dim"))
    return (i[0] if fancy else i[1]), i[2]


def _rjust(text, w):
    return " " * max(0, w - display_width(text)) + text


def _cut(text, w):
    if display_width(text) <= w:
        return text
    while text and display_width(text) > w - 1:
        text = text[:-1]
    return text + "…"


def _ljust(text, w):
    text = _cut(text, w)
    return text + " " * max(0, w - display_width(text))


# ----------------------------------------------------------------- panels
def cpu_panel(d, s, w, h, interval):
    inner = h - 2
    rp = max(20, min(34, w // 3))
    gw = w - 2 - rp - 1
    side = []
    side.append([(_ljust(s["name"] or "cpu", rp - 8), "sub"),
                 (_rjust("%d cores" % (s["count"] or 0), 8), "dim")])
    mw = max(4, rp - 10)
    side.append([("CPU ", "text", True)] + d.meter(s["cpu"], mw) +
                 [(_rjust("%s%%" % ("—" if s["cpu"] is None else
                                    "%.0f" % s["cpu"]), 6), "text")])
    cores = s["cores"]
    room = inner - 3
    if cores and room > 0:
        cols = 1 if len(cores) <= room else 2
        cw = rp // cols
        bw = max(2, cw - 10)
        per = (len(cores) + cols - 1) // cols
        for r in range(min(per, room)):
            line = []
            for cidx in range(cols):
                i = r + cidx * per
                if i < len(cores):
                    line += [(_ljust("C%d" % i, 4), "sub")] + d.meter(
                        cores[i], bw) + [(_rjust("%.0f%%" % cores[i], 5),
                                          "dim"), (" ", None)]
            side.append(line)
        if per > room:
            side.append([("… %d more" % (len(cores) - room * cols), "dim")])
    if s["load"]:
        side += [[]] * max(0, inner - len(side) - 1)
        side.append([("LAV: ", "sub"), (" ".join("%.2f" % x for x in
                                                  s["load"]), "text")])
    graph = d.graph(s["history"], gw, inner)
    up = "up " + fmt_secs(s["uptime"]) if s["uptime"] is not None else ""
    lines = []
    for i in range(inner):
        g = graph[i] if i < len(graph) else [(" " * gw, None)]
        if i == inner - 1 and up:
            text = g[0][0]
            g = [(up, "sub"), (text[len(up):], g[0][1])]
        lines.append(g + [(" ", None)] + (side[i] if i < len(side) else []))
    return d.box("cpu", lines, w, h, colour=FRAMES["cpu"],
                 right=[(time.strftime("%H:%M:%S"), "text"), ("  ", None),
                        ("%dms" % int(interval * 1000), "dim")])


def mem_panel(d, s, project, w):
    iw = w - 2
    bw = max(4, iw - 26)
    lines = []
    m = s["mem"]
    if m:
        lines.append([("Total:", "text", True),
                      (_rjust(fmt_bytes(m["total"]), iw - 6), "text")])
        for label, key in (("Used", "used"), ("Avail", "avail")):
            pct = 100.0 * m[key] / max(1, m["total"])
            lines.append([(_ljust(label, 6), "sub")] + d.meter(
                pct, bw, "green" if key == "avail" else None) +
                         [(_rjust(fmt_bytes(m[key]), 11), "text"),
                          (_rjust("%.0f%%" % pct, 5), "dim")])
        if m.get("swap_total"):
            pct = 100.0 * m["swap_used"] / m["swap_total"]
            lines.append([(_ljust("Swap", 6), "sub")] + d.meter(pct, bw) +
                         [(_rjust(fmt_bytes(m["swap_used"]), 11), "text"),
                          (_rjust("%.0f%%" % pct, 5), "dim")])
    else:
        lines.append([("memory: not readable on this system", "dim")])
    if s["disk"]:
        k = s["disk"]
        pct = 100.0 * k["used"] / max(1, k["total"])
        lines.append([(_ljust("Disk", 6), "sub")] + d.meter(pct, bw) +
                     [(_rjust(fmt_bytes(k["free"]), 11), "text"),
                      (_rjust("free", 5), "dim")])
    for g in s["gpu"] or []:
        lines.append([(_ljust("GPU", 6), "sub")] + d.meter(g["util"], bw) +
                     [(_rjust(fmt_bytes(g["used"]), 11), "text"),
                      (_rjust("%.0f%%" % g["util"], 5), "dim")])
    n = s["net"]
    if n:
        lines.append([("Net ", "sub"), ("▼ " if d.fancy else "v ", "green"),
                      (fmt_bytes(n["rx"]) + "/s" if n["rx"] is not None
                       else "—", "text"), ("   ", None),
                      ("▲ " if d.fancy else "^ ", "peach"),
                      (fmt_bytes(n["tx"]) + "/s" if n["tx"] is not None
                       else "—", "text")])
    procs = s["procs"]
    if project and project["live"]:
        cpu = sum((procs.get(r["pid"]) or {}).get("cpu") or 0
                  for r in project["live"])
        rss = sum((procs.get(r["pid"]) or {}).get("rss") or 0
                  for r in project["live"])
        lines.append([("Runs", "sub", True), ("  %d live" % len(
            project["live"]), "green"), ("   CPU %.1f%%  MEM %s" % (
                cpu, fmt_bytes(rss)) if procs else "", "text")])
    return lines


def trees_panel(d, store, stats, w, h, scroll, error=""):
    iw = w - 2
    rows = []
    if store is None:
        return d.box("trees", [[("not a Tawjih project here%s" % (
            ": " + error if error else ""), "dim")]], w, h,
            colour=FRAMES["trees"]), 0
    nodes = []

    def walk(pid):
        for n in store.children(pid):
            nodes.append(n)
            walk(n.id)
    walk("")
    if not nodes:
        return d.box("trees", [[("no files yet", "dim")]], w, h,
                     colour=FRAMES["trees"]), 0
    pre = tree_lines([(n.id, n.parent if n.parent in store.nodes else "")
                      for n in nodes], d.fancy)
    show_code = iw >= 58
    cw = 8 if show_code else 0
    stat_w = 18
    nos = {n.id: pre[n.id] + store.number(n.id) for n in nodes}
    nw = min(max(display_width(x) for x in nos.values()),
             max(8, int(iw * 0.45)))
    tw = max(6, iw - cw - nw - stat_w - 3)
    longest = max((s["seconds"] / s["runs"]) for s in stats.values()
                  if s["runs"]) if any(s["runs"] for s in stats.values()) \
        else 0
    head = [(" ", None), (_ljust("Code", cw), "dim") if cw else ("", None),
            (_ljust("No.", nw), "dim"), (" ", None),
            (_rjust("Title", tw), "dim"), (" ", None),
            (_rjust("Runs", 5), "dim"),
            (_rjust("OK", 5), "dim"), (_rjust("Avg", 8), "dim")]
    for n in nodes:
        s = stats.get(n.id)
        if s and s["running"]:
            ic, col = icon("running", d.fancy)
        elif s and s["last"]:
            ic, col = icon(s["last"], d.fancy)
        else:
            ic, col = (" ", None)
        line = [(ic, col)]
        if cw:
            line.append((_ljust(n.id, cw), "dim"))
        line += [(_ljust(nos[n.id], nw), "sub"), (" ", None),
                 (_rjust(_cut(n.title, tw), tw), "text" if s else "sub",
                  bool(s and s["running"])), (" ", None)]
        if s and s["runs"]:
            ok = 100.0 * (s["runs"] - s["fails"]) / s["runs"]
            avg = s["seconds"] / s["runs"]
            line += [(_rjust(str(s["runs"]), 5), "text"),
                     (_rjust("%.0f%%" % ok, 5), "green" if ok >= 99.5 else
                      ("yellow" if ok >= 70 else "red")),
                     (_rjust(fmt_secs(avg), 8),
                      grade(avg / longest if longest else 0))]
        else:
            line += [(_rjust("·" if d.fancy else ".", 5), "badge")]
        rows.append((n.id, line))
    room = h - 3
    running = [i for i, (nid, _) in enumerate(rows)
               if stats.get(nid, {}).get("running")]
    if scroll is None:
        scroll = max(0, running[0] - room // 2) if running else 0
    scroll = max(0, min(scroll, len(rows) - room))
    body = [head] + [r[1] for r in rows[scroll:scroll + room]]
    ran = sum(1 for n in nodes if stats.get(n.id, {}).get("runs"))
    return d.box("trees", body, w, h, colour=FRAMES["trees"],
                 left=[("%d files" % len(nodes), "dim")],
                 right=[("%d ran" % ran, "sub")],
                 bottom=[("%d-%d of %d" % (scroll + 1, min(
                     len(rows), scroll + room), len(rows)), "dim"),
                     ("  PgUp PgDn", "dim")]
                 if len(rows) > room else ()), scroll


def detail_panel(d, s, row, w, h):
    iw = w - 2
    if row is None:
        return d.box("run", [[("no run yet: tawjih run PATH --input "
                               "TEXT", "dim")]], w, h, colour=FRAMES["run"])
    ic, col = icon(row["status"], d.fancy)
    if row["kind"] == "job":
        lines = [[("Status: ", "sub"), (ic + " " + row["status"], col),
                  ("   Runtime: ", "sub"), (row["runtime"], "text"),
                  ("   Started: ", "sub"), (row["started"] or "—", "text")]]
        if row["pct"] is not None:
            lines.append([("Progress ", "sub")] + d.meter(
                row["pct"], max(4, iw - 16)) + [(_rjust(
                    "%.0f%%" % row["pct"], 6), "text")])
        lines.append([("$ " + row["command"], "dim")])
        for t in row["tail"][-(h - 2 - len(lines)):]:
            lines.append([(t, "text")])
        return d.box("job " + row["id"], lines, w, h, colour=FRAMES["run"],
                     left=[(row["name"], "accent")],
                     right=[(ic + " " + row["status"], col)])
    procs = s["procs"].get(row["pid"]) or {}
    lines = [[("Status: ", "sub"), (ic + " " + row["status"], col, True),
              ("   Elapsed: ", "sub"), (fmt_secs(row["seconds"]), "text"),
              ("   Steps: ", "sub"), ("%d/%d" % (row["done"], row["steps"]),
                                      "text")] +
             ([("   Round: ", "sub"), ("%d/%d" % tuple(row["round"]),
                                       "text")] if row["round"] else []),
             [("Calls: ", "sub"), (str(row["calls"]), "text"),
              ("   Tokens ", "sub"), ("in ", "dim"),
              (fmt_count(row["in"]), "text"), (" out ", "dim"),
              (fmt_count(row["out"]), "text"),
              ("   Retries ", "sub"), (str(row["retries"]), "text"),
              ("  Fallbacks ", "sub"), (str(row["fallbacks"]), "text"),
              ("  Cache ", "sub"), (str(row["cache"]), "text")],
             [("Backend: ", "sub"), (row["backend"] or "—", "text"),
              ("   Model: ", "sub"), (row["model"] or "—", "text")] +
             ([("   CPU ", "sub"), ("%.1f%%" % procs["cpu"], "text"),
               ("  MEM ", "sub"), (fmt_bytes(procs["rss"]), "text")]
              if procs.get("cpu") is not None else [])]
    steps = row["record"].get("steps") or []
    now = time.time()

    def took(p):
        if p.get("status") == "running" and p.get("started_at"):
            return now - p["started_at"]
        if p.get("seconds") is not None:
            return p["seconds"]
        return (p.get("usage") or {}).get("seconds")
    times = [took(p) or 0 for p in steps]
    longest = max(times) if times else 0
    nw = max([display_width(p.get("number", "")) for p in steps] + [3])
    bar = max(4, min(24, iw // 4))
    tw = max(6, iw - 3 - 4 - 8 - nw - 2 - bar - 8 - 2)
    lines.append([(" ", None), (_rjust("#", 3), "dim"), (" ", None),
                  (_ljust("Code", 8), "dim"), (_ljust("No.", nw + 1), "dim"),
                  (_ljust("Title", tw), "dim"), (_ljust("Time", bar + 9),
                                                 "dim")])
    room = h - 2 - len(lines)
    cur = [i for i, p in enumerate(steps) if p.get("status") == "running"]
    first = max(0, min((cur[0] - room // 2) if cur else 0,
                       len(steps) - room))
    for i, p in enumerate(steps[first:first + room], first):
        st = p.get("status", "waiting")
        ic2, c2 = icon(st, d.fancy)
        t = times[i]
        segs = [(ic2, c2), (_rjust(str(p.get("step", i + 1)), 3), "sub"),
                (" ", None), (_ljust(p.get("id", ""), 8), "dim"),
                (_ljust(p.get("number", ""), nw + 1), "sub"),
                (_ljust(p.get("title", ""), tw), "text",
                 st == "running")]
        if st in ("running", "done", "failed") and t:
            segs += d.meter(100.0 * t / longest if longest else 0, bar)
            segs += [(_rjust(fmt_secs(t), 8), c2 if st != "done"
                      else "text")]
        lines.append(segs)
    title = "%s %s" % (row["pid"] or "", row["name"])
    trees = " · ".join(row["trees"])
    return d.box(title.strip(), lines, w, h, colour=FRAMES["run"],
                 left=[(trees, "accent")] if trees else (),
                 right=[(ic + " " + row["status"], col)])


def runs_panel(d, rows, sel, w, h, interval):
    iw = w - 2
    nw = max(8, iw - 6 - 1 - 10 - 7 - 8 - 6 - 13 - 2 - 10)
    head = [(_rjust("PID", 6), "dim"), (" ", None), (_ljust("Path", nw),
                                                     "dim"),
            (_ljust("Status", 10), "dim"), (_rjust("Step", 7), "dim"),
            (_rjust("Time", 8), "dim"), (_rjust("Calls", 6), "dim"),
            (_rjust("Tokens", 13), "dim"), ("  ", None),
            (_ljust("Backend", 10), "dim")]
    lines = [head]
    room = h - 3
    first = max(0, min(sel - room // 2, len(rows) - room))
    for i, r in enumerate(rows[first:first + room], first):
        ic, col = icon(r["status"], d.fancy)
        if r["kind"] == "job":
            segs = [(_rjust("job", 6), "sky"), (" ", None),
                    (_ljust(r["name"], nw), "text"),
                    (_ljust(ic + " " + r["status"], 10), col),
                    (_rjust("%.0f%%" % r["pct"] if r["pct"] is not None
                            else "", 7), "text"),
                    (_rjust(fmt_secs(r["seconds"]) if r["seconds"] else "",
                            8), "text"), (_rjust("", 6), None),
                    (_rjust("", 13), None), ("  ", None),
                    (_ljust(r["runtime"], 10), "sub")]
        else:
            segs = [(_rjust(str(r["pid"] or ""), 6), "dim" if not r["live"]
                     else "text"), (" ", None),
                    (_ljust(r["name"], nw), "text", r["live"]),
                    (_ljust(ic + " " + r["status"], 10), col),
                    (_rjust("%d/%d" % (r["done"], r["steps"]), 7), "text"),
                    (_rjust(fmt_secs(r["seconds"]), 8), "text"),
                    (_rjust(str(r["calls"]), 6), "text"),
                    (_rjust("%s/%s" % (fmt_count(r["in"]),
                                       fmt_count(r["out"])), 13), "sub"),
                    ("  ", None), (_ljust(r["backend"], 10), "sub")]
        if i == sel:
            lines.append(("BG", (M["badge"], segs)))
        else:
            lines.append(segs)
    if not rows:
        lines.append([("no runs yet", "dim")])
    live = sum(1 for r in rows if r.get("live") and r["status"] == "running")
    return d.box("runs", lines, w, h, colour=FRAMES["runs"],
                 left=[("%d live" % live, "green" if live else "dim")],
                 right=[("%d shown" % len(rows), "dim")],
                 bottom=[("↑↓ select" if d.fancy else "up/down select",
                          "sub"), ("  PgUp PgDn trees  +/- %dms  q quit"
                                   % int(interval * 1000), "dim")])


def side_by_side(a, b):
    return [x + y for x, y in zip(a, b)]


def render(sysinfo, project, store, w, h, sel=0, scroll=None, interval=2.0,
           paint=None, fancy=True, error=""):
    """The board as h lines of width w. Returns (lines, tree scroll)."""
    d = Draw(paint or Paint(None), fancy)
    w, h = max(60, w), max(18, h)
    h1 = max(6, min(12, int(h * 0.3)))
    rest = h - h1
    wl = max(32, int(w * 0.40))
    wr = w - wl
    mem = mem_panel(d, sysinfo, project, wl)
    h2 = min(len(mem) + 2, max(4, rest - 5))
    left = d.box("mem", mem, wl, h2, colour=FRAMES["mem"],
                 left=[("disk", "sub"), ("·", "dim"), ("net", "sub")])
    tree, scroll = trees_panel(d, store, project["stats"] if project else {},
                               wl, rest - h2, scroll, error)
    left += tree
    rows = project["rows"] if project else []
    sel = max(0, min(sel, len(rows) - 1))
    row = rows[sel] if rows else None
    nsteps = len((row or {}).get("record", {}).get("steps") or []) \
        if row and row["kind"] == "run" else len((row or {}).get("tail", []))
    h3 = max(8, min(nsteps + 7, int(rest * 0.62)))
    right = detail_panel(d, sysinfo, row, wr, h3) + \
        runs_panel(d, rows, sel, wr, rest - h3, interval)
    lines = cpu_panel(d, sysinfo, w, h1, interval) + side_by_side(left, right)
    return lines, scroll


# ------------------------------------------------------------------- keys
class Keys:
    """Single key presses without Enter, on Linux, macOS and Windows."""

    def __enter__(self):
        self.old = None
        if os.name != "nt" and sys.stdin.isatty():
            import termios
            import tty
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        if self.old is not None:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    NAMES = {"\x1b[A": "up", "\x1b[B": "down", "\x1b[5~": "pgup",
             "\x1b[6~": "pgdn", "\x1b[H": "home", "\x1b[1~": "home",
             "\x1bOA": "up", "\x1bOB": "down", "k": "up", "j": "down",
             "\x1b": "q", "\x03": "q"}
    WIN = {"H": "up", "P": "down", "I": "pgup", "Q": "pgdn", "G": "home"}

    def get(self, timeout):
        if os.name == "nt":
            import msvcrt
            end = time.monotonic() + timeout
            while time.monotonic() < end:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\x00", "\xe0"):
                        return self.WIN.get(msvcrt.getwch())
                    return self.NAMES.get(ch, ch.lower())
                time.sleep(0.04)
            return None
        import select
        if not sys.stdin.isatty():
            time.sleep(timeout)
            return None
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        if not r:
            return None
        ch = os.read(self.fd, 16).decode("utf-8", "ignore")
        return self.NAMES.get(ch, ch[:1].lower())


def _enable_vt():
    if os.name == "nt":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            h = k.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            if k.GetConsoleMode(h, ctypes.byref(mode)):
                k.SetConsoleMode(h, mode.value | 0x0004)
        except (OSError, AttributeError):
            pass


def snapshot_json(root):
    sysm = System(root)
    proj = Project(root)
    sysm.sample()
    time.sleep(0.3)
    snap = proj.snapshot()
    s = sysm.sample([r["pid"] for r in snap["live"] if r.get("pid")])
    s.pop("history", None)

    def clean(r):
        return {k: v for k, v in r.items() if k not in ("record", "tail")}
    return {"system": s, "live": [clean(r) for r in snap["live"]],
            "recent": [clean(r) for r in snap["recent"]],
            "jobs": [clean(j) for j in snap["jobs"]],
            "files": snap["stats"]}


def cmd_top(a):
    """A live board of runs on paths and files beside the computer's
    performance. q quits, up/down choose a run, PgUp/PgDn move the trees,
    + and - change the refresh time."""
    root = a.project
    if getattr(a, "json", False):
        print(json.dumps(snapshot_json(root), ensure_ascii=False, indent=2))
        return 0
    colour = getattr(a, "color", "auto")
    paint = Paint.detect() if colour == "auto" else Paint(
        None if colour == "never" else "true")
    fancy = _fancy() and not getattr(a, "ascii", False)
    interval = max(0.25, float(getattr(a, "interval", 2.0) or 2.0))
    sysm, proj = System(root), Project(root)
    once = getattr(a, "once", False) or not sys.stdout.isatty()
    if once:
        sysm.sample()
        for _ in range(4):                       # a short CPU history
            time.sleep(0.15)
            sysm.sample()
        snap = proj.snapshot()
        s = sysm.sample([r["pid"] for r in snap["live"] if r.get("pid")])
        size = shutil.get_terminal_size((110, 34))
        lines, _ = render(s, snap, proj.tree(), a.width or size.columns,
                          a.height or size.lines - 1, 0, None, interval,
                          paint, fancy, proj.error)
        print("\n".join(lines))
        return 0
    _enable_vt()
    out = sys.stdout
    sel, scroll, last, s, snap = 0, None, 0.0, None, None
    size = None
    out.write("\033[?1049h\033[?25l\033[2J")
    try:
        with Keys() as keys:
            sysm.sample()
            while True:
                now = time.monotonic()
                if s is None or now - last >= interval:
                    snap = proj.snapshot()
                    s = sysm.sample([r["pid"] for r in snap["live"]
                                     if r.get("pid")])
                    last = now
                ts = shutil.get_terminal_size((110, 34))
                if ts != size:
                    size = ts
                    out.write("\033[2J")
                w = ts.columns - (1 if os.name == "nt" else 0)
                lines, scroll = render(s, snap, proj.tree(), w, ts.lines,
                                       sel, scroll, interval, paint, fancy,
                                       proj.error)
                out.write("\033[H" + "\r\n".join(lines[:ts.lines]))
                out.flush()
                key = keys.get(max(0.05, interval - (time.monotonic() -
                                                     last)))
                if key in ("q", "ض"):
                    break
                rows = len(snap["rows"])
                if key == "up":
                    sel = max(0, sel - 1)
                elif key == "down":
                    sel = min(max(0, rows - 1), sel + 1)
                elif key == "pgup":
                    scroll = max(0, (scroll or 0) - 8)
                elif key == "pgdn":
                    scroll = (scroll or 0) + 8
                elif key == "home":
                    sel, scroll = 0, None
                elif key in ("+", "="):
                    interval = min(10.0, interval + 0.5)
                elif key in ("-", "_"):
                    interval = max(0.5, interval - 0.5)
    except KeyboardInterrupt:
        pass
    finally:
        out.write("\033[0m\033[?25h\033[?1049l")
        out.flush()
    return 0


__all__ = ['Draw', 'Keys', 'Project', 'System', 'alive', 'cmd_top',
           'fmt_bytes', 'fmt_secs', 'render', 'snapshot_json', 'summary']
