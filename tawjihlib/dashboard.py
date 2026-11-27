"""tawjih dashboard (لوحة): the board of `tawjih top` as a web page, served
by Tawjih itself on this computer. Arabic (right to left) or English; the
processor, memory and each GPU's free memory; the project's model servers,
its runs, jobs and files; the chosen run or server; and the servers'
actions, with who did them. Everything on it is read from the project and
the system. Its keys act as in `top`: s start, x stop, r restart, t tune,
f flush, d the batch command.

The page has no outside address: no script, font or style is fetched, so it
works on a computer with no network. The address carries a token, as for
`serve`; without it nothing is answered.

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import json
import os
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .core import Store, who
from .monitor import (
    Project, System, server_action, server_use, watched,
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
                         "detail": " ".join(str(x[k]) for k in (
                             "status", "path", "pid", "changes", "step")
                             if x.get(k) not in (None, "", []))})
    rows += notes
    rows.sort(key=lambda x: x["time"])
    return rows[-LOG_LINES:][::-1]


class Board:
    """What the page shows, sampled at most once a second for all
    viewers, and the actions its keys ask for."""

    def __init__(self, root, min_free=8000, read_only=False, lang=None):
        self.root = str(Path(root).resolve())
        self.lang = lang or "ar"
        self.page = PAGE.replace("__LANG__", self.lang).replace(
            "__DIR__", "rtl" if self.lang == "ar" else "ltr").replace(
            "__FORCED__", "true" if lang else "false")
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


def make_server(board, host, port, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else (
                json.dumps(body, ensure_ascii=False).encode("utf-8")
                if not isinstance(body, str) else body.encode("utf-8"))
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; script-src 'unsafe-inline';"
                             " style-src 'unsafe-inline'; connect-src 'self';"
                             " img-src data:")
            self.end_headers()
            self.wfile.write(data)

        def ok(self):
            return secrets.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + token)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self.send(200, board.page, "text/html")
            if path == "/api/state":
                if not self.ok():
                    return self.send(401, {"error": "missing or wrong token"})
                return self.send(200, board.snapshot())
            self.send(404, {"error": "not found"})

        def do_POST(self):
            if self.path.split("?")[0] != "/api/action":
                return self.send(404, {"error": "not found"})
            if not self.ok():
                return self.send(401, {"error": "missing or wrong token"})
            try:
                n = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(min(n, 65536)) or b"{}")
            except ValueError:
                return self.send(400, {"error": "not JSON"})
            code, msg = board.act(str(req.get("server", "")),
                                  str(req.get("key", "")),
                                  str(req.get("text", ""))[:500])
            self.send(code, {"message": msg})
    return ThreadingHTTPServer((host, port), Handler)


def cmd_dashboard(a):
    """The board of `tawjih top` as a web page on this computer."""
    st = Store(a.project)
    token = a.token or os.environ.get("TAWJIH_TOKEN") or \
        secrets.token_urlsafe(18)
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: %s lets other computers reach the board; anyone "
              "with the token can start and stop servers" % a.host,
              file=sys.stderr)
    board = Board(st.root, a.min_free, a.read_only, a.lang)
    httpd = make_server(board, a.host, a.port, token)
    port = httpd.server_address[1]
    print("tawjih dashboard: http://%s:%d/#t=%s" % (
        "127.0.0.1" if a.host in ("0.0.0.0", "::") else a.host, port, token))
    print("the token is after #t= (the browser does not send that part to "
          "a server); Ctrl-C stops the board%s" % (
              "; read only" if a.read_only else ""))
    sys.stdout.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def add_dashboard_parser(sub):
    s = sub.add_parser("dashboard", aliases=["لوحة"], help="the board of top "
                       "as a web page on this computer, in Arabic or "
                       "English, with no outside address")
    s.add_argument("--host", default="127.0.0.1",
                   help="address to listen on (default: this computer only)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--token", help="the page's token (default: "
                   "$TAWJIH_TOKEN, else a new one)")
    s.add_argument("--min-free", type=int, default=8000, metavar="MIB",
                   help="a GPU with less free memory is shown in red")
    s.add_argument("--read-only", action="store_true",
                   help="show only: the keys that act on servers are off")
    s.add_argument("--lang", choices=["ar", "en"],
                   help="the board's language: ar (Arabic, right to left) "
                   "or en (English); L switches it on the page (default: "
                   "the browser's last choice, else ar)")
    s.set_defaults(fn=cmd_dashboard)
    return s


PAGE = r"""<!doctype html>
<html lang="__LANG__" dir="__DIR__">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tawjih · لوحة التحكم</title>
<style>
:root { --bg:#16161e; --panel:#1a1b26; --line:#292e42; --frame:#414868;
  --text:#c0caf5; --sub:#a9b1d6; --dim:#565f89; --green:#9ece6a;
  --cyan:#7dcfff; --blue:#7aa2f7; --pink:#f7768e; --yellow:#e0af68;
  --mauve:#bb9af7; --orange:#ff9e64; }
* { box-sizing:border-box; }
html, body { margin:0; background:var(--bg); color:var(--text);
  font:13px/1.45 ui-monospace, "DejaVu Sans Mono", Menlo, Consolas,
  "Noto Sans Arabic", "Segoe UI", Tahoma, monospace; }
body { min-height:100vh; display:flex; flex-direction:column; padding:10px;
  gap:10px; }
main { flex:1; display:grid; grid-template-columns:5fr 7fr; gap:12px;
  min-height:0; }
@media (max-width:900px) { main { grid-template-columns:1fr; } }
.col { display:flex; flex-direction:column; gap:14px; min-width:0; }
.box { position:relative; background:var(--panel); border:1px solid
  var(--frame); border-radius:6px; padding:14px 12px 10px; min-width:0; }
.box.g { border-color:var(--green); } .box.c { border-color:var(--cyan); }
.box.b { border-color:var(--blue); } .box.p { border-color:var(--pink); }
.box.y { border-color:#e0af6899; }
.ttl { position:absolute; top:-0.7em; inset-inline-start:10px;
  background:var(--bg); padding:0 6px; border-radius:3px; font-size:12px;
  white-space:nowrap; }
.ttl2 { position:absolute; top:-0.7em; inset-inline-end:10px;
  background:var(--bg); padding:0 6px; border-radius:3px; font-size:12px; }
.ltr { direction:ltr; unicode-bidi:isolate; }
.row { display:flex; align-items:center; gap:8px; }
.lbl { width:7.5em; flex:none; overflow:hidden; text-overflow:ellipsis;
  white-space:nowrap; }
.val { min-width:9em; text-align:end; color:var(--sub); }
.bar { display:flex; gap:2px; direction:ltr; flex:1; }
.bar i { display:inline-block; flex:1; max-width:7px; height:10px;
  border-radius:1px; background:var(--line); }
.sep { border-top:1px solid var(--line); margin:6px 0 4px; }
table { width:100%; border-collapse:collapse; font-size:12px; }
th { color:var(--dim); font-weight:normal; text-align:start; padding:2px 4px;
  border-bottom:1px solid var(--line); }
td { padding:2px 4px; border-bottom:1px solid #292e4255;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
  max-width:22em; }
td.n, th.n { text-align:end; }
tr.sel td { background:var(--line); color:var(--text); font-weight:600; }
tbody tr { cursor:pointer; } tbody tr:hover td { background:#24283b99; }
.chips { display:flex; flex-wrap:wrap; gap:5px; }
.chip { padding:1px 7px; border-radius:4px; background:#24283b;
  font-size:11.5px; cursor:pointer; }
.muted { color:var(--dim); } .ok { color:var(--green); }
.bad { color:var(--pink); } .warn { color:var(--yellow); }
.cy { color:var(--cyan); } .mv { color:var(--mauve); } .bl { color:var(--blue); }
.scroll { overflow:auto; max-height:var(--h, 260px); }
pre.tail { margin:4px 0 0; font:inherit; font-size:11.5px; color:var(--sub);
  white-space:pre-wrap; direction:ltr; text-align:left; max-height:150px;
  overflow:auto; }
footer { display:flex; flex-wrap:wrap; justify-content:space-between;
  gap:8px; color:var(--dim); font-size:12px; border-top:1px solid
  var(--line); padding-top:6px; }
footer .keys { display:flex; flex-wrap:wrap; gap:14px; }
footer .k { cursor:pointer; } footer .k b { color:var(--text); }
footer .k:hover { color:var(--text); }
button { font:inherit; color:var(--cyan); background:#24283b; border:1px
  solid var(--frame); border-radius:4px; padding:1px 10px; cursor:pointer; }
#msg { color:var(--yellow); }
.off { opacity:.45; pointer-events:none; }
</style>
</head>
<body>
<main>
 <div class="col">
  <section class="box g"><div class="ttl" id="tSys"></div>
   <div class="ttl2 muted ltr" id="tSys2"></div>
   <div id="sys"></div><div class="sep"></div><div id="gpus"></div>
   <div class="sep"></div><div id="free" class="row"></div></section>
  <section class="box c"><div class="ttl" id="tSrv"></div>
   <div class="ttl2 muted" id="tSrv2"></div><div id="servers"></div></section>
  <section class="box b" style="flex:1"><div class="ttl" id="tFiles"></div>
   <div class="ttl2 muted" id="tFiles2"></div>
   <div class="scroll" style="--h:340px"><table id="files"></table></div>
  </section>
 </div>
 <div class="col">
  <section class="box c"><div class="ttl" id="tDet"></div>
   <div class="ttl2" id="tDet2"></div><div id="detail"></div></section>
  <section class="box p" style="flex:1"><div class="ttl" id="tRuns"></div>
   <div class="ttl2 muted" id="tRuns2"></div>
   <div class="scroll" style="--h:300px"><table id="runs"></table></div>
  </section>
  <section class="box y"><div class="ttl" id="tLog"></div>
   <div class="scroll" style="--h:120px"><table id="log"></table></div>
  </section>
 </div>
</main>
<footer>
 <div class="keys" id="keys"></div>
 <div class="row"><span id="msg"></span><span class="muted ltr" id="clock">
 </span><button id="lang"></button></div>
</footer>
<script>
"use strict";
const T = {
 ar: {sys:"العتاد والذاكرة", gpusTitle:"البطاقات", minFree:"الحد الأدنى",
  free:"حرة للاختيار التلقائي:", none:"لا شيء", noGpu:"لا بطاقة: nvidia-smi غير موجود",
  srv:"الخوادم", srvN:"خادم", noSrv:"لا خوادم بعد: tawjih server add NAME --engine vllm --model M",
  files:"الأشجار", filesN:"ملف", runs:"التشغيلات", live:"حي", shown:"معروض",
  log:"سجل الأفعال", noLog:"لم يُفعل شيء بعد", det:"التفاصيل", nothing:"اختر صفاً",
  cols:{no:"الرقم", title:"العنوان", runs:"مرات", ok:"نجاح", avg:"المتوسط",
   pid:"PID", path:"المسار / الخادم", status:"الحالة", step:"الخطوة",
   time:"الزمن", calls:"الاستدعاءات", use:"المعالج / الذاكرة", backend:"المزوّد / المحرك",
   at:"الوقت", act:"الفعل", srv:"الخادم", by:"من", detail:"التفاصيل"},
  st:{running:"يعمل", done:"اكتمل", failed:"فشل", stopped:"متوقف", partial:"جزئي",
   waiting:"ينتظر", skipped:"تُخطّي", made:"أُعدّ", submitted:"أُرسل", collected:"جُمع"},
  l:{status:"الحالة", up:"يعمل منذ", engine:"المحرك", port:"المنفذ", gpus:"البطاقات",
   health:"الصحة", model:"النموذج", settings:"الإعدادات", gpuMem:"ذاكرة البطاقات",
   elapsed:"المدة", steps:"الخطوات", calls:"الاستدعاءات", tokens:"الرموز",
   retries:"إعادات", fallbacks:"بدائل", cache:"الذاكرة المؤقتة", backend:"المزوّد",
   runtime:"بيئة التنفيذ", started:"بدأ", step:"الخطوة"},
  k:{sel:"تحديد", s:"تشغيل", x:"إيقاف", r:"إعادة تشغيل", t:"ضبط", f:"تفريغ السياق",
   d:"أمر التوزيع", refresh:"التحديث"},
  ask:"KEY=VALUE ... (مثل mem=0.9 schedule-policy=lpm):", sure:"متأكد من {0} {1}؟",
  needSrv:"اختر خادماً (↑↓) أولاً", ro:"للعرض فقط", tok:"لا رمز في العنوان: افتح الرابط الذي طبعه tawjih dashboard",
  langBtn:"English", up:"يعمل منذ"},
 en: {sys:"system & gpu", gpusTitle:"GPUs", minFree:"min free",
  free:"free for auto:", none:"none", noGpu:"no GPU: nvidia-smi is not here",
  srv:"servers", srvN:"server", noSrv:"no servers yet: tawjih server add NAME --engine vllm --model M",
  files:"trees", filesN:"files", runs:"runs", live:"live", shown:"shown",
  log:"actions", noLog:"nothing done yet", det:"details", nothing:"choose a row",
  cols:{no:"No.", title:"Title", runs:"Runs", ok:"OK", avg:"Avg",
   pid:"PID", path:"Path / server", status:"Status", step:"Step",
   time:"Time", calls:"Calls", use:"CPU / RAM", backend:"Backend / engine",
   at:"Time", act:"Action", srv:"Server", by:"By", detail:"Detail"},
  st:{running:"running", done:"done", failed:"failed", stopped:"stopped", partial:"partial",
   waiting:"waiting", skipped:"skipped", made:"made", submitted:"submitted", collected:"collected"},
  l:{status:"Status", up:"Up", engine:"Engine", port:"Port", gpus:"GPUs",
   health:"Health", model:"Model", settings:"Settings", gpuMem:"GPU memory",
   elapsed:"Elapsed", steps:"Steps", calls:"Calls", tokens:"Tokens",
   retries:"Retries", fallbacks:"Fallbacks", cache:"Cache", backend:"Backend",
   runtime:"Runtime", started:"Started", step:"Step"},
  k:{sel:"select", s:"start", x:"stop", r:"restart", t:"tune", f:"flush",
   d:"batch command", refresh:"refresh"},
  ask:"KEY=VALUE ... (mem=0.9 schedule-policy=lpm):", sure:"{0} {1}: are you sure?",
  needSrv:"choose a server (↑↓) first", ro:"read only", tok:"no token in the address: open the link tawjih dashboard printed",
  langBtn:"العربية", up:"Up"}
};
// the language chosen when the board was started (--lang) comes first;
// without it, the one this browser chose last with L
const LANG0 = "__LANG__", FORCED = __FORCED__;
let lang = LANG0, S = null, sel = 0, every = 2000, timer = null, msgT = 0;
if (!FORCED) { try { lang = localStorage.getItem("tawjih.lang") || LANG0; } catch (e) {} }
const token = (location.hash.match(/t=([^&]+)/) || [])[1] || "";
const $ = id => document.getElementById(id);
const esc = s => String(s == null ? "" : s).replace(/[&<>"']/g,
  c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const L = () => T[lang];
const bytes = n => { if (n == null) return "—"; const u = ["B","KiB","MiB","GiB","TiB"];
  let i = 0; while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i ? n.toFixed(n < 10 ? 2 : 1) : n.toFixed(0)) + " " + u[i]; };
const secs = s => { if (s == null) return ""; s = Number(s);
  if (s < 60) return s.toFixed(1) + "s"; if (s < 3600) return Math.floor(s/60) + "m" +
  String(Math.floor(s%60)).padStart(2,"0") + "s"; return Math.floor(s/3600) + "h" +
  String(Math.floor(s%3600/60)).padStart(2,"0") + "m"; };
const grade = x => { const a=[158,206,106], b=[224,175,104], c=[247,118,142];
  const [lo, hi, t] = x < .5 ? [a, b, x/.5] : [b, c, (x-.5)/.5];
  return "rgb(" + lo.map((v,i) => Math.round(v + (hi[i]-v)*t)).join(",") + ")"; };
function bar(pct, n, solid) { n = n || 16; const f = Math.round(Math.max(0, Math.min(100, pct || 0)) / 100 * n);
  let h = '<span class="bar">'; for (let i = 0; i < n; i++)
    h += i < f ? '<i style="background:' + (solid || grade(i / (n - 1))) + '"></i>' : "<i></i>";
  return h + "</span>"; }
const icon = s => ({running:"▶", done:"✓", failed:"✗", stopped:"■", partial:"◐",
  skipped:"–", waiting:"·", submitted:"◔", collected:"✓"}[s] || "·");
const cls = s => ({running:"ok", done:"ok", collected:"ok", failed:"bad",
  stopped:"warn", partial:"warn"}[s] || "muted");
const st = s => (L().st[s] || s);

function sysPanel() {
  const s = S.system, l = L();
  $("tSys").innerHTML = '<span class="ok">' + esc(l.sys) + '</span> <span class="muted ltr">cpu · ram · nvidia-smi</span>';
  $("tSys2").textContent = (s.name || "") + (s.cores ? " · " + s.cores + "c" : "");
  const m = s.mem || {}, rows = [];
  rows.push('<div class="row ltr"><span class="lbl cy">CPU</span>' + bar(s.cpu, 18) +
    '<span class="val">' + (s.cpu == null ? "—" : s.cpu.toFixed(0) + "%") +
    (s.load ? ' <span class="muted">LAV ' + s.load.map(x => x.toFixed(2)).join(" ") + "</span>" : "") + "</span></div>");
  if (m.total) { const p = 100 * m.used / m.total;
    rows.push('<div class="row ltr"><span class="lbl mv">RAM</span>' + bar(p, 18, p > 85 ? "var(--pink)" : null) +
      '<span class="val">' + bytes(m.used) + " / " + bytes(m.total) + ' <span class="muted">' + p.toFixed(0) + "%</span></span></div>"); }
  if (s.disk) { const p = 100 * s.disk.used / s.disk.total;
    rows.push('<div class="row ltr"><span class="lbl">Disk</span>' + bar(p, 18) +
      '<span class="val">' + bytes(s.disk.free) + ' <span class="muted">free</span></span></div>'); }
  $("sys").innerHTML = rows.join("");
  const g = s.gpus || [];
  $("gpus").innerHTML = g.length ? g.map(x => { const low = x.free_mib != null && x.free_mib < S.min_free;
    const p = x.free_mib == null ? 0 : 100 * x.free_mib / Math.max(1, x.total_mib);
    return '<div class="row ltr"><span class="lbl">GPU ' + x.index + ' <span class="muted">' + esc(x.name) + "</span></span>" +
      bar(p, 18, low ? "var(--pink)" : "var(--green)") + '<span class="val ' + (low ? "bad" : "") + '">' +
      (x.free_mib == null ? "—" : x.free_mib + " MiB") + ' <span class="muted">' + x.util.toFixed(0) + "%</span></span></div>"; }).join("")
    : '<div class="muted">' + esc(L().noGpu) + "</div>";
  $("free").innerHTML = g.length ? '<span class="muted">' + esc(l.free) + '</span> <b class="ltr ' +
    (s.free_gpus.length ? "ok" : "bad") + '">' + (s.free_gpus.length ? s.free_gpus.join(",") : esc(l.none)) +
    '</b> <span class="muted ltr">≥ ' + S.min_free + " MiB</span>" : "";
}
function serversPanel() {
  const l = L(), srv = S.rows.filter(r => r.kind === "server");
  $("tSrv").innerHTML = '<span class="cy">' + esc(l.srv) + '</span> <span class="muted ltr">tawjih server</span>';
  $("tSrv2").textContent = srv.length + " " + l.srvN;
  $("servers").innerHTML = srv.length ? '<div class="chips ltr">' + srv.map(r =>
    '<span class="chip ' + (r.pid ? (r.health === 200 ? "ok" : "warn") : "muted") + '" data-name="' + esc(r.name) +
    '">● ' + esc(r.name) + ' <span class="muted">' + esc(r.engine) + ":" + esc(r.port) + "</span></span>").join("") + "</div>"
    : '<div class="muted ltr">' + esc(l.noSrv) + "</div>";
  document.querySelectorAll("#servers .chip").forEach(c => c.onclick = () => {
    const i = S.rows.findIndex(r => r.kind === "server" && r.name === c.dataset.name);
    if (i >= 0) { sel = i; draw(); } });
}
function filesPanel() {
  const l = L(), c = l.cols, f = S.files;
  $("tFiles").innerHTML = '<span class="bl">' + esc(l.files) + "</span>";
  $("tFiles2").textContent = f.length + " " + l.filesN;
  $("files").innerHTML = "<thead><tr><th>" + c.no + "</th><th>" + c.title + '</th><th class="n">' + c.runs +
    '</th><th class="n">' + c.ok + '</th><th class="n">' + c.avg + "</th></tr></thead><tbody>" +
    f.map(x => "<tr><td class=\"ltr muted\">" + (x.running ? '<span class="ok">▶</span> ' : "") + "&nbsp;".repeat(2 * x.depth) +
      esc(x.number) + "</td><td" + (x.running ? ' class="ok"' : x.machine ? "" : ' class="muted"') + ">" + esc(x.title) +
      '</td><td class="n ltr">' + (x.runs || "·") + '</td><td class="n ltr ' + (x.ok == null ? "" : x.ok >= 90 ? "ok" : x.ok >= 50 ? "warn" : "bad") +
      '">' + (x.ok == null ? "" : x.ok + "%") + '</td><td class="n ltr">' + (x.avg == null ? "" : secs(x.avg)) + "</td></tr>").join("") + "</tbody>";
}
function runsPanel() {
  const l = L(), c = l.cols, rows = S.rows;
  const live = rows.filter(r => r.kind === "run" && r.live && r.status === "running").length;
  $("tRuns").innerHTML = '<span class="bad">' + esc(l.runs) + '</span> <span class="' + (live ? "ok" : "muted") + '">' + live + " " + esc(l.live) + "</span>";
  $("tRuns2").textContent = rows.length + " " + l.shown;
  $("runs").innerHTML = "<thead><tr><th>" + c.pid + "</th><th>" + c.path + "</th><th>" + c.status + '</th><th class="n">' + c.step +
    '</th><th class="n">' + c.time + '</th><th class="n">' + c.calls + '</th><th class="n">' + c.use + "</th><th>" + c.backend +
    "</th></tr></thead><tbody>" + rows.map((r, i) => {
      let pid, step, calls, use, back;
      if (r.kind === "server") { pid = r.pid || "srv"; step = ":" + r.port; calls = r.health || "—";
        use = [r.cpu == null ? "" : r.cpu.toFixed(0) + "%", r.rss ? bytes(r.rss) : ""].filter(Boolean).join(" "); back = r.engine; }
      else if (r.kind === "job") { pid = "job"; step = r.pct == null ? "" : r.pct.toFixed(0) + "%"; calls = ""; use = ""; back = r.runtime; }
      else { pid = r.pid || ""; step = r.done + "/" + r.total; calls = r.calls;
        use = [r.cpu == null ? "" : r.cpu.toFixed(0) + "%", r.rss ? bytes(r.rss) : ""].filter(Boolean).join(" "); back = r.backend; }
      return '<tr data-i="' + i + '"' + (i === sel ? ' class="sel"' : "") + '><td class="ltr ' + (r.kind === "server" ? "mv" : "muted") + '">' + esc(pid) +
        '</td><td class="ltr">' + esc(r.name) + '</td><td class="' + cls(r.status) + '">' + icon(r.status) + " " + esc(st(r.status)) +
        '</td><td class="n ltr">' + esc(step) + '</td><td class="n ltr">' + secs(r.seconds) + '</td><td class="n ltr ' +
        (r.kind === "server" && r.health === 200 ? "ok" : "") + '">' + esc(calls) + '</td><td class="n ltr muted">' + esc(use) +
        '</td><td class="ltr mv">' + esc(back) + "</td></tr>"; }).join("") + "</tbody>";
  document.querySelectorAll("#runs tbody tr").forEach(tr => tr.onclick = () => { sel = Number(tr.dataset.i); draw(); });
}
function kv(k, v, c) { return '<span style="white-space:nowrap"><span class="muted">' + esc(k) + ':</span> <span class="ltr ' + (c || "") + '">' + esc(v) + "</span></span>"; }
function detailPanel() {
  const l = L().l, r = S.rows[sel];
  $("tDet").innerHTML = '<span class="cy">' + esc(L().det) + "</span>";
  if (!r) { $("tDet2").textContent = ""; $("detail").innerHTML = '<div class="muted">' + esc(L().nothing) + "</div>"; return; }
  $("tDet").innerHTML = '<b class="cy ltr">' + esc((r.pid ? r.pid + " " : "") + r.name) + '</b> <span class="mv">' +
    esc(r.kind === "server" ? L().srvN + " · " + r.engine : r.kind === "job" ? "job · " + r.runtime : (r.trees || []).join(" · ")) + "</span>";
  $("tDet2").innerHTML = '<span class="' + cls(r.status) + '">' + icon(r.status) + " " + esc(st(r.status)) + "</span>";
  let h = "";
  if (r.kind === "server") {
    h += '<div class="chips" style="gap:16px">' + kv(l.up, r.seconds ? secs(r.seconds) : "—") + kv(l.engine, r.engine) + kv(l.port, r.port) +
      kv(l.gpus, r.gpus || "—") + kv(l.health, r.health || "—", r.health === 200 ? "ok" : r.pid ? "bad" : "") + "</div>";
    h += '<div class="chips" style="gap:16px;margin-top:4px">' + kv(l.model, r.model || "—") +
      (r.cpu != null ? kv("CPU", r.cpu.toFixed(1) + "%") : "") + (r.rss ? kv("RAM", bytes(r.rss)) : "") +
      (r.gpu_mib != null ? kv(l.gpuMem, r.gpu_mib + " MiB") : "") + "</div>";
    if (r.settings.length) h += '<div style="margin-top:4px">' + kv(l.settings, r.settings.join(" ")) + "</div>";
    h += '<pre class="tail">$ ' + esc(r.command) + "\n" + esc(r.tail.join("\n")) + "</pre>";
  } else if (r.kind === "job") {
    h += '<div class="chips" style="gap:16px">' + kv(l.runtime, r.runtime) + kv(l.started, r.started || "—") +
      (r.pct != null ? kv("%", r.pct.toFixed(0) + "%") : "") + "</div>";
    if (r.pct != null) h += '<div class="row" style="margin-top:4px">' + bar(r.pct, 30) + "</div>";
    h += '<pre class="tail">$ ' + esc(r.command) + "\n" + esc(r.tail.join("\n")) + "</pre>";
  } else {
    h += '<div class="chips" style="gap:16px">' + kv(l.elapsed, secs(r.seconds)) + kv(l.steps, r.done + "/" + r.total) +
      kv(l.calls, r.calls) + kv(l.tokens, (r.in || 0) + " / " + (r.out || 0)) + kv(l.retries, r.retries) +
      kv(l.fallbacks, r.fallbacks) + kv(l.cache, r.cache) + "</div>";
    h += '<div class="chips" style="gap:16px;margin-top:4px">' + kv(l.backend, r.backend || "—") + kv(l.model, r.model || "—") +
      (r.cpu != null ? kv("CPU", r.cpu.toFixed(1) + "%") : "") + (r.rss ? kv("RAM", bytes(r.rss)) : "") + "</div>";
    const long = Math.max(0.001, ...r.steps.map(p => p.seconds || 0));
    h += '<table style="margin-top:6px"><thead><tr><th>#</th><th>' + esc(L().cols.no) + "</th><th>" + esc(L().cols.title) +
      "</th><th>" + esc(L().cols.time) + '</th><th class="n"></th></tr></thead><tbody>' + r.steps.map(p =>
        '<tr><td class="' + cls(p.status) + '">' + icon(p.status) + " " + esc(p.step) + '</td><td class="ltr muted">' + esc(p.number) +
        "</td><td" + (p.status === "running" ? ' class="ok"' : "") + ">" + esc(p.title) + "</td><td>" +
        (p.seconds ? bar(100 * p.seconds / long, 14) : "") + '</td><td class="n ltr">' + secs(p.seconds) + "</td></tr>").join("") + "</tbody></table>";
  }
  $("detail").innerHTML = h;
}
function logPanel() {
  const l = L(), c = l.cols;
  $("tLog").innerHTML = '<span class="warn">' + esc(l.log) + '</span> <span class="muted ltr">history/servers-*.jsonl</span>';
  $("log").innerHTML = S.log.length ? "<thead><tr><th>" + c.at + "</th><th>" + c.act + "</th><th>" + c.srv + "</th><th>" + c.by +
    "</th><th>" + c.detail + "</th></tr></thead><tbody>" + S.log.map(x => '<tr><td class="ltr muted">' + esc(x.time.slice(11) || x.time) +
    '</td><td class="ltr cy">' + esc(x.action) + '</td><td class="ltr">' + esc(x.server) + "</td><td>" + esc(x.by) +
    '</td><td class="ltr muted">' + esc(x.detail) + "</td></tr>").join("") + "</tbody>"
    : '<tbody><tr><td class="muted">' + esc(l.noLog) + "</td></tr></tbody>";
}
function keysBar() {
  const k = L().k, ro = S && S.read_only;
  const item = (key, label, off) => '<span class="k' + (off ? " off" : "") + '" data-k="' + key + '"><b class="ltr">' + key + "</b> " + esc(label) + "</span>";
  $("keys").innerHTML = item("↑↓", k.sel) + item("s", k.s, ro) + item("x", k.x, ro) + item("r", k.r, ro) + item("t", k.t, ro) +
    item("f", k.f, ro) + item("d", k.d, ro) + '<span class="k"><b class="ltr" data-k="+">+</b>/<b class="ltr" data-k="-">-</b> ' +
    esc(k.refresh) + ' <span class="ltr">' + every + "ms</span></span>" + (ro ? '<span class="warn">' + esc(L().ro) + "</span>" : "");
  document.querySelectorAll("#keys [data-k]").forEach(e => e.onclick = ev => { ev.stopPropagation(); key(e.dataset.k); });
}
function draw() {
  document.documentElement.lang = lang; document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
  $("lang").textContent = "[L] " + L().langBtn; document.title = "tawjih · " + (lang === "ar" ? "لوحة التحكم" : "dashboard");
  if (!S) { keysBar(); return; }
  sel = Math.max(0, Math.min(sel, S.rows.length - 1));
  sysPanel(); serversPanel(); filesPanel(); runsPanel(); detailPanel(); logPanel(); keysBar();
  $("clock").textContent = S.time;
}
function say(text) { $("msg").textContent = text; msgT = Date.now(); }
async function load() {
  if (!token) { say(L().tok); return; }
  try {
    const res = await fetch("/api/state", {headers: {Authorization: "Bearer " + token}});
    if (res.status === 401) { say(L().tok); return; }
    S = await res.json(); draw();
    if (msgT && Date.now() - msgT > 8000) { $("msg").textContent = ""; msgT = 0; }
  } catch (e) { say(String(e)); }
}
async function act(k, text) {
  const r = S && S.rows[sel];
  if (!r || r.kind !== "server") { say(L().needSrv); return; }
  if ((k === "x" || k === "r") && !confirm(L().sure.replace("{0}", L().k[k]).replace("{1}", r.name))) return;
  if (k === "t") { text = prompt(r.name + ": " + L().ask, ""); if (!text) return; }
  say("…");
  try {
    const res = await fetch("/api/action", {method: "POST", headers: {Authorization: "Bearer " + token,
      "Content-Type": "application/json"}, body: JSON.stringify({server: r.name, key: k, text: text || ""})});
    const j = await res.json(); say(j.message || j.error || res.status);
  } catch (e) { say(String(e)); }
  load();
}
function key(k) {
  if (k === "↑↓" || k === "down") { sel = Math.min((S ? S.rows.length : 1) - 1, sel + 1); draw(); }
  else if (k === "up") { sel = Math.max(0, sel - 1); draw(); }
  else if (k === "+") { every = Math.max(500, every - 500); start(); keysBar(); }
  else if (k === "-") { every = Math.min(10000, every + 500); start(); keysBar(); }
  else if (k === "l") { lang = lang === "ar" ? "en" : "ar"; try { localStorage.setItem("tawjih.lang", lang); } catch (e) {} draw(); }
  else if ("sxrtfd".includes(k) && k.length === 1) { if (!(S && S.read_only)) act(k); }
}
function start() { if (timer) clearInterval(timer); timer = setInterval(load, every); }
document.addEventListener("keydown", e => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const m = {ArrowDown: "down", ArrowUp: "up"}[e.key];
  if (m) { e.preventDefault(); key(m); return; }
  const k = e.key.toLowerCase();
  if (["+", "-", "=", "l", "s", "x", "r", "t", "f", "d"].includes(k)) { e.preventDefault(); key(k === "=" ? "+" : k); }
});
$("lang").onclick = () => key("l");
draw(); load(); start();
</script>
</body>
</html>
"""


__all__ = ['Board', 'PAGE', 'add_dashboard_parser', 'cmd_dashboard',
           'make_server']
