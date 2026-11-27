"""Model servers: start, watch and stop the engines that serve models
(vLLM, SGLang, LMDeploy, TensorRT-LLM, Triton, Aphrodite, TabbyAPI,
LiteLLM, Ollama, llama.cpp, or any command), and use them as providers.

Tawjih does not contain these engines. It writes their command lines from
a few common settings (model, port, GPUs, tensor and pipeline parallel,
context, memory, KV cache), passes anything else to them as it is
(--pass), runs them in the background, and reads their health, models and
metrics over HTTP.

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import getpass
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .core import (
    NAME_RE, Store, alive, die, key_values, kill_tree, spawn_detached, who,
)
from .providers import (
    load_providers,
)
from .engine import (
    build_prompt,
)
from .governance import (
    _save_providers,
)

SERVERS_FILE = "servers.json"
STATE_DIR = "servers"
COMMON = ("model", "host", "port", "gpus", "tp", "pp", "context", "mem",
          "kv_cache", "api_key_env", "served_name")
LABEL = {"tp": "--tp", "pp": "--pp", "context": "--context",
         "mem": "--mem", "kv_cache": "--kv-cache",
         "served_name": "--served-name", "api_key_env": "--api-key-env"}


# ------------------------------------------------------------------ engines
# Each engine: its program, the flags of the common settings it takes, how
# its key is given, and where its health, models and metrics are read.
# A setting an engine does not take is refused with the reason; anything
# else goes through --pass. Built from each engine's documented options.
ENGINES = {
    "vllm": {
        "argv": ["{exe:vllm}", "serve", "{model}", "--host", "{host}",
                 "--port", "{port}"],
        "flags": {"tp": "--tensor-parallel-size",
                  "pp": "--pipeline-parallel-size",
                  "context": "--max-model-len",
                  "mem": "--gpu-memory-utilization",
                  "kv_cache": "--kv-cache-dtype",
                  "served_name": "--served-model-name"},
        "kv": {"fp8": "fp8", "fp8_e4m3": "fp8_e4m3", "fp8_e5m2": "fp8_e5m2"},
        "key": ("env", "VLLM_API_KEY"),
        "health": "/health", "models": "/v1/models", "metrics": "/metrics",
        "kind": "openai", "base": "/v1", "image": "vllm/vllm-openai:v0.32.0"},
    "sglang": {
        "argv": ["{python}", "-m", "sglang.launch_server", "--model-path",
                 "{model}", "--host", "{host}", "--port", "{port}",
                 "--enable-metrics"],
        "flags": {"tp": "--tp-size", "pp": "--pp-size",
                  "context": "--context-length",
                  "mem": "--mem-fraction-static",
                  "kv_cache": "--kv-cache-dtype",
                  "served_name": "--served-model-name"},
        "kv": {"fp8": "fp8_e5m2", "fp8_e5m2": "fp8_e5m2",
               "fp8_e4m3": "fp8_e4m3"},
        "key": ("flag", "--api-key"),
        "health": "/health", "models": "/v1/models", "metrics": "/metrics",
        "kind": "openai", "base": "/v1"},
    "lmdeploy": {
        "argv": ["{exe:lmdeploy}", "serve", "api_server", "{model}",
                 "--server-name", "{host}", "--server-port", "{port}"],
        "flags": {"tp": "--tp", "context": "--session-len",
                  "mem": "--cache-max-entry-count",
                  "kv_cache": "--quant-policy", "served_name": "--model-name"},
        "kv": {"int8": "8", "int4": "4"},
        "key": ("flag", "--api-keys"),
        "health": "/v1/models", "models": "/v1/models", "metrics": None,
        "kind": "openai", "base": "/v1"},
    "trtllm": {
        "argv": ["{exe:trtllm-serve}", "{model}", "--host", "{host}",
                 "--port", "{port}"],
        "flags": {"tp": "--tp_size", "pp": "--pp_size",
                  "context": "--max_seq_len",
                  "mem": "--kv_cache_free_gpu_memory_fraction"},
        "kv": {}, "key": None,
        "health": "/health", "models": "/v1/models", "metrics": "/metrics",
        "kind": "openai", "base": "/v1"},
    "triton": {
        "argv": ["{exe:tritonserver}", "--model-repository", "{model}",
                 "--http-address", "{host}", "--http-port", "{port}"],
        "flags": {}, "kv": {}, "key": None,
        "health": "/v2/health/ready", "models": None,
        "metrics": ":8002/metrics", "kind": None, "base": ""},
    "aphrodite": {
        "argv": ["{exe:aphrodite}", "run", "{model}", "--host", "{host}",
                 "--port", "{port}"],
        "flags": {"tp": "--tensor-parallel-size",
                  "pp": "--pipeline-parallel-size",
                  "context": "--max-model-len",
                  "mem": "--gpu-memory-utilization",
                  "kv_cache": "--kv-cache-dtype",
                  "served_name": "--served-model-name"},
        "kv": {"fp8": "fp8"},
        "key": ("flag", "--api-keys"),
        "health": "/health", "models": "/v1/models", "metrics": "/metrics",
        "kind": "openai", "base": "/v1"},
    "tabby": {
        "argv": ["{python}", "main.py", "--host", "{host}", "--port",
                 "{port}", "--model-name", "{model}"],
        "flags": {"context": "--max-seq-len", "kv_cache": "--cache-mode",
                  "tp": None},
        "kv": {"fp16": "FP16", "q8": "Q8", "q6": "Q6", "q4": "Q4"},
        "key": None, "needs_dir": True,
        "health": "/health", "models": "/v1/models", "metrics": None,
        "kind": "openai", "base": "/v1"},
    "litellm": {
        "argv": ["{exe:litellm}", "--config", "{model}", "--host", "{host}",
                 "--port", "{port}"],
        "flags": {}, "kv": {}, "key": ("env", "LITELLM_MASTER_KEY"),
        "health": "/health/liveliness", "models": "/v1/models",
        "metrics": None, "kind": "openai", "base": "/v1"},
    "ollama": {
        "argv": ["{exe:ollama}", "serve"],
        "env": {"OLLAMA_HOST": "{host}:{port}"},
        "env_flags": {"context": "OLLAMA_CONTEXT_LENGTH",
                      "kv_cache": "OLLAMA_KV_CACHE_TYPE"},
        "flags": {}, "kv": {"f16": "f16", "q8_0": "q8_0", "q4_0": "q4_0"},
        "key": None, "model_optional": True,
        "health": "/", "models": "/api/tags", "metrics": None,
        "kind": "ollama", "base": ""},
    "llamacpp": {
        "argv": ["{exe:llama-server}", "-m", "{model}", "--host", "{host}",
                 "--port", "{port}", "--metrics"],
        "flags": {"context": "-c", "kv_cache": "--cache-type-k",
                  "served_name": "--alias"},
        "kv": {"f16": "f16", "q8_0": "q8_0", "q4_0": "q4_0"},
        "key": ("flag", "--api-key"),
        "health": "/health", "models": "/v1/models", "metrics": "/metrics",
        "kind": "openai", "base": "/v1"},
    "custom": {
        "argv": [], "flags": {}, "kv": {}, "key": None,
        "health": "/health", "models": "/v1/models", "metrics": "/metrics",
        "kind": "openai", "base": "/v1"},
}


def servers_file(root):
    return Path(root) / SERVERS_FILE


def load_servers(root):
    f = servers_file(root)
    if not f.is_file():
        return {"servers": {}}
    try:
        data = json.loads(f.read_text(encoding="utf-8-sig"))
    except ValueError as e:
        die("servers.json is not readable JSON (%s)" % e)
    data.setdefault("servers", {})
    return data


def save_servers(root, data):
    servers_file(root).write_text(json.dumps(data, indent=2,
                                             ensure_ascii=False) + "\n",
                                  encoding="utf-8")


def get_server(root, name):
    servers = load_servers(root)["servers"]
    if name not in servers:
        die("no server %s (see: tawjih server list)" % name)
    return servers[name]


def state_dir(root, name):
    return Path(root) / STATE_DIR / name


# ------------------------------------------------------------ command line
def _fill(word, cfg):
    def one(m):
        key = m.group(1)
        if key.startswith("exe:"):
            return cfg.get("exe") or key[4:]
        if key == "python":
            return cfg.get("python") or sys.executable
        return str(cfg.get(key, ""))
    return re.sub(r"\{([\w:\-]+)\}", one, word)


def command(cfg, key_value=None):
    """(argv, env) of a server. The key is never in what is shown or
    saved: with key_value None it is written as $VARIABLE."""
    eng = ENGINES[cfg["engine"]]
    if cfg["engine"] == "custom":
        argv = [_fill(w, cfg) for w in shlex.split(cfg["command"])]
    else:
        argv = [_fill(w, cfg) for w in eng["argv"]]
    env = {k: _fill(v, cfg) for k, v in (eng.get("env") or {}).items()}
    for k, var in (eng.get("env_flags") or {}).items():
        if cfg.get(k) not in (None, ""):
            env[var] = str(eng["kv"].get(cfg[k], cfg[k]) if k == "kv_cache"
                           else cfg[k])
    if cfg["engine"] == "ollama" and cfg.get("kv_cache"):
        env["OLLAMA_FLASH_ATTENTION"] = "1"    # a quantised cache needs it
    for k, flag in eng["flags"].items():
        v = cfg.get(k)
        if v in (None, "") or not flag:
            continue
        if k == "kv_cache":
            v = eng["kv"][v]
            if cfg["engine"] == "llamacpp":
                argv += [flag, v, "--cache-type-v", v]
                continue
        argv += [flag, str(v)]
    if cfg.get("gpus") and cfg["gpus"] != "auto":
        env["CUDA_VISIBLE_DEVICES"] = env["HIP_VISIBLE_DEVICES"] = \
            str(cfg["gpus"])
    var = cfg.get("api_key_env")
    if var and eng.get("key"):
        how, where = eng["key"]
        value = key_value if key_value is not None else "$" + var
        if how == "flag":
            argv += [where, value]
        else:
            env[where] = value
    argv += [str(x) for x in cfg.get("pass") or []]
    env.update({k: str(v) for k, v in (cfg.get("env") or {}).items()})
    return argv, env


def show_command(argv, env):
    pre = " ".join("%s=%s" % (k, v if v.startswith("$") else shlex.quote(v))
                   for k, v in env.items())
    cmd = " ".join(shlex.quote(a) if not a.startswith("$") else a
                   for a in argv)
    return (pre + " " if pre else "") + cmd


def check_server(cfg):
    """Refuse settings an engine does not take, with what to do."""
    eng = ENGINES.get(cfg.get("engine"))
    if not eng:
        die("unknown engine %s (one of: %s)" % (cfg.get("engine"),
                                                 ", ".join(ENGINES)))
    name = cfg["engine"]
    if name == "custom" and not cfg.get("command"):
        die("--engine custom needs --command \"PROGRAM ARGS\" ({model}, "
            "{host} and {port} are filled in)")
    if not cfg.get("model") and not eng.get("model_optional") \
            and name != "custom":
        die("--model is needed for %s%s" % (name, {
            "litellm": " (its config file)",
            "triton": " (the model repository folder)",
            "tabby": " (the model folder's name in TabbyAPI's models)"}.get(
                name, "")))
    if eng.get("needs_dir") and not cfg.get("dir"):
        die("%s runs from its own folder: give --dir FOLDER" % name)
    takes = set(eng["flags"]) | set(eng.get("env_flags") or {})
    for k in ("tp", "pp", "context", "mem", "kv_cache", "served_name"):
        if cfg.get(k) not in (None, "") and (k not in takes or
                                              not (eng["flags"].get(k) or
                                                   (eng.get("env_flags") or
                                                    {}).get(k))):
            die("%s does not take %s here; give the engine's own option "
                "with --pass" % (name, LABEL[k]))
    if cfg.get("kv_cache") and cfg["kv_cache"] not in eng["kv"]:
        die("%s --kv-cache takes %s" % (name, ", ".join(eng["kv"])
                                        or "nothing"))
    if cfg.get("api_key_env") and not eng.get("key") and name != "custom":
        die("%s takes no key from Tawjih; set it in the engine's own "
            "settings" % name)
    if cfg.get("mem") is not None and not 0 < float(cfg["mem"]) <= 1:
        die("--mem is a fraction of the GPU memory, above 0 and at most 1")
    for k in ("tp", "pp", "context", "port"):
        if cfg.get(k) is not None and int(cfg[k]) < 1:
            die("%s takes a number of 1 or more" % LABEL.get(k, "--" + k))
    if cfg.get("gpus") and cfg["gpus"] != "auto" and \
            not re.fullmatch(r"\d+(,\d+)*", str(cfg["gpus"])):
        die("--gpus takes GPU numbers (0,1,2,3) or auto")
    if cfg.get("min_free") is not None and int(cfg["min_free"]) < 0:
        die("--min-free takes MiB, 0 or more")
    if cfg.get("api_key_env") and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", cfg["api_key_env"]):
        die("--api-key-env takes the NAME of an environment variable")
    for k in cfg.get("env") or {}:
        if re.search(r"(?i)api.?key|secret|passw|token", k):
            die("--env %s looks like a secret; give its variable's name "
                "with --api-key-env, or set it where the server runs" % k)


def url(cfg, path):
    """The address of a path on the server (a path starting with :PORT is
    on another port of the same host)."""
    host = cfg.get("host") or "127.0.0.1"
    host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    if path.startswith(":"):
        port, _, rest = path[1:].partition("/")
        return "http://%s:%s/%s" % (host, port, rest)
    return "http://%s:%s%s" % (host, cfg["port"], path)


def health_path(cfg):
    return cfg.get("health") or ENGINES[cfg["engine"]]["health"]


def _get(address, method="GET", data=None, headers=None, timeout=5):
    """(status, text) of a request, or (None, reason) without an answer."""
    body = None if data is None else data.encode("utf-8")
    req = urllib.request.Request(address, data=body, method=method,
                                 headers=dict(headers or {}))
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return None, str(getattr(e, "reason", e))


def _auth(cfg, header=None, env_name=None):
    var = env_name or cfg.get("api_key_env")
    key = os.environ.get(var, "") if var else ""
    if not key:
        return {}
    if header and header.lower() != "authorization":
        return {header: key}
    return {"Authorization": "Bearer " + key}


# ------------------------------------------------------------- processes
def read_pid(root, name):
    f = state_dir(root, name) / "pid"
    try:
        pid = int(f.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if alive(pid) else None


def port_in_use(cfg):
    host = cfg.get("host") or "127.0.0.1"
    host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    with socket.socket() as s:
        s.settimeout(1)
        return s.connect_ex((host, int(cfg["port"]))) == 0


def port_owner(port):
    """The process listening on a port, when the system tells (lsof or ss
    on Linux and macOS, netstat on Windows)."""
    try:
        if os.name == "nt":
            out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                                 capture_output=True, text=True,
                                 timeout=10).stdout
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[1].endswith(":%d" % port) \
                        and parts[3].upper() == "LISTENING":
                    return int(parts[4])
            return None
        if shutil.which("lsof"):
            out = subprocess.run(["lsof", "-nP", "-iTCP:%d" % port,
                                  "-sTCP:LISTEN", "-t"], capture_output=True,
                                 text=True, timeout=10).stdout.split()
            return int(out[0]) if out else None
        if shutil.which("ss"):
            out = subprocess.run(["ss", "-ltnpH", "sport = :%d" % port],
                                 capture_output=True, text=True,
                                 timeout=10).stdout
            m = re.search(r"pid=(\d+)", out)
            return int(m.group(1)) if m else None
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def group_pids(pid):
    """A server and the processes it started (its process group)."""
    pids = {pid}
    proc = Path("/proc")
    if not proc.is_dir():
        return pids
    for d in proc.iterdir():
        if d.name.isdigit():
            try:
                stat = (d / "stat").read_text().rsplit(")", 1)[1].split()
                if int(stat[2]) == pid:            # its process group
                    pids.add(int(d.name))
            except (OSError, IndexError, ValueError):
                continue
    return pids


def gpu_memory(pids):
    """MiB of GPU memory those processes hold, from nvidia-smi, or None."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True,
            text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    total = 0
    for line in out.splitlines():
        try:
            p, mib = [int(x) for x in line.split(",")]
        except ValueError:
            continue
        if p in pids:
            total += mib
    return total


def _note(root, action, name, **extra):
    """Who started or stopped which server, in history/."""
    d = Path(root) / "history"
    d.mkdir(exist_ok=True)
    line = dict({"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "action": action, "server": name, "by": who()}, **extra)
    with open(d / ("servers-%s.jsonl" % time.strftime("%Y%m%d")), "a",
              encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")


def start(root, name, cfg, wait=0):
    if read_pid(root, name):
        die("%s is already running (pid %d)" % (name, read_pid(root, name)))
    if port_in_use(cfg):
        owner = port_owner(int(cfg["port"]))
        die("port %s is in use%s; stop that, or give the server another "
            "--port" % (cfg["port"], " by process %d" % owner if owner
                        else ""))
    key = None
    var = cfg.get("api_key_env")
    if var and ENGINES[cfg["engine"]].get("key"):
        key = os.environ.get(var)
        if not key:
            die("%s is not set: the server's key is read from it when it "
                "starts" % var)
    if cfg.get("gpus") == "auto":
        cfg = dict(cfg, gpus=choose_gpus(cfg))
        print("%s: GPUs %s (the most free memory)" % (name, cfg["gpus"]))
    argv, env = command(cfg, key)
    exe = argv[0]
    if not (shutil.which(exe) or Path(exe).is_file()):
        die("%s was not found: install %s, or give its path with --exe"
            % (exe, cfg["engine"]))
    d = state_dir(root, name)
    d.mkdir(parents=True, exist_ok=True)
    log = d / "log.txt"
    with open(log, "a", encoding="utf-8") as f:
        f.write("\n== %s started: %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                            show_command(*command(cfg))))
    full_env = dict(os.environ, **env)
    proc = spawn_detached(argv, cwd=cfg.get("dir") or None, log=log,
                          env=full_env)
    (d / "pid").write_text(str(proc.pid))
    shown = show_command(*command(cfg))
    (d / "started.json").write_text(json.dumps(
        {"pid": proc.pid, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
         "command": shown, "by": who()}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    _note(root, "start", name, pid=proc.pid, command=shown)
    print("%s started (pid %d): %s" % (name, proc.pid, shown))
    print("log: %s" % log)
    if not wait:
        print("follow it with: tawjih server status %s, tawjih server logs "
              "%s --follow" % (name, name))
        return 0
    deadline = time.monotonic() + wait
    eng = ENGINES[cfg["engine"]]
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            print("%s ended with status %s; the end of its log:" % (
                name, proc.returncode))
            print(tail(log, 20), end="")
            return 1
        code, _ = _get(url(cfg, health_path(cfg)), timeout=2)
        if code == 200:
            print("%s is ready at %s" % (name, url(cfg, eng.get("base")
                                                    or "")))
            return 0
        time.sleep(1)
    print("%s is not ready after %d s (large models take minutes); follow "
          "it with: tawjih server logs %s --follow" % (name, wait, name))
    return 1


# ----------------------------------------------------------------- GPUs
def gpu_list():
    """The NVIDIA GPUs with their memory in MiB, or None without
    nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free,"
             "utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 5:
            try:
                gpus.append({"index": int(parts[0]), "name": parts[1],
                             "total": int(parts[2]), "free": int(parts[3]),
                             "busy": int(parts[4])})
            except ValueError:
                continue
    return gpus


def choose_gpus(cfg, gpus=None):
    """As many GPUs as the server splits its model over (tp x pp), those
    with the most free memory and at least --min-free MiB. Never a guess:
    without nvidia-smi, or without enough GPUs, nothing starts."""
    gpus = gpu_list() if gpus is None else gpus
    if gpus is None:
        die("--gpus auto reads free memory with nvidia-smi, which is not "
            "here; give the GPU numbers (--gpus 0,1)")
    need = int(cfg.get("tp") or 1) * int(cfg.get("pp") or 1)
    least = int(cfg.get("min_free") or 0)
    fit = sorted((g for g in gpus if g["free"] >= least),
                 key=lambda g: (-g["free"], g["index"]))
    if len(fit) < need:
        die("%d GPU(s) with %d MiB free are needed, %d have it: %s" % (
            need, least, len(fit), ", ".join(
                "%d: %d MiB free" % (g["index"], g["free"]) for g in gpus)
            or "no GPU"))
    return ",".join(str(g["index"]) for g in sorted(fit[:need],
                                                    key=lambda g: g["index"]))


# --------------------------------------------------------------- prepare
# Programs that make a model ready to serve, run once before the server.
PREPARE = {
    "lmdeploy": {"awq": ["lmdeploy", "lite", "auto_awq", "{model}",
                         "--work-dir", "{out}"],
                 "hint": "serve it with: tawjih server set NAME --model OUT "
                         "--pass=--model-format --pass=awq"},
    "trtllm": {"build": ["trtllm-build", "--checkpoint_dir", "{model}",
                         "--output_dir", "{out}"],
               "hint": "serve it with: tawjih server set NAME --model OUT"},
}


def prepare(root, name, cfg, out, tool_args, step=None):
    table = PREPARE.get(cfg["engine"])
    if not table:
        die("%s has nothing to prepare here (prepare is for %s)" % (
            cfg["engine"], ", ".join(PREPARE)))
    steps = [k for k in table if k != "hint"]
    step = step or steps[0]
    if step not in steps:
        die("%s prepares: %s" % (cfg["engine"], ", ".join(steps)))
    if not out:
        die("prepare needs --out FOLDER for the result")
    argv = [w.replace("{model}", cfg.get("model") or "").replace(
        "{out}", out) for w in table[step]] + list(tool_args or [])
    if not shutil.which(argv[0]):
        die("%s was not found: install %s" % (argv[0], cfg["engine"]))
    env = dict(os.environ)
    gpus = cfg.get("gpus")
    if gpus == "auto":
        gpus = choose_gpus(dict(cfg, tp=1, pp=1))
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = env["HIP_VISIBLE_DEVICES"] = gpus
    d = state_dir(root, name)
    d.mkdir(parents=True, exist_ok=True)
    log = d / ("prepare-%s.txt" % step)
    shown = " ".join(shlex.quote(a) for a in argv)
    print("%s %s: %s" % (name, step, shown))
    t0 = time.monotonic()
    with open(log, "w", encoding="utf-8") as f:
        f.write("== %s\n" % shown)
        f.flush()
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env, text=True,
                                encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        code = proc.wait()
    took = round(time.monotonic() - t0, 1)
    _note(root, "prepare", name, step=step, command=shown, status=code,
          seconds=took, out=out)
    if code:
        print("%s %s failed (status %d); the output is in %s" % (
            name, step, code, log))
        return 1
    print("%s %s: done in %s s, in %s" % (name, step, took, out))
    print(table["hint"].replace("NAME", name).replace("OUT", out))
    return 0


# ----------------------------------------------------------------- bench
def _one_request(cfg, prompt, max_tokens, timeout):
    eng = ENGINES[cfg["engine"]]
    kind = cfg.get("provider_kind", eng["kind"])
    model = cfg.get("served_name") or cfg.get("model") or ""
    if kind == "ollama":
        body = {"model": model, "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"num_predict": max_tokens}}
        if cfg.get("context"):
            body["options"]["num_ctx"] = int(cfg["context"])
        address = url(cfg, "/api/chat")
    elif kind == "openai":
        body = {"model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        address = url(cfg, (eng.get("base") or "/v1") + "/chat/completions")
    else:
        die("%s does not answer as a chat model; bench needs an "
            "OpenAI-compatible or Ollama server" % cfg["engine"])
    t0 = time.monotonic()
    code, text = _get(address, "POST", json.dumps(body), _auth(cfg),
                      timeout=timeout)
    took = time.monotonic() - t0
    if code != 200:
        return {"ok": False, "seconds": took,
                "error": "%s %s" % (code or "no answer", text[:200])}
    try:
        data = json.loads(text)
    except ValueError:
        return {"ok": False, "seconds": took, "error": "not JSON"}
    if kind == "ollama":
        pin, pout = data.get("prompt_eval_count"), data.get("eval_count")
    else:
        u = data.get("usage") or {}
        pin, pout = u.get("prompt_tokens"), u.get("completion_tokens")
    return {"ok": True, "seconds": took, "prompt_tokens": pin,
            "output_tokens": pout}


def _pct(values, q):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def bench(root, name, cfg, prompt, requests, concurrency, max_tokens,
          vary=False, timeout=600, source=""):
    """Requests at a running server, as many at once as --concurrency:
    how long each takes and how many tokens a second come out."""
    from concurrent.futures import ThreadPoolExecutor
    if requests < 1 or concurrency < 1 or max_tokens < 1:
        die("--requests, --concurrency and --max-tokens take 1 or more")
    code, _ = _get(url(cfg, health_path(cfg)), timeout=5)
    if code != 200:
        die("%s does not answer at %s (start it first)" % (
            name, url(cfg, health_path(cfg))))

    def one(i):
        text = ("[%d] " % i + prompt) if vary else prompt
        return _one_request(cfg, text, max_tokens, timeout)
    print("%s: %d requests, %d at once, up to %d tokens each, a prompt of "
          "%d characters%s" % (name, requests, concurrency, max_tokens,
                               len(prompt), ", each different" if vary
                               else ""))
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one, range(requests)))
    wall = time.monotonic() - t0
    ok = [r for r in results if r["ok"]]
    lat = [r["seconds"] for r in ok]
    out_tok = sum(r.get("output_tokens") or 0 for r in ok)
    in_tok = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]
    summary = {
        "requests": requests, "concurrency": concurrency,
        "max_tokens": max_tokens, "ok": len(ok),
        "failed": len(results) - len(ok), "seconds": round(wall, 2),
        "latency_p50": round(_pct(lat, 0.5), 3) if lat else None,
        "latency_p95": round(_pct(lat, 0.95), 3) if lat else None,
        "output_tokens": out_tok,
        "output_tokens_per_second": round(out_tok / wall, 1) if wall else None,
        "prompt_tokens_mean": round(sum(in_tok) / len(in_tok)) if in_tok
        else None,
        "requests_per_second": round(len(ok) / wall, 2) if wall else None}
    print("ok %d, failed %d, %.1f s in all" % (summary["ok"],
                                               summary["failed"], wall))
    if lat:
        print("time per request: median %.2f s, 95%% under %.2f s" % (
            summary["latency_p50"], summary["latency_p95"]))
    if out_tok:
        print("output: %d tokens, %.1f tokens/s in all" % (
            out_tok, summary["output_tokens_per_second"]))
    else:
        print("output tokens: the server did not report them")
    if summary["prompt_tokens_mean"]:
        print("prompt: %d tokens on average" % summary["prompt_tokens_mean"])
    for r in results:
        if not r["ok"]:
            print("failed: %s" % r["error"])
            break
    d = Path(root) / "history"
    d.mkdir(exist_ok=True)
    f = d / ("bench-%s-%s.json" % (name, time.strftime("%Y%m%d-%H%M%S")))
    shown = {k: v for k, v in cfg.items() if k != "env"}
    f.write_text(json.dumps({
        "server": name, "settings": shown, "source": source,
        "prompt_chars": len(prompt), "vary": vary, "by": who(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "gpus": gpu_list(),
        "summary": summary, "requests": results}, indent=2,
        ensure_ascii=False), encoding="utf-8")
    print("record: %s" % f)
    return 0 if ok else 1


def stop(root, name):
    pid = read_pid(root, name)
    f = state_dir(root, name) / "pid"
    if not pid:
        f.unlink(missing_ok=True)
        print("%s is not running" % name)
        return 0
    kill_tree(pid)
    for _ in range(20):
        if not alive(pid):
            break
        time.sleep(0.5)
    if alive(pid) and os.name != "nt":
        import signal
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass
    f.unlink(missing_ok=True)
    _note(root, "stop", name, pid=pid)
    print("%s stopped (pid %d)" % (name, pid))
    return 0


def tail(path, lines):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return ""
    rows = text.splitlines(True)[-lines:]
    return "".join(rows)


def status_line(root, name, cfg):
    eng = ENGINES[cfg["engine"]]
    pid = read_pid(root, name)
    bits = ["%-12s %-9s" % (name, cfg["engine"])]
    if pid:
        bits.append("running (pid %d)" % pid)
    else:
        owner = port_owner(int(cfg["port"])) if port_in_use(cfg) else None
        bits.append("not started here" + (
            "; port %s is used by process %d" % (cfg["port"], owner)
            if owner else "; port %s is in use" % cfg["port"]
            if port_in_use(cfg) else ""))
    code, _ = _get(url(cfg, health_path(cfg)), timeout=3)
    bits.append("health %s" % (code if code else "no answer"))
    if code == 200 and eng.get("models"):
        mc, text = _get(url(cfg, eng["models"]), headers=_auth(cfg),
                        timeout=3)
        try:
            data = json.loads(text)
            ids = [m.get("id") or m.get("name") for m in
                   data.get("data") or data.get("models") or []]
            bits.append("models: %s" % (", ".join(i for i in ids if i)
                                        or "none"))
        except (ValueError, AttributeError):
            pass
    if pid:
        mib = gpu_memory(group_pids(pid))
        if mib is not None:
            bits.append("GPU memory %d MiB" % mib)
    bits.append(url(cfg, eng.get("base") or ""))
    return "  ".join(bits)


# --------------------------------------------------------------- services
def systemd_unit(root, name, cfg):
    argv, env = command(cfg)
    exe = shutil.which(argv[0]) or argv[0]
    argv = [exe] + argv[1:]
    var = cfg.get("api_key_env")
    lines = ["[Unit]", "Description=Tawjih server %s (%s)" % (name,
                                                              cfg["engine"]),
             "After=network-online.target", "", "[Service]", "Type=simple",
             "WorkingDirectory=%s" % (cfg.get("dir") or Path(root).resolve())]
    for k, v in env.items():
        if not v.startswith("$"):
            lines.append("Environment=%s=%s" % (k, v))
    if var:
        # systemd puts variables in ExecStart, not in Environment= lines:
        # a key the engine reads from its environment goes in the file
        # under the engine's name, one given as an argument under VAR
        held = [k for k, v in env.items() if v.startswith("$")] or [var]
        lines.append("# /etc/tawjih/%s.env holds %s=<the key>, readable "
                     "only by root" % (name, held[0]))
        lines.append("EnvironmentFile=/etc/tawjih/%s.env" % name)
    words = []
    for a in argv:
        words.append("${%s}" % a[1:] if a.startswith("$") else
                     shlex.quote(a))
    if not os.path.isabs(exe):
        lines.append("# %s was not found here: give its full path" % exe)
    lines += ["ExecStart=" + " ".join(words), "Restart=on-failure",
              "RestartSec=10", "User=%s" % getpass.getuser(), "",
              "[Install]", "WantedBy=multi-user.target", ""]
    return "\n".join(lines)


def docker_command(name, cfg, image):
    c = dict(cfg, host="0.0.0.0", python="python3", exe=None)
    argv, env = command(c)
    gpus = cfg.get("gpus")
    words = ["docker", "run", "-d", "--name", "tawjih-" + name, "--restart",
             "unless-stopped", "--gpus",
             "'\"device=%s\"'" % gpus if gpus else "all",
             "--shm-size", "16g", "-p", "%s:%s" % (cfg["port"], cfg["port"])]
    model = cfg.get("model") or ""
    if model and Path(model).exists():
        folder = Path(model).resolve()
        folder = folder if folder.is_dir() else folder.parent
        words += ["-v", "%s:%s" % (folder, folder)]
    for k, v in env.items():
        if k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
            continue                    # --gpus chooses the devices
        if v.startswith("$"):
            words += ["-e", "%s=%s" % (k, v)]
        else:
            words += ["-e", "%s=%s" % (k, shlex.quote(v))]
    words += ["--entrypoint", "''", image]
    words += [a if a.startswith("$") else shlex.quote(a) for a in argv]
    return " ".join(words)


def as_provider(root, name, cfg, pname=None):
    eng = ENGINES[cfg["engine"]]
    kind = cfg.get("provider_kind", eng["kind"])
    if not kind or kind == "none":
        die("%s does not answer as a chat model (no OpenAI-compatible or "
            "Ollama interface); use it through an engine that does" % name)
    pname = pname or name
    data = load_providers(root)
    p = dict(data["providers"].get(pname) or {})
    p.update(kind=kind, base_url=url(cfg, eng.get("base") or ""))
    model = cfg.get("served_name") or cfg.get("model")
    if model and cfg["engine"] not in ("litellm", "triton"):
        p["model"] = model
    if cfg.get("api_key_env"):
        p["api_key_env"] = cfg["api_key_env"]
    if kind == "ollama" and cfg.get("context"):
        p["num_ctx"] = int(cfg["context"])
    data["providers"][pname] = p
    if not data.get("default"):
        data["default"] = pname
    _save_providers(root, data)
    print("provider %s: %s at %s%s (use: --backend %s)" % (
        pname, kind, p["base_url"], ", model %s" % p["model"]
        if p.get("model") else "", pname))


# ---------------------------------------------------------------- command
def _settings(a, cfg):
    for k in COMMON + ("exe", "python", "dir", "image", "command",
                       "health", "provider_kind", "min_free"):
        v = getattr(a, "srv_" + k, None)
        if v is not None:
            cfg[k] = v if v != "" else None
    if a.srv_pass:
        cfg["pass"] = list(a.srv_pass)
    if a.srv_env:
        env = dict(cfg.get("env") or {})
        env.update(key_values(a.srv_env, "--env"))
        cfg["env"] = {k: v for k, v in env.items() if v is not None}
    return {k: v for k, v in cfg.items() if v not in (None, [], {})}


def cmd_server(a):
    """Model servers of this project (servers.json): add, set, list, show,
    start, stop, restart, status, logs, metrics, api, unit, provider, rm."""
    st = Store(a.project)
    root = st.root
    act, args = a.action, list(a.args)
    data = load_servers(root)
    servers = data["servers"]
    if act == "list":
        if not servers:
            print("no servers yet (tawjih server add NAME --engine vllm "
                  "--model MODEL)")
        for n in sorted(servers):
            cfg = servers[n]
            pid = read_pid(root, n)
            print("%-12s %-9s %-36s port %-6s %s" % (
                n, cfg["engine"], (cfg.get("model") or "")[:36], cfg["port"],
                "running (pid %d)" % pid if pid else "stopped"))
        return 0
    if act == "gpus":
        gpus = gpu_list()
        if gpus is None:
            die("nvidia-smi is not here, so the GPUs cannot be read")
        for g in gpus:
            print("%d  %-28s %6d MiB free of %6d  %3d%% busy" % (
                g["index"], g["name"][:28], g["free"], g["total"], g["busy"]))
        return 0
    if act == "engines":
        for n, e in ENGINES.items():
            takes = [LABEL[k] for k in ("tp", "pp", "context", "mem",
                                         "kv_cache", "served_name")
                     if e["flags"].get(k) or (e.get("env_flags") or {}).get(k)]
            print("%-10s %s%s" % (n, ", ".join(takes) or "-",
                                  "; kv-cache: " + ", ".join(e["kv"])
                                  if e["kv"] else ""))
        print("anything else: --pass ARG (repeatable), given to the engine "
              "as it is")
        return 0
    if not args:
        die("server %s needs a NAME" % act)
    name = args[0]
    if act in ("add", "set"):
        if not re.fullmatch(NAME_RE, name):
            die("a server's name is letters, digits, _ and -")
        if act == "add" and name in servers:
            die("server %s exists (change it with: server set %s)" % (name,
                                                                   name))
        if act == "set" and name not in servers:
            die("no server %s (add it first)" % name)
        cfg = dict(servers.get(name) or {})
        if a.srv_engine:
            cfg["engine"] = a.srv_engine
        if not cfg.get("engine"):
            die("give --engine (%s)" % ", ".join(ENGINES))
        cfg.setdefault("host", "127.0.0.1")
        cfg.setdefault("port", {"ollama": 11434, "tabby": 5000,
                                "litellm": 4000, "lmdeploy": 23333,
                                "sglang": 30000, "aphrodite": 2242,
                                "llamacpp": 8080}.get(cfg["engine"], 8000))
        cfg = _settings(a, cfg)
        check_server(cfg)
        servers[name] = cfg
        save_servers(root, data)
        print("%s %s: %s" % ("added" if act == "add" else "saved", name,
                             show_command(*command(cfg))))
        if a.srv_register:
            as_provider(root, name, cfg)
        return 0
    cfg = get_server(root, name)
    if act == "rm":
        if read_pid(root, name):
            die("%s is running; stop it first" % name)
        del servers[name]
        save_servers(root, data)
        print("removed %s (its log stays in %s)" % (name,
                                                    state_dir(root, name)))
        return 0
    if act == "show":
        eng = ENGINES[cfg["engine"]]
        print(show_command(*command(cfg)))
        for label, path in (("health", health_path(cfg)),
                            ("models", eng.get("models")),
                            ("metrics", eng.get("metrics"))):
            if path:
                print("  %-8s %s" % (label, url(cfg, path)))
        return 0
    if act == "start":
        return start(root, name, cfg, a.wait)
    if act == "stop":
        return stop(root, name)
    if act == "restart":
        stop(root, name)
        return start(root, name, cfg, a.wait)
    if act == "status":
        print(status_line(root, name, cfg))
        return 0
    if act == "logs":
        log = state_dir(root, name) / "log.txt"
        if not log.is_file():
            die("%s has no log yet (it was not started here)" % name)
        print(tail(log, a.lines), end="")
        if a.follow:
            with open(log, "rb") as f:
                f.seek(0, 2)
                try:
                    while True:
                        chunk = f.read()
                        if chunk:
                            sys.stdout.write(chunk.decode("utf-8",
                                                          "replace"))
                            sys.stdout.flush()
                        else:
                            time.sleep(1)
                except KeyboardInterrupt:
                    pass
        return 0
    if act == "metrics":
        path = ENGINES[cfg["engine"]].get("metrics")
        if not path:
            die("%s offers no metrics address known here; read one with: "
                "tawjih server api %s GET /PATH" % (cfg["engine"], name))
        code, text = _get(url(cfg, path), headers=_auth(cfg))
        if code != 200:
            die("metrics: %s" % (code or text))
        rows = [r for r in text.splitlines() if r and not r.startswith("#")]
        if a.grep:
            rows = [r for r in rows if a.grep.lower() in r.lower()]
        print("\n".join(rows))
        return 0
    if act == "api":
        if len(args) < 3:
            die("server api NAME METHOD PATH [--data JSON]")
        method, path = args[1].upper(), args[2]
        if method not in ("GET", "POST", "PUT", "DELETE", "PATCH"):
            die("METHOD is GET, POST, PUT, DELETE or PATCH")
        if not path.startswith("/"):
            die("PATH starts with /")
        if a.data is not None:
            try:
                json.loads(a.data)
            except ValueError as e:
                die("--data is not JSON: %s" % e)
        code, text = _get(url(cfg, path), method=method, data=a.data,
                          headers=_auth(cfg, a.key_header, a.key_env),
                          timeout=a.timeout)
        _note(root, "api", name, method=method, path=path, status=code)
        if code is None:
            die("no answer from %s: %s" % (url(cfg, path), text))
        print(text)
        return 0 if 200 <= code < 300 else 1
    if act == "unit":
        if cfg.get("gpus") == "auto":
            die("a service starts by itself, with no one to choose GPUs: "
                "give their numbers (server set %s --gpus 0,1)" % name)
        d = state_dir(root, name)
        d.mkdir(parents=True, exist_ok=True)
        if a.docker:
            image = a.srv_image or cfg.get("image") or \
                ENGINES[cfg["engine"]].get("image")
            if not image:
                die("give the engine's image with --image (e.g. "
                    "lmsysorg/sglang:TAG)")
            print(docker_command(name, cfg, image))
            if cfg.get("api_key_env"):
                print("# the key is read from %s where you run this"
                      % cfg["api_key_env"])
            print("# logs: docker logs -f tawjih-%s; restart: docker restart "
                  "tawjih-%s; stop: docker stop tawjih-%s" % (name, name,
                                                              name))
            return 0
        unit = systemd_unit(root, name, cfg)
        f = d / ("tawjih-%s.service" % name)
        f.write_text(unit, encoding="utf-8")
        print(unit)
        print("# written to %s. To install it:" % f)
        print("#   sudo cp %s /etc/systemd/system/" % shlex.quote(str(f)))
        print("#   sudo systemctl daemon-reload && sudo systemctl enable "
              "--now tawjih-%s" % name)
        print("#   logs: sudo journalctl -u tawjih-%s -f" % name)
        return 0
    if act == "provider":
        as_provider(root, name, cfg, args[1] if len(args) > 1 else None)
        return 0
    if act == "prepare":
        return prepare(root, name, cfg, a.out, a.tool_arg,
                       args[1] if len(args) > 1 else None)
    if act == "bench":
        given = [x for x in (a.prompt, a.prompt_file, a.from_file) if x]
        if len(given) != 1:
            die("bench takes one of --prompt TEXT, --prompt-file FILE or "
                "--from-file REF (a file of this project: its machine side)")
        if a.prompt_file:
            prompt = Path(a.prompt_file).read_text(encoding="utf-8")
            source = "prompt file " + a.prompt_file
        elif a.from_file:
            n = st.nodes[st.resolve(a.from_file)]
            prompt = build_prompt(st, n, a.input or "", "")
            source = "file %s (%s)" % (n.id, n.title)
        else:
            prompt, source = a.prompt, "prompt"
        return bench(root, name, cfg, prompt, a.requests, a.concurrency,
                     a.max_tokens, a.vary, a.timeout, source)
    die("unknown action %s" % act)


def add_server_parser(sub):
    s = sub.add_parser("server", aliases=["خادم"], help="model servers: "
                       "start, watch and stop vLLM, SGLang, LMDeploy, "
                       "TensorRT-LLM, Triton, Aphrodite, TabbyAPI, LiteLLM, "
                       "Ollama, llama.cpp or any command")
    s.add_argument("action", choices=["engines", "list", "add", "set", "show",
                                      "start", "stop", "restart", "status",
                                      "logs", "metrics", "api", "unit",
                                      "provider", "rm", "gpus", "prepare",
                                      "bench"])
    s.add_argument("args", nargs="*", help="NAME (api: NAME METHOD PATH; "
                   "provider: NAME [PROVIDER]; prepare: NAME [STEP])")
    s.add_argument("--engine", dest="srv_engine", choices=list(ENGINES))
    s.add_argument("--model", dest="srv_model", help="model id or folder "
                   "(litellm: its config file; triton: the model "
                   "repository; tabby: the model's folder name)")
    s.add_argument("--host", dest="srv_host", help="address to listen on "
                   "(default 127.0.0.1: this computer only)")
    s.add_argument("--port", dest="srv_port", type=int)
    s.add_argument("--gpus", dest="srv_gpus", metavar="0,1",
                   help="the GPUs it may use (CUDA_VISIBLE_DEVICES and "
                   "HIP_VISIBLE_DEVICES), or auto: those with the most free "
                   "memory when it starts, as many as --tp x --pp")
    s.add_argument("--tp", dest="srv_tp", type=int, metavar="N",
                   help="tensor parallel: one model split across N GPUs")
    s.add_argument("--pp", dest="srv_pp", type=int, metavar="N",
                   help="pipeline parallel: the layers split in N stages")
    s.add_argument("--context", dest="srv_context", type=int,
                   metavar="TOKENS", help="the longest context it serves")
    s.add_argument("--mem", dest="srv_mem", type=float, metavar="FRACTION",
                   help="the share of GPU memory it may take (0.9)")
    s.add_argument("--kv-cache", dest="srv_kv_cache", metavar="TYPE",
                   help="a smaller KV cache: fp8, int8, q8_0 ... (see: "
                   "server engines)")
    s.add_argument("--served-name", dest="srv_served_name",
                   help="the model's name in requests")
    s.add_argument("--api-key-env", dest="srv_api_key_env", metavar="VAR",
                   help="the NAME of the environment variable with the "
                   "server's key; the key is never written to a file")
    s.add_argument("--pass", dest="srv_pass", action="append", metavar="ARG",
                   help="an argument given to the engine as it is "
                   "(repeatable): --pass=--enable-prefix-caching")
    s.add_argument("--env", dest="srv_env", action="append",
                   metavar="KEY=VALUE", help="an environment variable for "
                   "the engine (repeatable; not for keys)")
    s.add_argument("--exe", dest="srv_exe", metavar="PATH",
                   help="the engine's program, when it is not on PATH")
    s.add_argument("--python", dest="srv_python", metavar="PATH",
                   help="sglang, tabby: the Python that has the engine")
    s.add_argument("--dir", dest="srv_dir", metavar="FOLDER",
                   help="the folder it runs in (tabby: TabbyAPI's folder)")
    s.add_argument("--command", dest="srv_command", metavar="\"PROG ARGS\"",
                   help="custom: the command line; {model}, {host} and "
                   "{port} are filled in")
    s.add_argument("--health", dest="srv_health", metavar="PATH",
                   help="custom: the health address (default /health)")
    s.add_argument("--provider-kind", dest="srv_provider_kind",
                   choices=["openai", "ollama", "none"],
                   help="custom: how it answers (default openai)")
    s.add_argument("--image", dest="srv_image", help="unit --docker: the "
                   "engine's container image")
    s.add_argument("--provider", dest="srv_register", action="store_true",
                   help="add, set: also make it a provider of this project")
    s.add_argument("--wait", type=int, default=0, metavar="SECONDS",
                   help="start: wait until it answers its health check")
    s.add_argument("--lines", type=int, default=40, help="logs: how many")
    s.add_argument("--follow", "-f", action="store_true",
                   help="logs: keep showing new lines (Ctrl-C ends)")
    s.add_argument("--grep", help="metrics: only lines with this text")
    s.add_argument("--data", help="api: a JSON body")
    s.add_argument("--key-header", metavar="NAME",
                   help="api: the header for the key (default "
                   "Authorization: Bearer)")
    s.add_argument("--key-env", metavar="VAR", help="api: another "
                   "variable's key, e.g. an admin key")
    s.add_argument("--timeout", type=float, default=30,
                   help="api: seconds to wait")
    s.add_argument("--min-free", dest="srv_min_free", type=int,
                   metavar="MIB", help="--gpus auto: only GPUs with this "
                   "much memory free")
    s.add_argument("--out", help="prepare: the folder for the result")
    s.add_argument("--tool-arg", action="append", metavar="ARG",
                   help="prepare: an argument given to the program as it is "
                   "(repeatable)")
    s.add_argument("--prompt", help="bench: the prompt")
    s.add_argument("--prompt-file", help="bench: a file with the prompt")
    s.add_argument("--from-file", metavar="REF", help="bench: the machine "
                   "side of a file of this project, as it is sent when it "
                   "runs")
    s.add_argument("--input", help="bench --from-file: the input it gets")
    s.add_argument("--requests", type=int, default=8,
                   help="bench: how many requests (default 8)")
    s.add_argument("--concurrency", type=int, default=1,
                   help="bench: how many at once (default 1)")
    s.add_argument("--max-tokens", type=int, default=256,
                   help="bench: the longest answer (default 256)")
    s.add_argument("--vary", action="store_true", help="bench: make each "
                   "prompt different, so no answer comes from a cache")
    s.add_argument("--docker", action="store_true",
                   help="unit: a docker run command instead of a systemd "
                   "unit")
    s.set_defaults(fn=cmd_server)
    return s


__all__ = ['ENGINES', 'add_server_parser', 'cmd_server', 'command',
           'load_servers', 'port_owner']
