"""tawjih health: one report on everything a run depends on.

The project's limits, each provider (its key, whether the policy allows it,
whether it answers, its breaker), each model server, the GPUs, the disk and
the libraries that are installed. Nothing is sent to a model: a provider is
asked for its list of models, which costs nothing.

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import importlib.metadata
import importlib.util
import json
import os
import shutil
import sys
import urllib.parse
from pathlib import Path

from .core import Die, Store
from .providers import (
    BACKENDS, NETWORK_KINDS, POLICY_FILE, _is_local, breaker_open,
    breaker_settings, load_breakers, load_policy, load_providers,
    provider_url,
)
from .serving import (
    ENGINES, _get, gpu_list, health_path, load_servers, read_pid,
    url as srv_url,
)

OK, WARN, FAIL, INFO = "ok", "warn", "FAIL", "-"
KEY_ENV = {"openai": "TAWJIH_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
           "gemini": "GEMINI_API_KEY"}
# the libraries of the nine layers that Tawjih drives: (module, package)
LIBRARIES = (
    ("torch", "torch"), ("transformers", "transformers"), ("trl", "trl"),
    ("peft", "peft"), ("accelerate", "accelerate"),
    ("datasets", "datasets"), ("bitsandbytes", "bitsandbytes"),
    ("vllm", "vllm"), ("sglang", "sglang"), ("lmdeploy", "lmdeploy"),
    ("tensorrt_llm", "tensorrt_llm"), ("llama_cpp", "llama-cpp-python"),
    ("awq", "autoawq"), ("tkinter", None),
)


def _models_request(kind, base, cfg):
    """(address, headers) of the free 'list the models' call, or None."""
    env = cfg.get("api_key_env") or KEY_ENV.get(kind)
    key = os.environ.get(env, "") if env else ""
    if kind == "gemini" and not key and not cfg.get("api_key_env"):
        key = os.environ.get("GOOGLE_API_KEY", "")
    base = base.rstrip("/")
    if kind == "ollama":
        return base + "/api/tags", {}
    if kind == "openai":
        return base + "/models", ({"Authorization": "Bearer " + key}
                                  if key else {})
    if kind == "anthropic":
        return base + "/v1/models", {"x-api-key": key,
                                     "anthropic-version": "2023-06-01"}
    if kind == "gemini":
        return base + "/models?" + urllib.parse.urlencode({"key": key}), {}
    return None


def _model_ids(text):
    try:
        data = json.loads(text)
    except ValueError:
        return None
    items = data.get("data") or data.get("models") or []
    ids = set()
    for m in items if isinstance(items, list) else []:
        if isinstance(m, dict):
            for k in ("id", "name", "model"):
                if m.get(k):
                    ids.add(str(m[k]).split("/")[-1])
    return ids


def check_provider(root, name, cfg, policy, local_only):
    """[(state, text)] for one provider."""
    out = _reach(root, name, cfg, policy, local_only)
    if cfg.get("rate"):
        out.append((INFO, "rate %s" % cfg["rate"]))
    bs = breaker_settings(cfg)
    if bs:
        left = breaker_open(root, name)
        n = (load_breakers(root).get(name) or {}).get("failures", 0)
        half = not left and n >= bs[0]
        out.append((WARN if left or half else OK,
                    "breaker %s (%d/%d failures)%s" % (
                        "OPEN" if left else "half-open" if half else "closed",
                        n, bs[0], ", %.0f s left" % left if left else
                        ": the next call decides" if half else "")))
    return out


def _reach(root, name, cfg, policy, local_only):
    kind = cfg.get("kind", name)
    out = []
    if kind not in NETWORK_KINDS:
        return [(OK, "built in (%s)" % kind)]
    base = provider_url(cfg)
    if policy["allow_network"] is False:
        return [(FAIL, "the policy (%s) allows no network provider" %
                 POLICY_FILE)]
    if policy["allow_network"] == "local" and not _is_local(base):
        return [(FAIL, "the policy allows only this computer; it is at %s"
                 % base)]
    env = cfg.get("api_key_env") or KEY_ENV.get(kind)
    has_key = bool(env and os.environ.get(env)) or (
        kind == "gemini" and not cfg.get("api_key_env") and
        bool(os.environ.get("GOOGLE_API_KEY")))
    if env:
        need = kind in ("anthropic", "gemini") or not _is_local(base)
        out.append((OK if has_key else FAIL if need else INFO,
                    "key in %s: %s" % (env, "set" if has_key else "NOT set")))
    if local_only and not _is_local(base):
        out.append((INFO, "%s not asked (--local)" % base))
    else:
        req = _models_request(kind, base, cfg)
        code, text = _get(req[0], headers=req[1], timeout=5)
        where = base
        if code == 200:
            ids = _model_ids(text)
            model = cfg.get("model")
            if model and ids is not None and kind in ("ollama", "openai") \
                    and model.split("/")[-1] not in ids \
                    and model.split(":")[0] not in {i.split(":")[0]
                                                    for i in ids}:
                out.append((WARN, "%s answers, but has no model %s (%s)" % (
                    where, model, ", ".join(sorted(ids)[:6]) or "none")))
            else:
                out.append((OK, "%s answers%s" % (
                    where, " (%d models)" % len(ids) if ids else "")))
        elif code in (401, 403):
            out.append((FAIL, "%s refuses the key (%s)" % (where, code)))
        elif code is None:
            out.append((FAIL, "%s does not answer: %s" % (where, text[:80])))
        else:
            out.append((WARN, "%s answers %s" % (where, code)))
    return out


def check_server(root, name, cfg):
    eng = ENGINES.get(cfg.get("engine"), {})
    pid = read_pid(root, name)
    code, _ = _get(srv_url(cfg, health_path(cfg)), timeout=3)
    where = srv_url(cfg, eng.get("base") or "")
    if code == 200:
        return [(OK, "%s %s healthy%s" % (cfg.get("engine"), where,
                                          " (pid %d)" % pid if pid else ""))]
    if pid:
        return [(FAIL, "%s started (pid %d) but health is %s" % (
            cfg.get("engine"), pid, code or "no answer"))]
    return [(INFO, "%s not running (%s)" % (cfg.get("engine"), where))]


def libraries():
    out = []
    for mod, dist in LIBRARIES:
        try:
            found = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append((mod, None))
            continue
        ver = ""
        if dist:
            try:
                ver = importlib.metadata.version(dist)
            except importlib.metadata.PackageNotFoundError:
                ver = "?"
        out.append((mod, ver or "with Python"))
    return out


def report(root, local_only=False):
    """{section: [(name, state, text)]}."""
    rep = {"project": [], "providers": [], "servers": [], "computer": [],
           "libraries": []}
    pol = load_policy(root)
    rep["project"].append(("policy", INFO, ", ".join(
        "%s %s" % (k, json.dumps(v)) for k, v in pol.items())))
    rep["project"].append(("redaction", INFO, "on (redact.txt)" if (
        Path(root) / "redact.txt").is_file() else "off"))
    try:
        data = load_providers(root)
    except Die as e:
        rep["providers"].append(("providers.json", FAIL, str(e)))
        data = {"providers": {}, "fallback": {}, "default": ""}
    provs = data["providers"]
    names = sorted(set(provs) | {x for v in data["fallback"].values()
                                 for x in v})
    if not names:
        names = [data["default"] or "dry"]
    for n in names:
        cfg = provs.get(n) or {"kind": n}
        if cfg.get("kind", n) not in BACKENDS:
            rep["providers"].append((n, FAIL, "unknown kind %s" %
                                     cfg.get("kind")))
            continue
        label = n + (" (default)" if data["default"] == n else "")
        for state, text in check_provider(root, n, cfg, pol, local_only):
            rep["providers"].append((label, state, text))
    for n, cfg in sorted(load_servers(root)["servers"].items()):
        for state, text in check_server(root, n, cfg):
            rep["servers"].append((n, state, text))
    gpus = gpu_list()
    if gpus is None:
        rep["computer"].append(("GPU", INFO, "no nvidia-smi here"))
    for g in gpus or []:
        rep["computer"].append(("GPU %d" % g["index"], OK, "%s, %d of %d MiB "
                                "free, %d%% busy" % (g["name"], g["free"],
                                                     g["total"], g["busy"])))
    du = shutil.disk_usage(root)
    free_gib = du.free / 2 ** 30
    rep["computer"].append(("disk", WARN if free_gib < 5 else OK,
                            "%.1f GiB free of %.1f GiB" % (
                                free_gib, du.total / 2 ** 30)))
    rep["computer"].append(("python", OK, sys.version.split()[0]))
    for mod, ver in libraries():
        rep["libraries"].append((mod, OK if ver else INFO,
                                 ver or "not installed"))
    return rep


def cmd_health(a):
    st = Store(a.project)
    rep = report(st.root, a.local)
    bad = sum(1 for rows in rep.values() for r in rows if r[1] == FAIL)
    warn = sum(1 for rows in rep.values() for r in rows if r[1] == WARN)
    if a.json:
        print(json.dumps({"failures": bad, "warnings": warn, "sections": {
            k: [{"name": n, "state": s, "text": t} for n, s, t in rows]
            for k, rows in rep.items()}}, ensure_ascii=False))
        return 1 if bad else 0
    for sec, rows in rep.items():
        if not rows:
            continue
        print("%s:" % sec)
        for n, s, t in rows:
            print("  %-4s %-20s %s" % (s, n, t))
    print("%d failing, %d to look at" % (bad, warn))
    return 1 if bad else 0


def add_health_parser(sub):
    s = sub.add_parser("health", aliases=["صحة"], help="one report: "
                       "the policy, each provider (key, answer, breaker), "
                       "each model server, the GPUs, the disk and the "
                       "installed libraries; nothing is sent to a model")
    s.add_argument("--local", action="store_true", help="ask nothing "
                   "outside this computer (remote providers are only "
                   "checked for their key)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_health)
    return s

__all__ = ["LIBRARIES", "add_health_parser", "check_provider",
           "check_server", "cmd_health", "libraries", "report"]
