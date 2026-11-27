"""Model providers (backends), usage records and redaction.

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from .core import (
    die,
)


# ------------------------------------------------------------------ backends
class ProviderError(RuntimeError):
    """A failed call to a provider: an HTTP status, or none when the
    connection itself failed or timed out."""
    def __init__(self, message, status=None, retry_after=None):
        super().__init__(message)
        self.status, self.retry_after = status, retry_after


_conf = threading.local()      # settings of the provider being called


def _setting(key, env=None, default=None):
    """A provider setting from providers.json, else the environment."""
    v = (getattr(_conf, "cfg", None) or {}).get(key)
    if v not in (None, ""):
        return v
    return os.environ.get(env, default) if env else default


def _api_key(default_env):
    """The key is read from an environment variable, never from a file:
    providers.json names the variable (api_key_env)."""
    return os.environ.get(_setting("api_key_env") or default_env, "")


def _post(url, body, headers):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers=dict({"Content-Type": "application/json"}, **headers))
    timeout = float(_setting("timeout", default=600))
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        after = e.headers.get("Retry-After") if e.headers else None
        raise ProviderError("HTTP %d from %s: %s" % (e.code, url, detail),
                            e.code, after)
    except urllib.error.URLError as e:
        raise ProviderError("Connection failed to %s: %s" % (url, e.reason))
    except (TimeoutError, OSError) as e:
        raise ProviderError("No answer from %s: %s" % (url, e))


def _model(override=""):
    m = (override or "").strip() or os.environ.get("TAWJIH_MODEL")
    if not m:
        die("name the model: pass --model NAME, set model in node header, "
            "or set the TAWJIH_MODEL environment variable")
    return m


def use_model(a):
    """--model on the command line wins over the environment variable."""
    if getattr(a, "model", None):
        os.environ["TAWJIH_MODEL"] = a.model


_usage = threading.local()     # token counts of the last answer, per thread


def _note_usage(tokens_in, tokens_out, served=None):
    """Token counts of the last answer, and the model the provider says
    answered (often a dated or digested version of the name asked for)."""
    _usage.last = {"in": int(tokens_in or 0), "out": int(tokens_out or 0)}
    if isinstance(served, str) and served:
        _usage.last["served"] = served[:200]


def backend_dry(prompt, model=""):
    m_tag = (" [model: %s]" % model) if model else ""
    return "[dry run%s: no model was called; prompt was %d characters]" % (
        m_tag, len(prompt))


def _options():
    """Settings sent to the server as they are (providers.json "options"):
    anything the server takes that Tawjih has no word for."""
    opts = _setting("options") or {}
    return dict(opts) if isinstance(opts, dict) else {}


def _extra(body):
    """The request body with the provider's options added; the model and
    the messages stay Tawjih's."""
    for k, v in _options().items():
        if k not in ("model", "messages", "stream"):
            body[k] = v
    return body


_told_ctx = set()


def _context_warning(prompt, num_ctx, where):
    """Ollama keeps only as many tokens as its context holds and drops the
    rest without an error. A token is about 4 characters of English (fewer
    for Arabic), so a prompt of more than 4 characters per token of context
    may not fit. Said once per provider and size."""
    est = len(prompt) // 4
    limit = int(num_ctx) if num_ctx else 4096
    if est <= limit:
        return
    key = (where, limit)
    if key in _told_ctx:
        return
    _told_ctx.add(key)
    if num_ctx:
        msg = ("the prompt is about %d tokens or more, and num_ctx is %d: "
               "Ollama keeps only the last %d tokens, so the model may not "
               "see all of it. Raise num_ctx (tawjih provider set NAME "
               "--num-ctx N)" % (est, limit, limit))
    else:
        msg = ("the prompt is about %d tokens or more. Ollama's context is "
               "4096 tokens by default on GPUs with less than 24 GiB, and a "
               "longer prompt is cut without an error. Set num_ctx (tawjih "
               "provider set NAME --num-ctx N, or TAWJIH_NUM_CTX)" % est)
    print("warning: %s: %s" % (where, msg), file=sys.stderr)


def backend_ollama(prompt, model=""):
    host = _setting("base_url", "OLLAMA_HOST",
                    "http://localhost:11434").rstrip("/")
    if not re.match(r"https?://", host):
        host = "http://" + host         # OLLAMA_HOST is often host:port
    opts = _options()
    num_ctx = _setting("num_ctx", "TAWJIH_NUM_CTX") or opts.get("num_ctx")
    if num_ctx:
        try:
            opts["num_ctx"] = int(num_ctx)
        except (TypeError, ValueError):
            die("num_ctx must be a whole number, not %r" % num_ctx)
    _context_warning(prompt, opts.get("num_ctx"), "ollama at " + host)
    body = {"model": _model(model), "stream": False,
            "messages": [{"role": "user", "content": prompt}]}
    if opts:
        body["options"] = opts
    data = _post(host + "/api/chat", body, {})
    _note_usage(data.get("prompt_eval_count"), data.get("eval_count"),
                data.get("model"))
    return data["message"]["content"]


def backend_anthropic(prompt, model=""):
    key = _api_key("ANTHROPIC_API_KEY") or die(
        "set %s" % (_setting("api_key_env") or "ANTHROPIC_API_KEY"))
    base = _setting("base_url", "ANTHROPIC_BASE_URL",
                    "https://api.anthropic.com")
    data = _post(base.rstrip("/") + "/v1/messages",
                 _extra({"model": _model(model),
                         "max_tokens": int(os.environ.get(
                             "TAWJIH_MAX_TOKENS", "4096")),
                         "messages": [{"role": "user", "content": prompt}]}),
                 {"x-api-key": key, "anthropic-version": "2023-06-01"})
    u = data.get("usage") or {}
    _note_usage(u.get("input_tokens"), u.get("output_tokens"),
                data.get("model"))
    return "".join(b.get("text", "") for b in data.get("content", []))


def backend_openai(prompt, model=""):
    """Any provider exposing the OpenAI-style /chat/completions endpoint."""
    base = _setting("base_url", "TAWJIH_BASE_URL",
                    "https://api.openai.com/v1")
    key = _api_key("TAWJIH_API_KEY")
    headers = {"Authorization": "Bearer " + key} if key else {}
    data = _post(base.rstrip("/") + "/chat/completions",
                 _extra({"model": _model(model),
                         "messages": [{"role": "user", "content": prompt}]}),
                 headers)
    u = data.get("usage") or {}
    _note_usage(u.get("prompt_tokens"), u.get("completion_tokens"),
                data.get("model"))
    return data["choices"][0]["message"]["content"]


def backend_manual(prompt, model=""):
    """A person (or any chat model) plays the model: paste the answer."""
    bar = "=" * 60
    if model:
        print("\n[suggested model: %s]" % model, file=sys.stderr)
    print("\n%s\n%s\n%s" % (bar, prompt, bar), file=sys.stderr)
    print("answer (finish with a line containing only '.'):", file=sys.stderr)
    lines = []
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == ".":
            break
        lines.append(line.rstrip("\n"))
    return "\n".join(lines).strip()


BACKENDS = {"dry": backend_dry, "manual": backend_manual,
            "openai": backend_openai, "ollama": backend_ollama,
            "anthropic": backend_anthropic}
NETWORK_KINDS = ("ollama", "openai", "anthropic")
DEFAULT_POLICY = {"timeout": 600, "retries": 2,
                  "retry_status": [429, 500, 502, 503, 504],
                  "backoff": "exponential", "backoff_seconds": 2.0}
PROVIDER_KEYS = ("kind", "base_url", "model", "api_key_env", "timeout",
                 "retries", "retry_status", "backoff", "backoff_seconds",
                 "num_ctx", "options")


# ---------------------------------------------------------------- providers
def providers_file(root):
    return Path(root) / "providers.json"


def load_providers(root):
    """providers.json: named providers (a kind such as ollama or openai, a
    base URL, a model, the NAME of the environment variable holding the key,
    and a retry policy), the default, and fallback chains."""
    data = {}
    f = providers_file(root) if root else None
    if f and f.is_file():
        try:
            data = json.loads(f.read_text(encoding="utf-8-sig"))
        except ValueError as e:
            die("providers.json is not readable JSON (%s)" % e)
    data.setdefault("providers", {})
    data.setdefault("fallback", {})
    data.setdefault("default", "")
    return data


def default_backend(root):
    """The provider used when none is named: providers.json's default, or
    the dry run."""
    return load_providers(root).get("default") or "dry"


def backend_kind(root, name):
    """The kind (dry, manual, ollama ...) behind a provider name."""
    if name in BACKENDS:
        provs = load_providers(root)["providers"]
        return (provs.get(name) or {}).get("kind", name)
    return provider_chain(root, name, False)[0][1]["kind"]


def provider_chain(root, name, fallback=True):
    """[(name, settings)] to try in turn for a provider or built-in kind."""
    data = load_providers(root)
    provs = data["providers"]
    if name not in provs and name not in BACKENDS:
        die("unknown provider '%s' (built in: %s%s)" % (
            name, ", ".join(sorted(BACKENDS)), "; in providers.json: " +
            ", ".join(sorted(provs)) if provs else ""))
    chain, seen = [], set()
    todo = [name] + (list(data["fallback"].get(name, [])) if fallback
                     else [])
    for n in todo:
        if n in seen:
            continue
        seen.add(n)
        cfg = dict(provs.get(n) or {"kind": n})
        if cfg.get("kind") not in BACKENDS:
            die("provider %s has an unknown kind '%s'" % (n, cfg.get("kind")))
        chain.append((n, cfg))
    return chain


# ---------------------------------------------------------------- redaction
REDACT_PATTERNS = {
    "email": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    "phone": r"(?<![\w+])(?:(?:\+|00)\d{1,3}[\s.-]?\d(?:[\s.-]?\d){6,12}"
             r"|05\d(?:[\s-]?\d){7})(?!\w)",
    "card": r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)",
    "iban": r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,3})?\b",
    "national_id": r"(?<!\d)[12]\d{9}(?!\d)",
    "ip": r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}"
          r"(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])",
}
REDACT_TEMPLATE = r"""# Tawjih redaction: what is replaced by a placeholder such as [EMAIL_1]
# before any text is sent to a model, and put back in the answer.
# Delete this file to turn redaction off.
#
# Built-in patterns (delete a line to leave that kind alone):
@email
@phone
@card
@iban
@national_id
@ip
#
# Your own words and names, one per line (matched without regard to case),
# or a regular expression after 're:'.
# Mohammed Khalid
# re: Project\s+Falcon\w*
"""


def _luhn(digits):
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


class Redactor:
    """Replaces private text with placeholders before a model sees it, and
    puts it back in the answer. Patterns cannot know what is private: what
    they miss is sent as it is."""
    def __init__(self, rules):
        self.rules = rules            # [(label, compiled pattern)]

    @staticmethod
    def load(root):
        f = Path(root) / "redact.txt"
        if not f.is_file():
            return None
        rules, words = [], []
        for k, line in enumerate(f.read_text(encoding="utf-8-sig").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("@"):
                name = line[1:].strip()
                if name not in REDACT_PATTERNS:
                    die("redact.txt line %d: unknown pattern @%s (%s)" % (
                        k, name, ", ".join(sorted(REDACT_PATTERNS))))
                rules.append((name.upper(), re.compile(REDACT_PATTERNS[name])))
            elif line.startswith("re:"):
                try:
                    rules.append(("PRIVATE", re.compile(line[3:].strip())))
                except re.error as e:
                    die("redact.txt line %d: %s" % (k, e))
            else:
                words.append(line)
        # long, specific patterns first, so a phone number cannot take a
        # piece of an IBAN or a card
        order = ["IBAN", "CARD", "EMAIL", "IP", "NATIONAL_ID", "PHONE"]
        rules.sort(key=lambda r: order.index(r[0]) if r[0] in order else -1)
        if words:   # longest first, so a full name wins over a part of it
            rules.insert(0, ("PRIVATE", re.compile("|".join(
                re.escape(w) for w in sorted(words, key=len, reverse=True)),
                re.I)))
        return Redactor(rules)

    def mask(self, text):
        found = {}                    # original -> placeholder
        counts = {}
        given = text                  # a placeholder never already in it

        def sub(label):
            def f(m):
                val = m.group(0)
                if label == "CARD" and not _luhn(re.sub(r"\D", "", val)):
                    return val
                if val not in found:
                    while True:
                        counts[label] = counts.get(label, 0) + 1
                        ph = "[%s_%d]" % (label, counts[label])
                        if ph not in given:
                            break
                    found[val] = ph
                return found[val]
            return f
        for label, pat in self.rules:
            text = pat.sub(sub(label), text)
        return text, {v: k for k, v in found.items()}

    @staticmethod
    def unmask(text, mapping):
        for ph in sorted(mapping, key=len, reverse=True):
            text = text.replace(ph, mapping[ph])
        return text


def _wait(policy, attempt, err):
    if err.retry_after and str(err.retry_after).isdigit():
        return min(60.0, float(err.retry_after))
    base = float(policy.get("backoff_seconds", 2.0))
    if policy.get("backoff") == "fixed":
        return min(60.0, base)
    return min(60.0, base * (2 ** attempt))


def call_model(store_root, backend, prompt, model="", fallback=True):
    """Every model call goes through here: the project's policy, redaction
    (when the project has redact.txt), the call with its provider's retry
    policy and fallback chain, the placeholders put back, and what it cost.
    Returns (answer, usage)."""
    backend = backend or default_backend(store_root)
    chain = provider_chain(store_root, backend, fallback)
    check_network(store_root, chain)
    red = Redactor.load(store_root) if store_root else None
    sent, mapping = red.mask(prompt) if red else (prompt, {})
    _usage.last = None
    t0 = time.monotonic()
    failures, answer, used, attempts = [], None, None, 0
    for k, (name, cfg) in enumerate(chain):
        policy = dict(DEFAULT_POLICY, **{x: cfg[x] for x in DEFAULT_POLICY
                                         if x in cfg})
        # the step's model on its own provider; a fallback's own model first
        m = (model or cfg.get("model", "")) if k == 0 else \
            (cfg.get("model") or model)
        for attempt in range(int(policy["retries"]) + 1):
            attempts += 1
            _conf.cfg = cfg
            try:
                answer = BACKENDS[cfg["kind"]](sent, model=m) or ""
                used = name
                break
            except ProviderError as e:
                transient = e.status is None or e.status in policy[
                    "retry_status"]
                if transient and attempt < int(policy["retries"]):
                    wait = _wait(policy, attempt, e)
                    print("  %s: %s; trying again in %.0f s" % (
                        name, str(e)[:120], wait), file=sys.stderr)
                    time.sleep(wait)
                    continue
                failures.append("%s: %s" % (name, str(e)[:200]))
                break
            finally:
                _conf.cfg = None
        if used:
            break
        if k + 1 < len(chain):
            print("  %s failed; trying %s" % (name, chain[k + 1][0]),
                  file=sys.stderr)
    if used is None:
        err = ProviderError("every provider failed: " + " | ".join(failures))
        err.attempts = attempts
        raise err
    usage = dict(getattr(_usage, "last", None) or {},
                 seconds=round(time.monotonic() - t0, 3),
                 chars_in=len(sent), chars_out=len(answer),
                 redacted=len(mapping), provider=used, attempts=attempts,
                 sent=sent, answer=answer)   # as they crossed the network
    if mapping:
        answer = Redactor.unmask(answer, mapping)
    return answer, usage


# ------------------------------------------------------------------- policy
POLICY_FILE = "tawjih.policy.json"
POLICY_KEYS = {"allow_code": (True, False),
               "allow_network": (True, False, "local"),
               "sandbox": (True, False),
               "human_approval": ("always", "run", "never")}
POLICY_DEFAULT = {"allow_code": True, "allow_network": True,
                  "sandbox": False, "human_approval": "never"}


def load_policy(root):
    """tawjih.policy.json: limits a project sets for itself. It can only
    restrict: what it allows still needs the usual options (--allow-code,
    --allow-run ...)."""
    f = Path(root) / POLICY_FILE if root else None
    pol = dict(POLICY_DEFAULT)
    if f and f.is_file():
        try:
            data = json.loads(f.read_text(encoding="utf-8-sig"))
        except ValueError as e:
            die("%s is not readable JSON (%s)" % (POLICY_FILE, e))
        for k, v in data.items():
            if k not in POLICY_KEYS or v not in POLICY_KEYS[k]:
                die("%s: %s cannot be %r" % (POLICY_FILE, k, v))
            pol[k] = v
    return pol


def _is_local(url):
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.endswith(
        ".localhost")


def check_network(root, chain):
    """Refuse providers the project's policy does not allow."""
    allow = load_policy(root)["allow_network"]
    if allow is True:
        return
    for name, cfg in chain:
        kind = cfg["kind"]
        if kind not in NETWORK_KINDS:
            continue
        if allow is False:
            die("the project's policy (%s) allows no network provider; %s "
                "is one" % (POLICY_FILE, name))
        url = cfg.get("base_url") or {
            "ollama": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
            "openai": os.environ.get("TAWJIH_BASE_URL",
                                     "https://api.openai.com/v1"),
            "anthropic": os.environ.get("ANTHROPIC_BASE_URL",
                                        "https://api.anthropic.com")}[kind]
        if not _is_local(url):
            die("the project's policy (%s) allows only providers on this "
                "computer; %s is at %s" % (POLICY_FILE, name, url))


__all__ = [
    'BACKENDS', 'DEFAULT_POLICY', 'NETWORK_KINDS', 'POLICY_DEFAULT',
    'POLICY_FILE', 'POLICY_KEYS', 'PROVIDER_KEYS', 'ProviderError', '_conf',
    '_api_key', '_is_local', 'backend_kind', '_setting', '_wait', 'check_network',
    'default_backend', 'load_policy', 'load_providers', 'provider_chain',
    'providers_file', 'REDACT_PATTERNS', 'REDACT_TEMPLATE', 'Redactor', '_luhn',
    '_model', '_note_usage', '_post', '_usage', 'backend_anthropic',
    'backend_dry', 'backend_manual', 'backend_ollama', 'backend_openai',
    'call_model', 'use_model',
]
