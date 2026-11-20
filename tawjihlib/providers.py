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
import urllib.request
from pathlib import Path

from .core import (
    die,
)


# ------------------------------------------------------------------ backends
def _post(url, body, headers):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers=dict({"Content-Type": "application/json"}, **headers))
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError("HTTP %d from %s: %s" % (e.code, url, detail))
    except urllib.error.URLError as e:
        raise RuntimeError("Connection failed to %s: %s" % (url, e.reason))


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


def _note_usage(tokens_in, tokens_out):
    _usage.last = {"in": int(tokens_in or 0), "out": int(tokens_out or 0)}


def backend_dry(prompt, model=""):
    m_tag = (" [model: %s]" % model) if model else ""
    return "[dry run%s: no model was called; prompt was %d characters]" % (
        m_tag, len(prompt))


def backend_ollama(prompt, model=""):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    data = _post(host + "/api/chat",
                 {"model": _model(model), "stream": False,
                  "messages": [{"role": "user", "content": prompt}]}, {})
    _note_usage(data.get("prompt_eval_count"), data.get("eval_count"))
    return data["message"]["content"]


def backend_anthropic(prompt, model=""):
    key = os.environ.get("ANTHROPIC_API_KEY") or die("set ANTHROPIC_API_KEY")
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    data = _post(base.rstrip("/") + "/v1/messages",
                 {"model": _model(model),
                  "max_tokens": int(os.environ.get("TAWJIH_MAX_TOKENS", "4096")),
                  "messages": [{"role": "user", "content": prompt}]},
                 {"x-api-key": key, "anthropic-version": "2023-06-01"})
    u = data.get("usage") or {}
    _note_usage(u.get("input_tokens"), u.get("output_tokens"))
    return "".join(b.get("text", "") for b in data.get("content", []))


def backend_openai(prompt, model=""):
    """Any provider exposing the OpenAI-style /chat/completions endpoint."""
    base = os.environ.get("TAWJIH_BASE_URL", "https://api.openai.com/v1")
    key = os.environ.get("TAWJIH_API_KEY", "")
    headers = {"Authorization": "Bearer " + key} if key else {}
    data = _post(base.rstrip("/") + "/chat/completions",
                 {"model": _model(model),
                  "messages": [{"role": "user", "content": prompt}]}, headers)
    u = data.get("usage") or {}
    _note_usage(u.get("prompt_tokens"), u.get("completion_tokens"))
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
        for k, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
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

        def sub(label):
            def f(m):
                val = m.group(0)
                if label == "CARD" and not _luhn(re.sub(r"\D", "", val)):
                    return val
                if val not in found:
                    counts[label] = counts.get(label, 0) + 1
                    found[val] = "[%s_%d]" % (label, counts[label])
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


def call_model(store_root, backend, prompt, model=""):
    """Every model call goes through here: redaction (when the project has
    redact.txt), the call, the placeholders put back, and what it cost.
    Returns (answer, usage)."""
    red = Redactor.load(store_root) if store_root else None
    sent, mapping = red.mask(prompt) if red else (prompt, {})
    _usage.last = None
    t0 = time.monotonic()
    answer = BACKENDS[backend](sent, model=model) or ""
    usage = dict(getattr(_usage, "last", None) or {},
                 seconds=round(time.monotonic() - t0, 3),
                 chars_in=len(sent), chars_out=len(answer),
                 redacted=len(mapping))
    if mapping:
        answer = Redactor.unmask(answer, mapping)
    return answer, usage


__all__ = [
    'BACKENDS', 'REDACT_PATTERNS', 'REDACT_TEMPLATE', 'Redactor', '_luhn',
    '_model', '_note_usage', '_post', '_usage', 'backend_anthropic',
    'backend_dry', 'backend_manual', 'backend_ollama', 'backend_openai',
    'call_model', 'use_model',
]
