"""Fuzz targets: each takes arbitrary bytes and must either succeed or fail
with the error the function documents. Any other exception is a bug.

They are run two ways:
- by tests/test_fuzz.py, with random and mutated inputs from the standard
  library only, on every test run;
- by ClusterFuzzLite with Atheris (fuzz/fuzz_*.py), coverage-guided, on
  pull requests (.clusterfuzzlite/, .github/workflows/cflite.yml).

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
import json
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))

from tawjihlib.core import (  # noqa: E402
    Die, FlowError, build_flow, conform, parse_examples,
)
from tawjihlib.providers import REDACT_PATTERNS, Redactor  # noqa: E402
import re  # noqa: E402


def _text(data):
    return data.decode("utf-8", "replace")


def path_language(data):
    """A path file's text: parsed into statements and built into a flow.
    Accepted flows are acyclic or bounded; a refusal is a FlowError."""
    text = _text(data)
    try:
        lines = [l.split("#", 1)[0].strip() for l in text.splitlines()]
        stmts = [shlex.split(l) for l in lines if l]
    except ValueError:
        return                       # an unreadable line: refused as such
    try:
        flow = build_flow(stmts, lambda ref: ref)
    except FlowError:
        return
    # what is accepted must hold together
    for e in flow.edges:
        assert e["src"] in flow.nodes and e["dst"] in flow.nodes, e
        assert e["src"] != e["dst"], e
    if flow.iterate:
        rounds = flow.iterate[0]
        assert isinstance(rounds, int) and rounds >= 1, flow.iterate


CONTRACTS = ["", "text", "json", "json(a, b)", "json(a)", "none",
             "json(x, y, z)", "schema(missing)"]


def contract(data):
    """An answer checked against a contract: a pair (text, problem), and
    for JSON contracts the text passed on is JSON."""
    text = _text(data)
    spec = CONTRACTS[data[0] % len(CONTRACTS)] if data else ""
    if data and data[0] >= 128:
        spec = _text(data[:16])      # contract text of its own
    try:
        out, problem = conform(text, spec, "output")
    except Die:
        return                       # a contract that cannot be used
    assert isinstance(out, str)
    assert problem is None or isinstance(problem, str)
    if problem is None and spec.startswith("json"):
        json.loads(out)


def examples(data):
    """The examples side: parsed into examples and problems, never raised."""
    out, problems = parse_examples(_text(data))
    assert isinstance(out, list) and isinstance(problems, list)
    for ex in out:
        assert isinstance(ex, dict)


_REDACT = Redactor([(name.upper(), re.compile(p))
                    for name, p in REDACT_PATTERNS.items()])


def redaction(data):
    """What is masked before a model sees it comes back unchanged."""
    text = _text(data)
    sent, mapping = _REDACT.mask(text)
    assert Redactor.unmask(sent, mapping) == text


TARGETS = {"path_language": path_language, "contract": contract,
           "examples": examples, "redaction": redaction}
