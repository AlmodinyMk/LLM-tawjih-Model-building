"""Management statements: CREATE, EDIT, MOVE, COPY, MERGE, DELETE, CHECK,
GC and PATH, in English or Arabic, translated into the program's own
commands (see the command do).

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""

import re
import shlex

from .core import (
    FlowError,
)
from .providers import (
    BACKENDS,
)

# The words of management statements. They are a vocabulary of their own:
# none of them is reserved in the path language, so common Arabic words
# such as في and من stay free there.
OPS = {
    "CREATE": "CREATE", "أنشئ": "CREATE", "انشئ": "CREATE",
    "EDIT": "EDIT", "عدّل": "EDIT", "عدل": "EDIT",
    "MOVE": "MOVE", "انقل": "MOVE",
    "COPY": "COPY", "انسخ": "COPY",
    "MERGE": "MERGE", "ادمج": "MERGE",
    "DELETE": "DELETE", "احذف": "DELETE",
    "CHECK": "CHECK", "افحص": "CHECK",
    "GC": "GC", "نظّف": "GC", "نظف": "GC",
    "PATH": "PATH", "مسار": "PATH",
}
PLACES = {
    "UNDER": "UNDER", "تحت": "UNDER", "INTO": "INTO", "في": "INTO",
    "FROM": "FROM", "من": "FROM", "SIDE": "SIDE", "جانب": "SIDE",
    "USING": "USING", "بواسطة": "USING",
}
SIDE_WORDS = {
    "HUMAN": "human", "بشري": "human",
    "MACHINE": "instructions", "آلي": "instructions", "الي": "instructions",
    "INSTRUCTIONS": "instructions", "تعليمات": "instructions",
    "CODE": "code", "كود": "code",
    "READING": "reading", "قرائي": "reading",
    "EXAMPLES": "examples", "أمثلة": "examples", "امثلة": "examples",
}
SEPARATORS = ("/", ";", "؛")
DESTRUCTIVE = {"DELETE", "GC", "MERGE"}


class ScriptError(Exception):
    pass


def _word(table, tok):
    return table.get(tok.replace("ـ", "").upper())


def split_script(text):
    """Statements: one per line, or separated by '/', ';' or '؛' (outside
    quotes); a line starting with '#' is a comment. A ',' stays inside a statement: in PATH it separates the
    path's own statements."""
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        lex = shlex.shlex(line, posix=True, punctuation_chars=";؛")
        lex.whitespace_split = True
        lex.commenters = ""
        try:
            toks = list(lex)
        except ValueError as e:
            raise ScriptError("unreadable line (%s): %s" % (e, line))
        cur = []
        for t in toks:
            if t in SEPARATORS:
                if cur:
                    out.append(cur)
                cur = []
            else:
                cur.append(t)
        if cur:
            out.append(cur)
    return out


def translate(tokens, backend="dry", yes=False):
    """One management statement as the argument list of a command, and
    whether it removes or overwrites something."""
    if not tokens:
        raise ScriptError("empty statement")
    text = " ".join(tokens)
    op = _word(OPS, tokens[0])
    if op is None:
        raise ScriptError(
            "a statement starts with CREATE, EDIT, MOVE, COPY, MERGE, "
            "DELETE, CHECK, GC or PATH (or their Arabic words); for a "
            "connection between files write: PATH name A TO B. Not "
            "understood: " + text)
    rest = tokens[1:]

    def need(n, what):
        if len(rest) < n:
            raise ScriptError("%s needs %s: %s" % (op, what, text))

    def place(i, allowed):
        if i >= len(rest) or _word(PLACES, rest[i]) not in allowed:
            raise ScriptError("%s expects %s after %s: %s" % (
                op, " or ".join(allowed), rest[i - 1], text))
        if i + 1 >= len(rest):
            raise ScriptError("%s needs a file after %s: %s" % (
                op, rest[i], text))
        return rest[i + 1]

    if op == "CREATE":
        need(1, "a title")
        title, parent, template, i = rest[0], None, None, 1
        while i < len(rest):
            w = _word(PLACES, rest[i])
            if w in ("UNDER", "INTO") and i + 1 < len(rest):
                parent = rest[i + 1]
            elif w == "FROM" and i + 1 < len(rest):
                template = rest[i + 1]
            else:
                raise ScriptError("CREATE \"title\" [UNDER file] [FROM "
                                  "template]: " + text)
            i += 2
        if template:
            argv = ["scaffold", template, "--title", title]
        else:
            argv = ["new", title]
        return argv + (["--parent", parent] if parent else []), False
    if op == "EDIT":
        need(1, "a file")
        ref, side, model, i = rest[0], None, None, 1
        back = backend
        while i < len(rest) and _word(PLACES, rest[i]) in ("SIDE", "USING"):
            if i + 1 >= len(rest):
                raise ScriptError("%s needs a word after it: %s" % (
                    rest[i], text))
            if _word(PLACES, rest[i]) == "SIDE":
                side = _word(SIDE_WORDS, rest[i + 1])
                if side is None:
                    raise ScriptError("the side is HUMAN, MACHINE, CODE, "
                                      "READING or EXAMPLES: " + text)
            else:
                if rest[i + 1].lower() in BACKENDS:
                    back = rest[i + 1].lower()
                else:
                    model = rest[i + 1]
            i += 2
        ask = " ".join(rest[i:]).strip()
        if not side:
            raise ScriptError("EDIT needs SIDE (HUMAN, MACHINE, CODE, "
                              "READING or EXAMPLES): " + text)
        if not ask:
            raise ScriptError("EDIT needs what to ask the model, after the "
                              "side: " + text)
        argv = ["edit", ref, "--side", side, "--ask", ask, "--backend",
                back] + (["--model", model] if model else [])
        return argv + (["--yes"] if yes else ["--stage"]), False
    if op in ("MOVE", "COPY"):
        need(3, "a file, UNDER and a file")
        if len(rest) != 3:
            raise ScriptError("%s file UNDER file: %s" % (op, text))
        dest = place(1, ("UNDER", "INTO"))
        return [op.lower(), rest[0], "--to", dest], False
    if op == "MERGE":
        if len(rest) != 3:
            raise ScriptError("MERGE b INTO a (b's human and reading sides "
                              "go to a, and b is removed): " + text)
        a = place(1, ("INTO",))
        return ["merge", a, rest[0]], True
    if op == "DELETE":
        if len(rest) != 1:
            raise ScriptError("DELETE takes one file: " + text)
        return ["rm", rest[0]] + (["--yes"] if yes else []), True
    if op == "CHECK":
        if len(rest) > 1:
            raise ScriptError("CHECK [file]: " + text)
        return ["check"] + (["--branch", rest[0]] if rest else []), False
    if op == "GC":
        if rest:
            raise ScriptError("GC takes nothing: " + text)
        return ["gc"] + (["--yes"] if yes else []), True
    # PATH
    need(2, "a name and its statements")
    name = rest[0]
    if not re.fullmatch(r"[\w\-]+", name):
        raise ScriptError("PATH needs a name first (letters, digits, '_', "
                          "'-'): " + text)
    return ["path", name] + rest[1:], False


def translate_script(text, backend="dry", yes=False):
    """All statements of a script, each as (statement, argv, destructive);
    every statement is checked before anything runs."""
    out, problems = [], []
    for k, st in enumerate(split_script(text), 1):
        try:
            argv, bad = translate(st, backend, yes)
            out.append((" ".join(st), argv, bad))
        except (ScriptError, FlowError) as e:
            problems.append("statement %d: %s" % (k, e))
    return out, problems


__all__ = [
    'DESTRUCTIVE', 'OPS', 'PLACES', 'SEPARATORS', 'SIDE_WORDS',
    'ScriptError', 'split_script', 'translate', 'translate_script',
]
