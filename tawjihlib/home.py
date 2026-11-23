"""The home screen: the sprout logo, the project at a glance, the commands
as cards, and short explanations by topic (tawjih.py home [TOPIC]).

Drawn with Unicode blocks and ANSI colours when the terminal shows them,
and as plain text otherwise. English only: terminals do not lay out
right-to-left text reliably.

Part of Tawjih. Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
import os
import shutil
import sys
import textwrap
from pathlib import Path

from .core import Store, __version__
from .providers import load_policy, load_providers
from .commands import display_width

REPO = "https://github.com/AlmodinyMk/LLM-tawjih-Model-building"

# The sprout logo, one character per pixel ('.' is empty).
LOGO = (
    "..........Lg............",
    "..........LLg...........",
    ".........gLLg..gg.......",
    "..........gLg.gLLg......",
    "..........gLggLWLLg.....",
    "...........gLgLLLLg.....",
    "...........gsLLLLg......",
    "............s.gg........",
    "............s...........",
    "..ggg.......s...........",
    ".gLWWLgg....s...........",
    "..gLLLLLgg..s...........",
    "....ggLLLLgss...........",
    "........ggss............",
    "............s...........",
    "............s..gggg.....",
    "............sgLWWWLgg...",
    "............sgLLLLLLLLg.",
    "............s.ggLLLggg..",
    "............s....ggg....",
    "............s...........",
    "....bbB....BsB..b.......",
    "..bBBbBBbBBBBBbBBbbB....",
    ".bBbbBBBbbBBbBbbBBBbBb..",
    "bbBbBbbBbbBbBbbBBbBbbbBb",
    "........................",
)
LOGO_COLORS = {"L": (183, 224, 92), "g": (110, 170, 58),
               "W": (236, 248, 214), "s": (128, 182, 66),
               "B": (120, 84, 56), "b": (78, 54, 36)}

# Catppuccin Mocha (MIT licence), as in the Linux windows.
C = {"text": (205, 214, 244), "sub": (166, 173, 200), "dim": (108, 112, 134),
     "accent": (203, 166, 247), "green": (166, 227, 161),
     "peach": (250, 179, 135), "blue": (137, 180, 250),
     "yellow": (249, 226, 175), "badge": (69, 71, 90),
     "frame": (69, 71, 90)}

CARDS = [
    ("Recommended", True, [
        ("Write your first file", "start", "init · new · open",
         "Make a project, add a file with its title and its place in a "
         "tree, then open its human and machine sides in your editor."),
        ("Run a path", "run", "path · produce · run",
         "Join files with TO, AND, IF and AMONG, read the plan with "
         "--plan, run it on a provider; the record stays in runs/."),
    ]),
    ("Files and trees", False, [
        ("new", "files", "new TITLE --parent REF",
         "A three-sided file: human, instructions and code, reading."),
        ("open", "edit", "open REF",
         "Edit the sides in Notepad, nano, vim or VS Code."),
        ("tree", "view", "tree [REF]",
         "Code, number on tree lines, title and sides of every file."),
        ("search", "find", "search TEXT [--branch REF]",
         "Every side of every tree, or one branch."),
        ("move", "manage", "move REF --to REF",
         "Also copy, merge, split, rm, alias; ids never change."),
        ("do", "script", "do \"CREATE ... UNDER 1\"",
         "Management sentences in English or Arabic, as a plan first."),
    ]),
    ("Paths and runs", False, [
        ("path", "build", "path NAME 1 TO 2 IF ok ELSE 3",
         "Save a path; AND runs side by side, AMONG ... BY key chooses."),
        ("produce", "run", "produce NAME REFS --input TEXT",
         "Build, show and run a path across trees in one command."),
        ("check", "verify", "check",
         "Broken links, contracts that do not fit, stale machine sides."),
        ("graph", "see", "graph --path NAME -o FILE.md",
         "A Mermaid drawing of a path or of the trees."),
        ("eval", "measure", "eval NAME",
         "Run the examples of a path and score the answers."),
        ("trace", "explain", "trace --output FILE",
         "Which files an output came from; --cost for what runs cost."),
    ]),
    ("Contracts, providers and safety", False, [
        ("contract", "json", "contract REF --produces-schema F.json",
         "What a step expects and produces: json, keys or a schema."),
        ("provider", "models", "provider add NAME --kind ollama",
         "Named providers, retries, fallback chains, a test call."),
        ("policy", "limits", "policy set --allow-code false",
         "Limits the project sets for itself; it only restricts."),
        ("sandbox", "code", "sandbox REF --input FILE",
         "Run a file's code in a separate, limited process."),
        ("mcp", "assistants", "mcp [--allow-run]",
         "Let an AI assistant read, stage changes or run paths."),
        ("redact", "privacy", "redact --init",
         "redact.txt hides names and numbers before a model sees them."),
        ("whoami", "identity", "whoami set --name NAME",
         "Your id and name in records: who ran, approved, reviewed."),
    ]),
    ("Architectures and jobs", False, [
        ("arch", "design", "arch create TITLE --preset tiny",
         "A model architecture as a three-faced file: hypothesis, spec, "
         "papers."),
        ("arch build", "verify", "arch build REF --run",
         "Built with transformers; parameters checked against the count."),
        ("top", "watch", "top",
         "Runs, files and the computer live, like btop; q quits."),
        ("--trace", "Langfuse", "run NAME --input TEXT --trace",
         "Each run as a trace in Langfuse or any OpenTelemetry tool."),
        ("runtime", "jobs", "runtime add NAME --kind ssh --host H",
         "Run long jobs here, on another computer, or on a service."),
        ("show --type", "faces", "show REF --type machine",
         "One face of a file: human, machine or science."),
        ("--max-workers", "speed", "run NAME --max-workers 4",
         "Steps whose inputs are ready run at the same time."),
        ("ITERATE", "rounds", "path NAME 1 TO 2, ITERATE 3 UNTIL ok",
         "The whole path again on its result, until it is good."),
    ]),
]

TOPICS = {
    "start": ("Your first project", [
        ("Make the project", "tawjih.py init\n"
         "Creates files/, paths/ and runs/ in this folder. Everything is "
         "plain text you can read, copy and keep in git."),
        ("Add files", "tawjih.py new \"Extract data\"\n"
         "tawjih.py new \"Summarize\" --parent 1\n"
         "Each file has a stable code (its id, such as 10536c) and a number "
         "(1, 1.1 ...) that follows its place in the tree."),
        ("Write the sides", "tawjih.py open 1\n"
         "human: what the file is for, in your words.\n"
         "instructions: what the model is told.  code: Python, when a step "
         "is better computed than asked.  reading: notes and sources."),
        ("Run", "tawjih.py produce first 1 TO 1.1 --input \"...\" "
         "--backend manual\n"
         "manual shows each prompt and lets you answer it yourself; dry "
         "runs without a model; ollama, openai and anthropic call one."),
    ]),
    "files": ("Files and trees", [
        ("Three faces", "Every file is a folder files/CODE/ with three "
         "faces under one code, number and title: human.md, machine.md "
         "(instructions, code, examples) and science.md (the reading "
         "side). tree marks them H I C R; a dot is an empty side.\n"
         "tawjih show 1.1 --type machine"),
        ("Codes and numbers", "The code never changes; the number changes "
         "when files move. Paths and links store codes, so moving a file "
         "breaks nothing. alias gives a file a name of your own."),
        ("Stale", "When a file's human side changes after its machine side "
         "was written, tree shows ! and check warns."),
    ]),
    "paths": ("The path language", [
        ("Joining files", "1 TO 2         2 gets 1's output\n"
         "1 AND 2 TO 3   1 and 2 both feed 3\n"
         "1 TO 2 IF ok ELSE 3     on a word in the output\n"
         "1 TO 2 IF status = pass ELSE 3     on a JSON key\n"
         "1 AMONG 2 3 4 BY route  the key names the next file"),
        ("Steps", "EACH 2: once per item of a list.  APPROVE 2: ask a "
         "person first.  2 USING MODEL, 2 RETRY 3 UNTIL ok, 2 PICK 3."),
        ("Arabic", "The same words in Arabic: إلى و إذا وإلا بين عبر حسب ... "
         "Commas or / separate statements."),
    ]),
    "contracts": ("Contracts", [
        ("Between steps", "contract 2 --expects \"json(name, total)\"\n"
         "contract 3 --produces-schema verdict.json\n"
         "A step's answer is checked before it is passed on; RETRY asks "
         "again, and check compares connected files before any run."),
        ("Schemas", "A part of JSON Schema is supported: type, properties, "
         "required, enum, limits, pattern, items. Anything else is "
         "refused rather than half-checked."),
    ]),
    "providers": ("Providers", [
        ("Built in", "dry (no model), manual (you answer), ollama, openai "
         "(any compatible server), anthropic. Keys are read from "
         "environment variables only."),
        ("Named", "provider add local --kind ollama --model llama3.2\n"
         "provider fallback local cloud\n"
         "provider test local\n"
         "Busy servers are tried again with a growing wait; when one "
         "fails, the next in its chain answers."),
    ]),
    "safety": ("Safety", [
        ("Code", "Code sides run only with --allow-code, after you read "
         "them; --sandbox runs them in a separate process with limits."),
        ("Policy", "policy set --allow-code false --allow-network local\n"
         "The project's own limits. The file can only restrict, so an "
         "assistant that edits it cannot turn anything on."),
        ("Privacy", "redact.txt hides patterns before any model call and "
         "puts them back in the answer."),
    ]),
    "mcp": ("Assistants over MCP", [
        ("Start", "tawjih.py mcp                 read only\n"
         "tawjih.py mcp --allow-write   changes go to stage/ for review\n"
         "tawjih.py mcp --allow-run     runs paths (sends data, costs)"),
        ("Approval", "--human-approval always asks you before every "
         "change and run; approve ID --yes answers from the terminal."),
    ]),
    "arch": ("The architecture lab", [
        ("A design is a file", "tawjih arch create \"Apex 001\" --preset "
         "tiny\nThe human face holds the hypothesis and decisions, the "
         "science face the papers, the machine face the JSON "
         "specification: layers, hidden, heads, kv_heads, ffn, norm, "
         "positions, vocab, context, moe."),
        ("Lineage and variants", "tawjih arch derive 1 \"Apex 002\" --set "
         "kv_heads=1\ntawjih arch mutate 1 --count 6 --yes\nA derived "
         "design is a child of its source, so the tree is its history."),
        ("Check and build", "tawjih arch validate 1\ntawjih arch build 1 "
         "--run\nParameter counts are exact; build makes the model with "
         "transformers, runs it forward and backward, and checks the "
         "count. arch rank compares them."),
    ]),
    "top": ("Watching runs (top)", [
        ("The board", "tawjih top\nThe processor, memory, disk, network "
         "and GPU beside the runs of paths: the step running now, the time "
         "each step took, calls, tokens, retries; and each file of the trees "
         "with its runs, success and average time. Arabic: مراقبة."),
        ("Keys", "up and down choose a run, PgUp and PgDn move the trees, + "
         "and - change the refresh time, q quits."),
        ("Without a screen", "tawjih top --once\ntawjih top --json\nOne "
         "frame, or the same facts as JSON for a script."),
    ]),
    "langfuse": ("Traces in Langfuse", [
        ("Where", "LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_HOST "
         "in the environment (or TAWJIH_OTLP_ENDPOINT for another "
         "OpenTelemetry collector).\ntawjih observe check"),
        ("What", "tawjih run NAME --input TEXT --trace\nThe path, each step "
         "and each model call with its model, prompt, answer, tokens and "
         "times; texts after redaction, or none with --trace-no-content."),
        ("When it fails", "A trace that cannot be sent waits in "
         "history/traces-pending/ for: tawjih observe flush"),
    ]),
    "training": ("Training (plugin)", [
        ("From files to data", "python plugins/neural_bridge.py compile "
         "--out datasets\nTurns the examples of your files into training "
         "data."),
        ("Adapters", "train, merge, convert, quantize, and lora add, bind, "
         "route, fuse, serve. Each prints its plan; --run carries it out."),
        ("To the Hub", "pipeline NAME --base-model M --push both --repo "
         "you/NAME --run --yes\nsetup brings the libraries (here or on "
         "--runtime gpu); push uploads to a private repository with a model "
         "card; lora serve --serverless sets vLLM up for cold starts."),
        ("PEFT methods", "train NAME --peft dora --base-model M --run\n"
         "lora, dora, rslora, pissa, olora, loha, lokr, ia3, vera, oft, "
         "boft, fourierft, hra, randlora, ln, prompt, ptuning; --target "
         "chooses the layers and --peft-arg KEY=VALUE passes any other "
         "argument."),
    ]),
}


# ------------------------------------------------------------------ colours
class Paint:
    """ANSI styles: true colour, 256 colours, or none."""

    def __init__(self, mode):
        self.mode = mode

    @staticmethod
    def detect():
        if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
            return Paint(None)
        if os.name == "nt" and not (os.environ.get("WT_SESSION")
                                    or os.environ.get("TERM_PROGRAM")):
            return Paint(None)
        true = os.environ.get("COLORTERM", "") in ("truecolor", "24bit") \
            or os.environ.get("WT_SESSION") or os.environ.get(
                "TERM_PROGRAM") in ("vscode", "iTerm.app", "WezTerm")
        return Paint("true" if true else "256")

    def _rgb(self, rgb, ground):
        layer = 38 if ground == "fg" else 48
        if self.mode == "true":
            return "\033[%d;2;%d;%d;%dm" % ((layer,) + tuple(rgb))
        r, g, b = (round(v / 255 * 5) for v in rgb)
        return "\033[%d;5;%dm" % (layer, 16 + 36 * r + 6 * g + b)

    def __call__(self, text, fg=None, bg=None, bold=False, italic=False,
                 under=False):
        if not self.mode or not text:
            return text
        codes = ""
        if fg:
            codes += self._rgb(C.get(fg, fg) if isinstance(fg, str) else fg,
                               "fg")
        if bg:
            codes += self._rgb(C.get(bg, bg) if isinstance(bg, str) else bg,
                               "bg")
        codes += ("\033[1m" if bold else "") + ("\033[3m" if italic else "") \
            + ("\033[4m" if under else "")
        return codes + text + "\033[0m"


def logo_lines(paint):
    """The sprout in half blocks: two pixels per character cell."""
    out = []
    for y in range(0, len(LOGO), 2):
        top, low = LOGO[y], LOGO[y + 1]
        line = ""
        for a, b in zip(top, low):
            ca, cb = LOGO_COLORS.get(a), LOGO_COLORS.get(b)
            if not ca and not cb:
                line += " "
            elif not paint.mode:
                line += "█" if ca and cb else ("▀" if ca else "▄")
            elif ca and cb:
                line += paint("▀", fg=ca, bg=cb)
            else:
                line += paint("▀" if ca else "▄", fg=ca or cb)
        out.append(line)
    return out


def _plain(text):
    import re
    return re.sub(r"\033\[[0-9;]*m", "", text)


def _pad(text, width):
    """Pad text that may hold colour codes to a display width."""
    return text + " " * max(0, width - display_width(_plain(text)))


# ------------------------------------------------------------- the project
def project_facts(root):
    """(label, text, colour) lines about the project in this folder."""
    import contextlib
    import io
    try:
        with contextlib.redirect_stderr(io.StringIO()):  # no "error:" line
            st = Store(root)
    except SystemExit:
        return [("Project  ", "none here; start with: tawjih.py init",
                 "peach")]
    nodes = list(st.nodes.values())
    machine = sum(1 for n in nodes if n.instructions.strip()
                  or n.code.strip())
    stale = sum(1 for n in nodes if n.is_stale())
    paths = len(list(st.pdir.glob("*.path"))) if st.pdir.is_dir() else 0
    where = st.root.resolve()
    try:
        where = "~" + os.sep + str(where.relative_to(Path.home()))
    except (ValueError, RuntimeError):
        pass
    out = [("Project  ", str(where), "text"),
           ("         ", "%d tree(s) · %d file(s) · %d with a machine side "
            "· %d path(s)" % (len(st.children("")), len(nodes), machine,
                               paths), "sub")]
    if stale:
        out.append(("         ", "%d stale: tawjih.py check" % stale,
                    "peach"))
    provs = load_providers(st.root)
    name = provs.get("default") or ""
    if name:
        kind = (provs["providers"].get(name) or {}).get("kind", name)
        chain = provs["fallback"].get(name) or []
        out.append(("Provider ", "%s (%s)%s" % (
            name, kind, ("  then " + " → ".join(chain)) if chain else ""),
            "green"))
    else:
        out.append(("Provider ", "dry run; add one with: provider add",
                    "sub"))
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            pol = load_policy(st.root)
    except SystemExit:
        out.append(("Policy   ", "tawjih.policy.json is broken", "peach"))
        return out
    bits = []
    if not pol["allow_code"]:
        bits.append("no code")
    if pol["allow_network"] is False:
        bits.append("no network")
    elif pol["allow_network"] == "local":
        bits.append("this computer only")
    if pol["sandbox"]:
        bits.append("sandbox")
    if pol["human_approval"] != "never":
        bits.append("approval: " + pol["human_approval"])
    out.append(("Policy   ", " · ".join(bits) or "no limits of its own",
                "sub"))
    return out


def _wrap(paint, label, text, fg, avail):
    pieces = textwrap.wrap(text, max(12, avail - display_width(label)),
                           break_on_hyphens=False) or [""]
    return [paint(label if i == 0 else " " * display_width(label), fg="dim")
            + paint(p, fg=fg) for i, p in enumerate(pieces)]


# ------------------------------------------------------------------ layout
def _card(card, width, paint):
    title, tag, cmd, desc = card
    badge = " %s " % tag
    gap = width - display_width(title) - display_width(badge)
    head = paint(title, fg="text", bold=True) + " " * max(1, gap) + \
        paint(badge, fg="accent", bg="badge")
    lines = [head] + [paint(x, fg="accent", italic=True)
                      for x in textwrap.wrap(cmd, width,
                                             break_on_hyphens=False)]
    lines += [paint(x, fg="sub") for x in textwrap.wrap(desc, width)]
    return lines


def _grid(cards, columns, width, paint, framed):
    out = []
    gutter = 3
    inner = width - (4 if framed else 0)
    cw = (inner - gutter * (columns - 1)) // columns
    for i in range(0, len(cards), columns):
        row = [_card(c, cw, paint) for c in cards[i:i + columns]]
        h = max(len(r) for r in row)
        for r in row:
            r += [""] * (h - len(r))
        for k in range(h):
            line = (" " * gutter).join(_pad(r[k], cw) for r in row)
            if framed:
                line = paint("│ ", fg="frame") + _pad(line, inner) + \
                    paint(" │", fg="frame")
            out.append(line)
        if i + columns < len(cards):
            out.append(paint("│", fg="frame") + " " * (width - 2) +
                       paint("│", fg="frame") if framed else "")
    if framed:
        out = [paint("╭" + "─" * (width - 2) + "╮", fg="frame")] + out + \
              [paint("╰" + "─" * (width - 2) + "╯", fg="frame")]
    return out


def render_home(root, width=None, paint=None):
    paint = paint or Paint.detect()
    width = width or shutil.get_terminal_size((100, 30)).columns
    width = max(48, min(width, 132))
    wide = width >= 100
    avail = width - (31 if wide else 2)
    info = [paint("Tawjih ", fg="text", bold=True) +
            paint(" v%s " % __version__, fg="accent", bg="badge")]
    info += _wrap(paint, "", "Machine command directing: LLM work built "
                  "file by file", "green", avail)
    info += _wrap(paint, "", "Three-sided files, paths across trees, "
                  "contracts, and models on your computer or far away",
                  "sub", avail)
    info.append("")
    for label, text, fg in project_facts(root):
        info += _wrap(paint, label, text, fg, avail)
    info += ["", paint(REPO, fg="blue", under=True)]
    info += _wrap(paint, "Help     ", "tawjih.py COMMAND -h", "text", avail)
    info += _wrap(paint, "Explain  ", "tawjih.py home TOPIC", "text", avail)
    out = []
    logo = logo_lines(paint)
    if wide:
        for k in range(max(len(logo), len(info) + 1)):
            left = logo[k] if k < len(logo) else " " * 24
            right = info[k - 1] if 0 < k <= len(info) else ""
            out.append("  " + left + "   " + right)
    else:
        out += ["  " + x for x in logo] + [""] + ["  " + x for x in info]
    out += ["", "  " + paint("Choose where to start below.", fg="sub",
                             italic=True), ""]
    columns = 3 if width >= 108 else 2 if width >= 72 else 1
    for name, framed, cards in CARDS:
        out.append("  " + paint(name, fg="peach", under=True))
        out.append("")
        cols = min(columns, 2) if framed else columns
        out += ["  " + x for x in _grid(cards, cols, width - 4, paint,
                                        framed)]
        out.append("")
    keys = [("home TOPIC", " explanations"), ("COMMAND -h", " options"),
            ("tree", " your files")]
    line = ""
    for k, what in keys:
        piece = paint(k, fg="text", bold=True) + paint(what, fg="sub")
        if line and display_width(_plain(line + piece)) + 3 > width - 2:
            out.append("  " + line)
            line = ""
        line += ("   " if line else "") + piece
    out.append("  " + line)
    out += ["  " + x for x in _wrap(paint, "topics: ", " ".join(TOPICS),
                                    "dim", width - 2)]
    return "\n".join(out)


def render_topic(topic, width=None, paint=None):
    paint = paint or Paint.detect()
    width = width or shutil.get_terminal_size((100, 30)).columns
    width = max(48, min(width, 100))
    title, parts = TOPICS[topic]
    out = ["  " + paint("Tawjih", fg="text", bold=True) + paint(
        "  ·  ", fg="dim") + paint(title, fg="peach", under=True), ""]
    for head, body in parts:
        out.append("  " + paint(head, fg="text", bold=True))
        for para in body.split("\n"):
            if para.startswith(("tawjih.py", "python ", "provider ",
                                "contract ", "policy ", "1 ")):
                out.append("    " + paint(para, fg="accent"))
            else:
                out += ["    " + paint(x, fg="sub")
                        for x in textwrap.wrap(para, width - 6)]
        out.append("")
    others = [t for t in TOPICS if t != topic]
    out.append("  " + paint("more: tawjih.py home " + " | ".join(others),
                            fg="dim"))
    return "\n".join(out)


def cmd_home(a):
    """The home screen, or an explanation of one topic."""
    topic = getattr(a, "topic", None)
    if topic:
        if topic not in TOPICS:
            from .core import die
            die("no topic '%s'; topics: %s" % (topic, ", ".join(TOPICS)))
        print(render_topic(topic))
    else:
        print(render_home(getattr(a, "project", ".")))
    return 0


__all__ = ['CARDS', 'LOGO', 'LOGO_COLORS', 'Paint', 'REPO', 'TOPICS',
           'cmd_home', 'logo_lines', 'project_facts', 'render_home',
           'render_topic']
