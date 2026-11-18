# tawjih — توجيه أوامر الآلة

A personal, file-based way to direct a language model. Knowledge is kept in
trees of small files; each file has three sides; a *path* strings files
together into a workflow that the model and your own code execute step by step.

One Python file, standard library only. Linux and Windows, in the terminal.

[العربية](README.ar.md)

> **Status: 0.6.1, experimental.** The test suite passes on Linux and Windows.
> No real AI provider has been used with it yet, and the Linux desktop
> windows have not been used by a person; see
> [What is and is not verified](#what-is-and-is-not-verified).

![How Tawjih is built: interfaces, build time, the three-sided file, run time, the network, and the project on disk](docs/images/architecture-en.png)

## Screenshots

The example path run in the terminal, then run again with `--resume`:

![The terminal program running the example path across two trees](docs/images/running-terminal-en.png)

The four Linux windows and a side conversation:

![The four Linux windows and a side conversation](docs/images/running-windows-en.png)

In these screenshots the model's answers came over HTTP from a stand-in that
imitates Ollama; everything else is the program's real output.

## The idea

Every file has three sides:

| Side | Holds | Used when |
|---|---|---|
| **human** | the expressive human concept of the subject | building the file |
| **machine** | instructions for the model, and optional Python code | building, and **running a path** |
| **reading** | passages filtered from deposited books and research, with their source | building the file |

Files form trees with up to six levels: general → section → chapter → topic →
branch → part. Every file has a stable id (`e1d13b`) and a positional number
(`1.1.1.1.1`); commands accept either.

Two phases are kept apart on purpose:

- **Build time** (`open`, `edit`): you, optionally helped by an AI provider,
  turn the reading side into the human concept and the human concept into
  machine language.
- **Run time** (`run`): only the machine side is sent. The model sees the
  step's instructions, the original request and the previous step's output,
  and nothing else. Instructions therefore have to be self-contained.

## Quick start

```
python tawjih.py -p examples/quadratic trees
python tawjih.py -p examples/quadratic produce branch --plan
python tawjih.py -p examples/quadratic produce branch --input "2x² − 4x = 6" --allow-code --backend manual
```

`--backend manual` prints each prompt and waits for you to paste an answer
(finish with a line containing only `.`), so the example needs no account.
For step 1 answer `{"a": 2, "b": -4, "c": -6}`.

Your own project:

```
python tawjih.py init
python tawjih.py new "Mathematics"
python tawjih.py new "Algebra" --parent 1 --open
python tawjih.py tree
python tawjih.py produce myprod 1.1 TO 2.3 AND 1.2 TO 1.4 --input "..." --backend ollama
```

Use `-p FOLDER` before the command to work on a project elsewhere.

## Cross-tree production

1. `trees` lists the available trees.
2. Pick files from any trees, by id or number.
3. Connect them with words after `produce NAME`. The path is saved under
   that name and runs again with `produce NAME`.
4. The machine side of each file is extracted (see [SPEC.md](SPEC.md)).
5. Each file runs after the files that feed it and receives their outputs.
6. The result and full log go to `runs/`; a record of the operation (trees,
   connections, the digest of each machine side used, which steps ran or
   were skipped) goes to `history/`.

| Write | Meaning |
|---|---|
| `A TO B` | A's output goes to B |
| `A TO B AND C` | ... to both B and C (branching) |
| `B AND C TO D` | D receives both outputs (joining) |
| `A TO B IF word` | when A's output is JSON with that key: only when its value is true or non-empty; otherwise only when the output contains the word. `IF NOT word` for the opposite |
| `A AMONG B C D` | A's output names one of them; only that one runs |
| `A TO D THROUGH B C` | A → B → C → D |

Statements chain (`A TO B AND C TO D`), and several are separated by `,` or
`/` (use `/` in PowerShell). Arabic words work too: إلى، و، إذا، ليس، بين، عبر.
Files written with no connecting word are a plain sequence.

```
produce review 1.1 TO 1.2 IF error / 1.1 TO 1.3 IF NOT error / 1.2 AND 1.3 TO 2.1
```

`produce --plan` shows each step, its size in characters and what feeds it,
without running.

A step that returns nothing stops the production. After a failure, run the
same command with `--resume`: steps whose prompt, provider and model are
unchanged take their answer from `cache/` and the model is not called again
for them. When a kept answer is itself the cause (a model replied in prose
where the next step expected JSON), the error names the step to ask again:
`--resume --redo REF`. `--no-cache` keeps no answers, `cache` shows what is
kept and `cache --clear` empties it. `--model NAME` chooses the model for
one command.
`produce` will not start if a step has an empty machine side, or has code
and `--allow-code` is missing.

## Contracts, models and staleness

A file's header may carry three optional lines, set with
`contract REF --expects SPEC --produces SPEC --model NAME` (`none` clears
one; without options it shows them):

```
model: llama3.1:8b        the model this step uses
expects: json(a, b, c)    what its input must be
produces: json(roots)     what its output must be
```

A contract is `json`, `json(key, ...)` or `text`. A JSON answer is accepted
when it is wrapped in a code fence or has a sentence around it, as models
often write; the bare JSON is what goes to the next step. An answer that
breaks the contract stops the run, is not kept in the cache, and the error
names the step to ask again. `check` compares the contracts of connected
files before anything runs, and reports an unknown contract (a typo) as an
error.

The model of a step is `--model` if given, else the file's `model`, else
`TAWJIH_MODEL`.

When you rebuild a machine side (with `open`, `edit` or the Linux windows)
the program records the state of the human and reading sides. If either
changes afterwards, the file is marked **stale**: `!` in `tree`, a warning in
`check` and `produce`, and in `graph`. If you review a stale file and its
machine side still fits, `confirm REF` records that; an `edit` asked to
rebuild a side that comes back unchanged records it too.

`IF` and JSON: when a step's output is a JSON object, `IF is_group` tests the
value of the key `is_group` (true, non-zero, non-empty; the strings
`"false"`, `"no"` and `"لا"` count as false), not whether the
letters appear somewhere in the text. `check` and `generate` add a note when
an `IF` word is not one of the keys the source file promises in `produces`.

## The network: drawing and generating paths

`graph` draws your trees or a path:

```
python tawjih.py graph -o trees.md                       # the trees
python tawjih.py graph --path branch -o branch.md        # one path
python tawjih.py graph --path branch --format dot -o branch.dot
```

The `.md` file holds a Mermaid diagram that GitHub shows as a picture;
`--format mermaid` gives the diagram alone and `--format dot` gives Graphviz.
In a path, files are grouped by tree, `IF` conditions label the arrows,
`AMONG` choices are dashed, handoffs between trees are thick, and stale
files are marked.

`generate` proposes a path, and runs nothing:

```
python tawjih.py generate solve2 --start 1.1.1.1.1 --end 1.1.1.1.3
python tawjih.py generate review --goal "check and explain a solution" --backend ollama
python tawjih.py generate review --accept
```

With `--start` and `--end` it finds the shortest chain of files whose
`produces` fits the next one's `expects`, without a model. With `--goal` a
model reads the path language, your files and their contracts, and replies
with statements. Either way the proposal goes to `paths/NAME.proposed` with
any problem listed in it (unknown file, loop, a file without a machine side,
a contract mismatch). Read it, edit it if you like, and `--accept` saves it
as a path after checking it again. `--scope REF` limits the files to one
branch.

## Books and their sources

`deposit` reads a book from its source, which keeps its structure and its
mathematics. PDF is refused: it records where letters sit on a page, so
exponents, arrows and fractions come out broken (`271³²¹` becomes `271321`).

| Source | What deposit does |
|---|---|
| `.txt` | kept as it is; `#` lines are headings |
| `.md` | `#` and underlined headings; `#` lines inside code blocks are protected |
| `.tex` | `\part` … `\subsubsection` become headings, `\input` and `\include` are read in, theorem-like environments are labelled, lists become `- ` lines, mathematics stays LaTeX; pictures and comments are left out |
| `.html` | `<h1>`–`<h6>` become headings; scripts, styles and navigation are left out; MathML is replaced by its TeX |
| `.xml`, `.ptx` | PreTeXt: chapters and sections become headings, `xi:include` files are read in, `<m>` and `<md>` become `$…$` and `$$…$$`, theorems, examples and figures get the book's numbers (Example 3.8) and references point to them; Sage cells, index entries and pictures are left out. Other XML: its text |

`deposit BOOK --part 3 --part 4` keeps only those chapters (by number or
title). Deposit the whole book this way rather than a single chapter file,
so the numbering is the book's own. Many open textbooks (PreTeXt, LaTeX on
arXiv) publish their source; look for it before using a PDF.

## Commands

| Command | Purpose |
|---|---|
| `init` | create a project in the current folder |
| `new TITLE [--parent REF] [--level L] [--open]` | create a file |
| `open REF [--editor CMD] [--reading]` | open one editor per side; save on close |
| `tree [--branch REF]` | show the trees; `H I C R` mark the sides that have content, `!` a stale machine side |
| `search QUERY [--branch REF] [--side S]` | search titles and sides |
| `show REF` / `read REF [--all]` | print a file / print its reading side |
| `link SRC DST [--type T]` | record a link between any two files |
| `contract REF [--expects S] [--produces S] [--model M]` | show or set a file's contracts and model |
| `check [--branch REF]` | validate parents, levels, cycles, links, paths, code syntax, contracts, staleness |
| `trees` | list the available trees |
| `cache [--clear]` | show or empty the kept model answers |
| `produce NAME [WORDS ...] [--plan] --input TEXT [--backend B] [--allow-code] [--code-timeout S]` | cross-tree production: select, order, validate, run, save, document |
| `path NAME [WORDS ...]` | define a path without running it, or show a saved one |
| `run NAME --input TEXT [--backend B] [--allow-code]` | run a saved path |
| `edit REF --side S --ask TEXT [--source FILE] [--backend B]` | have a model rewrite one side; shows a diff and asks before saving |
| `deposit FILE [--part N\|TITLE]` / `library` | add a book (text, Markdown, LaTeX, HTML, XML/PreTeXt) to the library / list books |
| `confirm REF ...` | record that a machine side still fits its human and reading sides (clears the stale mark) |
| `filter BOOK [--parent REF] [--ask TEXT] [--backend B]` | have a model decide, passage by passage, what to keep and for which file; writes a report only |
| `accept BOOK` | append the kept passages to the reading side of their files |
| `import FILE --parent REF` | split a text on `#` headings into child files, without a model |
| `graph [--path NAME] [--branch REF] [--format md\|mermaid\|dot] [-o FILE]` | draw the trees or a path; the format follows the `-o` extension (`.mmd`, `.dot`, `.gv`) unless given |
| `generate NAME (--start REF --end REF \| --goal TEXT --backend B) [--scope REF]` | propose a path; `generate NAME --accept [--replace]` saves it |
| `tangle [--branch REF \| --path NAME] [-o FILE]` | export the code sides as Python functions, one per step |

## Desktop windows, for Linux users

**For Linux users only.** `linux/tawjih_desk.py` opens four independent
windows over a project: books and research, the human concept, machine
language, and search with a task schedule. Each window has a side
conversation with the AI that cannot change the project. It runs on GTK 3 or
Tkinter, in English, with optional AI help. Everything else in this
repository works on Linux and Windows. See
[linux/README.md](linux/README.md).

## Editors

`open` writes each side to a scratch file and starts an editor:

- Windows: Notepad, one window per side.
- Linux: nano, else vim, inside the terminal. With vim the sides appear side
  by side.
- VS Code: `--editor vscode`. The machine side opens as two files,
  `instructions.md` and `code.py`, next to `human.md`, so Python and
  Markdown support apply; close the window to save. `--split` does the same
  with any editor.
- Anything else: `--editor "COMMAND"` or the `TAWJIH_EDITOR` variable.

If an edit is interrupted, its text stays in `work/` and the next `open` of
that file refuses to overwrite it until you pass `--discard`.

The machine side is one document with two parts, instructions above the
`=== code ===` line and Python below it. The reading editor opens only when
the file already has passages, or with `--reading`.

## AI providers

| `--backend` | Talks to | Environment |
|---|---|---|
| `dry` (default) | nothing; shows what would be sent | — |
| `manual` | you paste the answer | — |
| `ollama` | a local Ollama server | `OLLAMA_HOST` (optional) |
| `openai` | any OpenAI-compatible `/chat/completions` endpoint | `TAWJIH_BASE_URL`, `TAWJIH_API_KEY` |
| `anthropic` | Anthropic Messages API | `ANTHROPIC_API_KEY` |

Name the model with `--model NAME` or the `TAWJIH_MODEL` variable, for all
three network backends. Keys are
read from the environment and never written to disk.

## Code steps

Code on the machine side runs as one step of a path. It sees `input` (the
previous step's output), `origin` (the original request), `vars` (a dict
shared by all steps), `steps` (the outputs of the files upstream of it in the
path, by id or number), `meta` (this file's id, title, number and tree) and
`linked(type="")` (the files it links to, with their output when they are
upstream), and sets `output`. Reading `steps` for a file that is not
upstream is an error: its output might not exist yet.

**Code is executed with `exec`, without any sandbox, with your user's full
permissions.** It runs only when you pass `--allow-code`. Do not run paths
whose code you have not read, including code a model wrote through `edit`.
A code step that runs longer than 60 seconds is stopped
(`--code-timeout SECONDS`, `0` for no limit). The limit is checked between
Python lines, so it stops a loop that never ends but not one long call into
C code, and it makes pure-Python loops several times slower while it is on.
It is a safeguard against mistakes, not a sandbox.

## Project layout

```
files/    one .md file per tree file (header + four marked sections)
paths/    one .path file per path, and .proposed files from generate
library/  deposited books and filter reports   (git-ignored)
runs/     log, result and state of every run   (git-ignored)
cache/    kept model answers, for --resume     (git-ignored)
history/  production records, and previous
          versions of edited files             (git-ignored)
```

## Limitations

- Branches run one after another, not in parallel. There are no loops or
  automatic retries. A failed production is run again with `--resume`.
- A kept answer is reused as it is. Models answer differently each time, so
  `--resume` freezes the first answer until the prompt changes.
- `IF` only tests a key of a JSON output or whether a word appears in the
  text, and `AMONG` only
  finds a candidate's number, id or title in it. Both depend on the model
  wording its answer as the instructions ask.
- A JSON contract tolerates a code fence and sentences around the JSON, but
  a step still fails when the model gives no JSON at all.
- Links matter only to code that calls `linked()`.
- `generate --start/--end` finds only straight chains, and only through
  files that declare both contracts. A model's proposal is only as good as
  the model; it is checked, not trusted.
- `tangle` exports code only: instructions need a model, and the exported
  module does not run the path.
- Books are read from text, Markdown, LaTeX, HTML or XML (PreTeXt). PDF
  and Word are refused; the LaTeX and HTML readers handle common markup, not
  every package or page layout.
- Filtering makes one model call per passage, and copies kept passages
  verbatim; it does not summarise.
- Staleness is recorded only when a machine side is written through
  `open`, `edit` or the Linux windows, not when you edit a file by hand.
- A line consisting only of `=== human ===` (or another section marker)
  inside your text will be read as a section boundary.
- External providers receive whatever you send them: with `edit`, the file's
  sides and its ancestors' human sides; with `filter`, book passages and the
  tree's titles.

## What is and is not verified

Verified by the test suite (`python -m unittest discover -s tests -v`), on
Linux with Python 3.12: file format, trees and numbering, structural checks,
running paths and cross-tree productions with branching, joining, `IF`,
`AMONG` and `THROUGH` (including that the human and reading sides never reach the
model), `edit`, the deposit → filter → accept pipeline, `open` with a
scripted stand-in editor, contracts, model choice, staleness, `steps` and
`linked`, `tangle`, `graph` and `generate`. The diagrams from `graph` were
rendered with the Mermaid command-line tool and with Graphviz.

The same tests pass on Windows and Linux, on Python 3.11 and 3.13, in the
GitHub Actions workflow that runs at every push. The Linux desktop windows
were driven by a script on a virtual display; see
[linux/README.md](linux/README.md).

Not yet verified: real editors and the desktop windows used by a person, and
real providers. The three network backends were checked only
against a local mock server that imitates the response formats.

## License

Copyright (c) 2026 Mohammed Khalid Almodiny

Licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International](LICENSE) (CC BY-NC-SA 4.0): you may use, share and adapt this
work for noncommercial purposes, provided you give credit and release your
adaptations under the same license. Commercial use needs separate permission
from the copyright holder.
