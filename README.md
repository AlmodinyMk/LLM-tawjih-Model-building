# tawjih — توجيه أوامر الآلة

A personal, file-based way to direct a language model. Knowledge is kept in
trees of small files; each file has three sides; a *path* strings files
together into a workflow that the model and your own code execute step by step.

One Python file, standard library only. Linux and Windows, in the terminal.

[العربية](README.ar.md)

> **Status: 0.4, experimental.** The test suite passes on Linux and Windows.
> No real AI provider has been used with it yet, and the Linux desktop
> windows have not been used by a person; see
> [What is and is not verified](#what-is-and-is-not-verified).

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
| `A TO B IF word` | only when A's output contains the word; `IF NOT word` for the opposite |
| `A AMONG B C D` | A's output names one of them; only that one runs |
| `A TO D THROUGH B C` | A → B → C → D |

Statements chain (`A TO B AND C TO D`), and several are separated by `,` or
`/` (use `/` in PowerShell). Arabic words work too: إلى، و، إذا، ليس، بين، عبر.
Files written with no connecting word are a plain sequence.

```
produce review 1.1 TO 1.2 IF error / 1.1 TO 1.3 IF NOT error / 1.2 AND 1.3 TO 2.1
```

`produce --plan` shows each step and what feeds it, without running.
`produce` will not start if a step has an empty machine side, or has code
and `--allow-code` is missing.

## Commands

| Command | Purpose |
|---|---|
| `init` | create a project in the current folder |
| `new TITLE [--parent REF] [--level L] [--open]` | create a file |
| `open REF [--editor CMD] [--reading]` | open one editor per side; save on close |
| `tree` | show all trees; `H I C R` mark the sides that have content |
| `show REF` / `read REF [--all]` | print a file / print its reading side |
| `link SRC DST [--type T]` | record a link between any two files |
| `check` | validate parents, levels, cycles, links, paths, code syntax |
| `trees` | list the available trees |
| `produce NAME [WORDS ...] [--plan] --input TEXT [--backend B] [--allow-code]` | cross-tree production: select, order, validate, run, save, document |
| `path NAME [WORDS ...]` | define a path without running it, or show a saved one |
| `run NAME --input TEXT [--backend B] [--allow-code]` | run a saved path |
| `edit REF --side S --ask TEXT [--source FILE] [--backend B]` | have a model rewrite one side; shows a diff and asks before saving |
| `deposit FILE` / `library` | add a text book to the library / list books |
| `filter BOOK [--parent REF] [--ask TEXT] [--backend B]` | have a model decide, passage by passage, what to keep and for which file; writes a report only |
| `accept BOOK` | append the kept passages to the reading side of their files |
| `import FILE --parent REF` | split a text on `#` headings into child files, without a model |

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
- Anything else: `--editor "code --wait"` or the `TAWJIH_EDITOR` variable.

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

Set the model name in `TAWJIH_MODEL` for all three network backends. Keys are
read from the environment and never written to disk.

## Code steps

Code on the machine side runs as one step of a path. It sees `input` (the
previous step's output), `origin` (the original request) and `vars` (a dict
shared by all steps), and sets `output`.

**Code is executed with `exec`, without any sandbox, with your user's full
permissions.** It runs only when you pass `--allow-code`. Do not run paths
whose code you have not read, including code a model wrote through `edit`.

## Project layout

```
files/    one .md file per tree file (header + four marked sections)
paths/    one .path file per path (ordered ids)
library/  deposited books and filter reports   (git-ignored)
runs/     log and result of every production   (git-ignored)
history/  production records, and previous
          versions of edited files             (git-ignored)
```

## Limitations

- Branches run one after another, not in parallel. There are no loops or
  retries, and a failed production restarts from step 1.
- `IF` only tests whether a word appears in the output, and `AMONG` only
  finds a candidate's number, id or title in it. Both depend on the model
  wording its answer as the instructions ask.
- A step that expects structured output from the model fails if the model
  does not follow the format.
- Links are stored and checked but do not affect anything yet.
- Books must be plain UTF-8 text; PDF and Word need converting first.
- Filtering makes one model call per passage, and copies kept passages
  verbatim; it does not summarise.
- Nothing warns you that a file's reading or human side changed after its
  machine side was last built.
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
model), `edit`, the deposit → filter → accept pipeline, and `open` with a
scripted stand-in editor.

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
