# tawjih — توجيه أوامر الآلة

A personal, file-based way to direct a language model. Knowledge is kept in
trees of small files; each file has three sides; a *path* strings files
together into a workflow that the model and your own code execute step by step.

One Python file, standard library only. Linux and Windows, in the terminal.

[العربية](README.ar.md)

> **Status: 0.1, experimental.** The test suite passes on Linux. Windows and
> real AI providers have not been exercised by the author of this version yet;
> see [What is and is not verified](#what-is-and-is-not-verified).

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
python tawjih.py -p examples/quadratic tree
python tawjih.py -p examples/quadratic run solve --input "2x² − 4x = 6" --allow-code --backend manual
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
python tawjih.py path mypath 1.1 1.2
python tawjih.py run mypath --input "..." --backend ollama
```

Use `-p FOLDER` before the command to work on a project elsewhere.

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
| `path NAME [REF ...]` | define a path (with refs) or show it |
| `run NAME --input TEXT [--backend B] [--allow-code]` | execute a path |
| `edit REF --side S --ask TEXT [--source FILE] [--backend B]` | have a model rewrite one side; shows a diff and asks before saving |
| `deposit FILE` / `library` | add a text book to the library / list books |
| `filter BOOK [--parent REF] [--ask TEXT] [--backend B]` | have a model decide, passage by passage, what to keep and for which file; writes a report only |
| `accept BOOK` | append the kept passages to the reading side of their files |
| `import FILE --parent REF` | split a text on `#` headings into child files, without a model |

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
runs/     a log of every run                   (git-ignored)
history/  previous versions of edited files    (git-ignored)
```

## Limitations

- Paths are linear: no branching, conditions, loops or retries.
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
running paths (including that the human and reading sides never reach the
model), `edit`, the deposit → filter → accept pipeline, and `open` with a
scripted stand-in editor.

Not yet verified: Windows (the included GitHub Actions workflow runs the same
tests on Windows and on Python 3.9 at every push), real editors driven by a
person, and real providers. The three network backends were checked only
against a local mock server that imitates the response formats.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24182176.svg)](https://doi.org/10.5281/zenodo.24182176)
