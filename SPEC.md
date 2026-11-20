# Tawjih specification (0.7.0)

This is the contract the program and the tests follow.

## File

One file per node, `files/<id>.md`, UTF-8. The id in the header must equal
the file name.

```
---
id: e1d13b            stable, never changes
title: ...
level: 5              1 general, 2 section, 3 chapter, 4 topic, 5 branch, 6 part
parent: 15d0c1        empty for the root of a tree
order: 1              position among siblings
links: de5595:uses    optional, comma separated
model: ...            optional, the model this file's step uses
expects: json(a, b)   optional, contract on the step's input
produces: json(x)     optional, contract on the step's output
built_sha: ...        optional, written by the program (see Staleness)
tags: a, b            optional, cross-tree labels
---
=== human ===         the expressive human concept
=== instructions ===  machine side, part 1: text addressed to the model
=== code ===          machine side, part 2: Python
=== reading ===       passages filtered from deposited books
```

`model`, `expects` and `produces` are set with `contract REF [--expects S]
[--produces S] [--model M]`, which validates the contract, keeps the previous
version in `history/`, and warns about saved paths it now breaks.

A **tree** is a root file (empty parent) and everything below it. A child's
level must be deeper than its parent's. Files without the optional header
lines are read as before.

## Machine side and staleness

The machine side of a file is its `instructions` and `code` sections, with
leading and trailing blank lines removed. Its **digest** is the first 16 hex
characters of the SHA-256 of `instructions + "\n=== code ===\n" + code`.
A file with both parts empty has no machine side and cannot be a step.

Whenever the machine side is written through `open`, `edit` or the Linux
windows, `built_sha` records a digest of the `human` and `reading` sides.
If either changes later, the file is **stale**: `!` in `tree`, a warning in
`check` and `produce`, and a mark in `graph`. A file without `built_sha` is
never stale. `confirm REF` records the current digest without changing the
machine side (it was reviewed and still fits); so does `edit` on
`instructions` or `code` when the model returns the side unchanged.

## Contracts

A contract is `json`, `json(key, ...)` or `text`. Any other value is an
error reported by `check` and at run time, so a typo cannot switch the check
off.

- `text` accepts any output.
- `json(...)` accepts an answer that is a JSON object, or that contains one
  inside a code fence, or between its first `{` and last `}`. The keys
  listed must be present. The bare JSON, without the fence or the sentences
  around it, is what the step passes on.
- `expects` is checked on the step's input before the step runs; `produces`
  on its output. A model answer that breaks `produces` stops the run and is
  not kept in the cache.
- `check` compares the contracts of connected files: an output going to a
  step that expects `text` always fits; otherwise the kinds must match and,
  when both list keys, the keys expected must be among those produced.

## Model

The model of a step is, in order: `--model` on the command line, the file's
`model` header, the `TAWJIH_MODEL` environment variable.

## Path language

A path is saved as `paths/<name>.path`, one statement per line, files given
by id. On the command line files may also be given by number, and statements
are separated by `,` or `/`. Words are case-insensitive; Arabic equivalents
in brackets.

| Statement | Meaning |
|---|---|
| `A TO B` [إلى] | A's output is B's input |
| `A TO B AND C` [و] | A's output goes to both B and C |
| `B AND C TO D` | D receives the outputs of B and C together |
| `A TO B IF word` [إذا] | the connection is active only if `word` holds for A's output: when the output is a JSON object with `word` as a top-level key (case-insensitive), its value is true, non-zero and non-empty (the strings `false`, `no`, `0`, `none`, `null`, `لا` and `خطأ` count as false); otherwise the output contains `word` (case-insensitive). `IF NOT word` [ليس] for the opposite |
| `A AMONG B C D` [بين] | A decides: the candidate named first in A's output (by number, id or title) is the only one that runs; if none is named and exactly one candidate's `expects` accepts the output, that one runs |
| `A TO D THROUGH B C` [عبر] | A → B → C → D |

Statements chain left to right: `A TO B AND C TO D` connects A to B and C,
and both to D. `IF` applies to the connections of the `TO` just before it.
A file listed without any connecting word in a file that has none is the
old form: a plain sequence, one file per line.

A path must not contain a loop, and a file appears in it once. `check` and
`generate` note an `IF` word that is not one of the keys its source file
promises in a `json(...)` contract, since it will be looked for in the text.

## Production

1. Files run one at a time, each after every file that feeds it.
2. A file nothing feeds receives the original request. A file fed by one
   active connection receives that output. Fed by several, it receives them
   together, each under a `## from: <title> [id]` line.
3. A file none of whose connections is active is **skipped**, and so
   passes nothing on.
4. Only the machine side is extracted. The human and reading sides are
   never sent to the model or exposed to code.
5. If the file has code, it runs first with:
   - `input`, `origin` (the original request), `vars` (shared by all steps);
   - `steps`: outputs of the files **upstream** of this one (those whose
     output can reach it through the path), by id or number. Asking for any
     other file is an error, because its output might not exist yet. An
     upstream file that was skipped has no entry;
   - `meta`: this file's id, title, number and tree;
   - `linked(type="")`: the files this one links to, each with its output
     when that file is upstream, else `None`;

   and sets `output`. Code that runs longer than the limit (`--code-timeout`,
   60 seconds by default, `0` for none; checked between Python lines) fails
   with `CodeTimeout`. Code that sets no `output` and prints nothing passes
   its input on unchanged; code that sets `output` to an empty value has
   produced an empty output. If the file has instructions, the model
   receives the instructions, the code result if any, the original request
   and the input; its answer is the output.
6. Contracts are checked as described above.
7. A connection between files of different trees is a **handoff**.
8. The result is the output of the files that feed nothing and did run;
   several are joined, each under its title.
9. A failing step stops the production. A step that produces no output is
   a failing step. When a kept model answer may be the cause, the error
   names the step to ask again with `--resume --redo`.

## Answer cache

Each answer from a model is kept in `cache/`, in a file named by the
SHA-256 of the provider name, the step's model and the exact prompt. With
`--resume`, a step whose file exists takes its answer from it and the model
is not called; any change to the instructions, the input, the provider or
the model gives another name. Code always runs again. Without `--resume`
the model is always called. `--no-cache` keeps nothing, and the `dry` back
end is never kept. `--resume --redo REF` asks the named step again and
replaces its kept answer.

## The network

`graph` draws the network as Mermaid, Markdown (Mermaid inside a
`mermaid` code block, which GitHub shows as a picture) or Graphviz DOT.
Without `--format` the format follows the `-o` extension: `.mmd` and
`.mermaid` give Mermaid, `.dot` and `.gv` DOT, anything else Markdown.

- Without `--path`: the trees, parent to child, optionally one branch.
- With `--path NAME`: the path's files grouped by tree, each with its
  machine side and contracts. Connections carry their `IF` condition;
  `AMONG` choices are dashed; handoffs between trees are thick.
- Files with a machine side are outlined; stale files are marked.

`generate NAME` proposes a path and runs nothing:

- In the proposal each statement is followed by a comment that repeats it
  with titles in place of ids.
- `--start REF --end REF`: the shortest chain of files with a machine side
  in which each file's `produces` fits the next file's `expects`. Both
  contracts must be declared; no model is used.
- `--goal TEXT --backend B`: a model receives the path language, the files
  with a machine side (id, number, tree, title, contracts) and the goal, and
  replies with statements.
- `--scope REF` limits either to one branch.

The proposal is written to `paths/NAME.proposed` with any problem found
(unknown file, loop, a file without a machine side, a contract mismatch)
listed in it. `generate NAME --accept` checks it again and only then saves
it as `paths/NAME.path`, refusing to replace an existing path without
`--replace`.

## Search and tangle

- `search QUERY [--branch REF] [--side S] [--tag T]` searches titles, ids
  and sides across all trees or one branch, literally; `--rank` and
  `--semantic` are described under Managing a project.
- `tangle [--branch REF | --path NAME] [-o FILE]` exports the code sides as
  a Python module: one function `step_<id>(input, origin="", vars=None,
  steps=None, meta=None, linked=None)` per file, returning its output, with
  the human side as its docstring, and a `STEPS` list in tree or execution
  order. Instructions sides are not included, and the module does not run
  the path.

## Records

- `runs/<name>-<time>.md`: every prompt and output, step by step.
- `runs/<name>-<time>.result.txt`: the final output (successful runs only).
- `runs/<name>-<time>.state.json`: the outputs of the steps that ran, by
  id, written for failed runs too.
- `history/production-<name>-<time>.json`: the operation itself: trees,
  connections, handoffs, each step's id, number, tree, machine digest,
  status and the files it received from, the model, the backend, the
  input and its digest, and where the log and result are.
- `cache/<hash>.json`: one kept model answer, with its prompt.
- `runs/eval-<name>-<time>.json`: an evaluation, case by case.
- `state.json`: the digests used by `sync` and `status`.
- `index/embeddings.json`: the semantic search index.
- `history/<id>-<time>.md`: the previous version of a file, saved whenever
  a command or the Linux windows change it (see Versions).
- `paths/<name>.proposed`: a path proposed by `generate`, until accepted.

## Managing a project

**Restructuring** never changes an id, so paths stay valid; positional
numbers change. Each changed file is first saved to `history/`.

- `move REF (--to REF | --root [--level L]) [--position N | --before REF |
  --after REF]`: the file and its branch get the new parent; every level in
  the branch shifts by the same amount, which must keep them within 1-6; a
  file cannot move into its own branch. Siblings are renumbered 1, 2, ...
- `rename REF TITLE`: files that choose REF with `AMONG` in a saved path
  have each case-insensitive occurrence of the old title in their
  instructions replaced (`--no-update` leaves them); other machine sides
  that mention it are reported.
- `copy REF (--to REF | --root) [--branch] [--title T]`: new ids; sides,
  contracts, model, tags, `built_sha` and links are copied; links to files
  inside the copied set point to the copies.
- `rm REF [--branch] [--unlink] [--yes]`: refused while a saved path names
  the file (or a file of the branch), while it has children (without
  `--branch`), or while other files link to it (without `--unlink`, which
  removes those links). Without `--yes` nothing is removed.
- `split REF --passages SPEC --title T`: the reading side's passages are
  its blocks starting with `## `; those listed (`2,4-6`) move to a new file
  of the same level placed right after REF, which keeps the rest.
- `merge A B [--keep]`: B's human side (after a comment line naming B) and
  reading side are appended to A's, its links and tags added; without
  `--keep` B must have no children, no machine side and no path, links to it
  are moved to A, and it is removed.

**Versions**: `history/<id>-<YYYYMMDD-HHMMSS>[-k].md`, numbered newest first
from 1.

- `log REF` lists them with the sides and header fields that changed after
  each.
- `diff REF [--version K] [--side S]` prints changed header fields and a
  unified diff per changed side.
- `revert REF [--version K] [--side S] [--header]` saves the current file,
  then copies the chosen sides (default all four) and with `--header` the
  title, links, model, contracts and tags. When the instructions or code
  come back, `built_sha` comes back with them. A removed file (an id with
  versions but no file) is restored whole under its old parent.

**State**: `state.json` holds, per file, digests of the human side, the
machine side and the reading side as Tawjih last wrote them. A difference
is an edit made outside Tawjih. Projects older than 0.7 start without it.

- `sync` reports such edits, then records the current state;
  `--confirm-machine` sets `built_sha` for files whose machine side was
  edited; `--clean-work` removes `work/` leftovers; `--watch S` repeats.
- `status` lists: stale files; files in a saved path without a machine side;
  machine sides without contracts; empty human sides; edits outside Tawjih;
  folders left in `work/`.
- `tag REF WORD...`, `untag REF WORD...`: the header line `tags: a, b`.
  A tag is one word of letters, digits, `-`, `_` or `.`.
- `refs REF`: parent, children, links out and in, each saved path with the
  files it receives from and gives to, shared tags, and machine sides that
  mention the title.
- `clean [--runs DAYS] [--yes]`: `work/` folders, run records older than
  DAYS, and links to missing files are removed with `--yes`; files with a
  missing parent and unreadable paths are reported only.
- `audit [REF | --path NAME]` parses each code side and reports, with line
  numbers, imports of process, network, file-system and code-loading
  modules and calls that run programs, delete, move or write files, open
  files for writing, use the network or turn text into code. Exit status 1
  when there is a finding marked danger. `produce` and `run` print these
  findings before running code. It reads what is written; it is not a
  sandbox.

**Search**: `--rank` scores the title, each reading passage and each
paragraph of the other sides with Okapi BM25 (k1 1.5, b 0.75) over words
of two or more letters. `index --backend ollama|openai [--model M]` writes
`index/embeddings.json`: provider, model and, per piece of the human and
reading sides, its digest and vector; pieces whose digest is already there
are not embedded again. `--semantic` embeds the query with the same provider
and ranks pieces by cosine similarity.

**Outputs**

- `export` writes, per file in tree order, a heading at its depth with its
  number and title, its tags, its human side, with `--machine` its
  instructions and code, and unless `--no-reading` its reading passages as
  quotations. HTML escapes all text, marks every block `dir="auto"`, and
  loads MathJax 3 from jsDelivr when it finds `$...$` (`--no-mathjax`).
- `pack` writes a zip holding `manifest.json` (version, files, roots,
  paths), `files/<id>.md` for the branch or the paths' files, and
  `paths/<name>.path` for the named paths and every saved path whose files
  are all in the pack. `unpack` accepts only those entry names, refuses
  existing ids unless `--new-ids` (which renames them in files and paths),
  attaches the roots under `--parent` (levels shifted) or as new trees, and
  renames a clashing path `NAME-2`, `NAME-3`, ...
- `ingest` writes Markdown: `.docx` headings from the style's outline level
  or a style named "heading N" (or the Arabic "عنوان N"), list paragraphs as
  `- `, tables as `| a | b |`, Office Math as its text between `$`; `.epub`
  through the HTML reader in spine order; `.pdf` through `pdftotext`.
- `serve` answers `GET /health`; with `Authorization: Bearer TOKEN`,
  `GET /paths` (name, steps, whether it has code) and `POST /paths/NAME`
  with `{"input": "..."}`, which runs the path and returns `{"result",
  "messages"}`; 401 without the token, 403 for a path with code unless
  started with `--allow-code`, 404, 400 for a bad body, 413 above 1 MB, 500
  when the run fails. Runs are serialised.
- `eval NAME` reads `paths/NAME.cases.jsonl` (or `--cases`): per line
  `{"input", "expect", "match", "name"}`. `contains` (default for text) and
  `not-contains` compare case-insensitively with each expected string;
  `equals` compares the trimmed output; `regex` searches; `json` (default
  when `expect` is an object) requires every expected key with an equal
  value. The record is `runs/eval-NAME-<time>.json`; exit status 1 if a
  case fails.
- `dataset -o F --source eval|runs|steps` writes one JSON object per line,
  `{"messages": [user, assistant]}` or with `--format prompt` `{"prompt",
  "completion"}`; `--meta` adds where it came from. `eval`: passed cases
  (`--include-failed` for all); `runs`: finished productions, from the
  input kept in their record; `steps`: each cached model answer with its
  prompt. Duplicates and empty answers are skipped.

## Library

`deposit FILE [--part N|TITLE ...]` converts the source to text with `#`
headings, by extension: `.txt` as is; `.md` (setext headings, code fences
protected); `.tex` (in this order: `\input`/`\include` read in except
when commented out; code environments and `\verb` protected; comments
removed; the document's own commands and `\newtheorem` environments read
and expanded; the body taken after `\begin{document}` only when
`\documentclass` opens the file, up to the last `\end{document}`;
mathematics protected; then sectioning commands, labelled theorem-like
environments, `- ` items, `[footnote: …]`, `[ref: …]`, `[cite: …]`,
accents and special characters; a remaining command keeps its arguments,
separated by spaces, or its name; mathematics and code put back last); `.html` (`h1`-`h6`,
MathML replaced by its TeX annotation or `alttext`); `.xml`/`.ptx`
(PreTeXt: divisions, `xi:include`, `<m>` → `$…$`, `<me>`/`<md>` → `$$…$$`,
blocks numbered per chapter with one shared counter, `xref` replaced by the
target's label; Sage, index and images left out; other XML gives its text).
PDF, Word, EPUB, ODT and RTF are refused. A body line that would start with
`#` is indented by one space. `--part` keeps the top-level divisions named
by position or title. The index records the format.

`filter` splits a book on its `#` headings. Each passage is titled with the
headings above it, joined by ` > ` (`Chapter 3 > 3.3 Subgroups > Definitions
and Examples`), since two sections may share a title. A book without
headings is cut into passages of about `--size` characters.
