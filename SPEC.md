# Tawjih specification (0.3)

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
---
=== human ===         the expressive human concept
=== instructions ===  machine side, part 1: text addressed to the model
=== code ===          machine side, part 2: Python
=== reading ===       passages filtered from deposited books
```

A **tree** is a root file (empty parent) and everything below it. A child's
level must be deeper than its parent's.

## Machine side

The machine side of a file is its `instructions` and `code` sections, with
leading and trailing blank lines removed. Its **digest** is the first 16 hex
characters of the SHA-256 of `instructions + "\n=== code ===\n" + code`.
A file with both parts empty has no machine side and cannot be a step.

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
| `A TO B IF word` [إذا] | the connection is active only if A's output contains `word` (case-insensitive); `IF NOT word` [ليس] for the opposite |
| `A AMONG B C D` [بين] | A decides: the candidate named first in A's output (by number, id or title) is the only one that runs. Naming none stops the production |
| `A TO D THROUGH B C` [عبر] | A → B → C → D |

Statements chain left to right: `A TO B AND C TO D` connects A to B and C,
and both to D. `IF` applies to the connections of the `TO` just before it.
A file listed without any connecting word in a file that has none is the
old form: a plain sequence, one file per line.

A path must not contain a loop, and a file appears in it once.

## Production

1. Files run one at a time, each after every file that feeds it.
2. A file nothing feeds receives the original request. A file fed by one
   active connection receives that output. Fed by several, it receives them
   together, each under a `## from: <title> [id]` line.
3. A file none of whose connections is active is **skipped**, and so
   passes nothing on.
4. Only the machine side is extracted. The human and reading sides are
   never sent to the model or exposed to code.
5. If the file has code, it runs first with `input`, `origin` (the original
   request) and `vars` (shared by all steps), and sets `output`. If it has
   instructions, the model receives the instructions, the code result if
   any, the original request and the input; its answer is the output. With
   code only, the code result is the output.
6. A connection between files of different trees is a **handoff**.
7. The result is the output of the files that feed nothing and did run;
   several are joined, each under its title.
8. A failing step stops the production.

`produce` refuses to start if any step has no machine side, or has code
while `--allow-code` is absent.

## Records

- `runs/<name>-<time>.md`: every prompt and output, step by step.
- `runs/<name>-<time>.result.txt`: the final output (successful runs only).
- `history/production-<name>-<time>.json`: the operation itself: trees,
  connections, handoffs, each step's id, number, tree, machine digest,
  status (done, skipped, failed) and the files it received from, the
  backend, a digest of the input, and where the log and result are. Written
  for failed runs too.
- `history/<id>-<time>.md`: the previous version of a file, saved whenever
  `open` or `edit` changes it.
