# Tawjih specification (0.15.0)

This is the contract the program and the tests follow.

## File

Each node is a folder `files/<id>/` (the id is 6 hexadecimal digits) with
four UTF-8 files that share its code, number and title:

| File | Holds |
|---|---|
| `head.json` | the header below, as a JSON object; empty fields left out; `links` is a list of `[id, type]`, `tags` and `aliases` lists |
| `human.md` | the human face: the `human` section |
| `machine.md` | the machine face: `=== instructions ===`, `=== code ===` and, when there are any, `=== examples ===`; a machine.md without these marker lines is all instructions |
| `science.md` | the science face: the `reading` section |

A face file may start with a line `<!-- tawjih ID FACE: TITLE -->`, which
the program writes and ignores when reading. A missing face file is an
empty face. The id in `head.json` must equal the folder name. The type
words for the faces are `human`, `machine`, `science` (Arabic `بشري`,
`آلي`, `علمي`; `reading` is accepted for `science`).

A node may instead be one file `files/<id>.md` (the form before 0.15),
which holds the same header and sections in one text, shown below. It is
read as it is and becomes a folder the next time the program saves it, or
with `migrate --yes`; a node present both ways is an error. Releases,
packs, backups and the archive keep each node as this one text.

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
retries: 2            optional, see Failures
fallback: 3c2a91      optional, see Failures
approve: after        optional, see Failures
split: train          optional, see Examples
weight: 1             optional, see Examples
examples_sha: ...     optional, written by the program
locked: yes           optional, see Governance
aliases: quad, q      optional, other names usable as REF
each: yes             optional, FOR EACH: yes or the key of a list
reviewed_sha: ...     optional, written by review --ok
reviewed_on: date     optional, written by review --ok
reviewed_by: N (id)   optional, written by review --ok (see Identity)
---
=== human ===         the expressive human concept
=== instructions ===  machine side, part 1: text addressed to the model
=== code ===          machine side, part 2: Python
=== reading ===       passages filtered from deposited books
=== examples ===      optional: worked questions (see Examples)
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

A contract is `json`, `json(key, ...)`, `schema(NAME)` or `text`. Any other value is an
error reported by `check` and at run time, so a typo cannot switch the check
off.

- `text` accepts any output.
- `json(...)` accepts an answer that is a JSON object, or that contains one
  inside a code fence, or between its first `{` and last `}`. The keys
  listed must be present. The bare JSON, without the fence or the sentences
  around it, is what the step passes on.
- `schema(NAME)` accepts what `json` accepts, when the JSON value also
  conforms to the JSON Schema in `schemas/NAME.json` (NAME: letters, digits,
  `_` and `-`). The supported keywords are `$schema`, `$id`, `$comment`,
  `title`, `description`, `default`, `examples`, `type` (one or a list of
  object, array, string, number, integer, boolean, null), `properties`,
  `required`, `additionalProperties` (false or a schema), `items` (a schema),
  `enum`, `const`, `minimum`, `maximum`, `exclusiveMinimum`,
  `exclusiveMaximum`, `minLength`, `maxLength`, `pattern` (Python `re`,
  searched), `minItems`, `maxItems`, `uniqueItems`, `minProperties` and
  `maxProperties`. A schema with any other keyword is an error, as is a
  missing or unreadable schema file: `check` reports both. At most ten
  violations are reported, each with its JSON location. For `check`, a
  produced schema promises the keys of its `properties`, and an expected one
  needs the keys in its `required`.
- `expects` is checked on the step's input before the step runs; `produces`
  on its output. A model answer that breaks `produces` stops the run and is
  not kept in the cache.
- `check` compares the contracts of connected files: an output going to a
  step that expects `text` always fits; otherwise the kinds must match and,
  when both list keys, the keys expected must be among those produced.

## Model

The model of a step is, in order: `--model` on the command line, the path's
`USING`, the file's `model` header, the provider's `model` in
`providers.json`, the `TAWJIH_MODEL` environment variable.

## Providers

`--backend NAME` names a built-in kind (`dry`, `manual`, `ollama`, `openai`,
`anthropic`) or a provider in `providers.json`; without it, the file's
`default`, else `dry`. An unknown name is an error.

```
{"default": "local",
 "providers": {"local": {"kind": "ollama", "base_url": "http://localhost:11434",
                         "model": "llama3.2"},
               "cloud": {"kind": "openai", "base_url": "https://...",
                         "api_key_env": "EXAMPLE_KEY", "model": "m",
                         "timeout": 600, "retries": 2,
                         "retry_status": [429, 500, 502, 503, 504],
                         "backoff": "exponential", "backoff_seconds": 2}},
 "fallback": {"local": ["cloud"]}}
```

A provider's settings come before the environment variables of its kind
(`OLLAMA_HOST`, `TAWJIH_BASE_URL`, `ANTHROPIC_BASE_URL`); its key is read
from the variable named by `api_key_env`, else the kind's usual variable.
No key is ever written to a file. A built-in kind without an entry uses the
defaults shown. A call that fails with no connection, a timeout or a status
in `retry_status` is made again up to `retries` times; the wait is
`backoff_seconds`, times 2 to the number of tries before with `exponential`,
or the server's `Retry-After` in seconds; never more than 60 seconds. Any
other failure, or the last try, moves to the next provider in
`fallback[NAME]` (each tried once in the chain, with its own retries). The
first provider gets the step's model, or its own when the step names none; a
fallback gets its own model, else the step's. When every provider fails, the
step fails as for any model call (and its `fallback:` file runs). Usage
records `provider` and `attempts`.

## Policy

`tawjih.policy.json` restricts the project; nothing in it can allow what the
command line does not.

| Key | Values (default first) | Effect |
|---|---|---|
| `allow_code` | `true`, `false` | `false`: `--allow-code` and `sandbox` are refused, and a run asked to run code fails |
| `allow_network` | `true`, `"local"`, `false` | `false`: only `dry` and `manual` (and other non-network kinds); `"local"`: network providers only at `localhost`, `127.0.0.1` or `::1`, fallbacks included |
| `sandbox` | `false`, `true` | `true`: code steps always run in the sandbox |
| `human_approval` | `"never"`, `"run"`, `"always"` | the least `--human-approval` an MCP server may start with |

An unknown key or value is an error. `policy set` changes keys and needs
`--yes` for a looser value; `policy reset --yes` removes the file.

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
| `A TO B IF word` [إذا] | the connection is active only if `word` holds for A's output: when the output is a JSON object with `word` as a top-level key (case-insensitive), its value is true, non-zero and non-empty (the strings `false`, `no`, `0`, `none`, `null`, `لا` and `خطأ` count as false); otherwise the output contains `word` (case-insensitive). `IF NOT word` [ليس] for the opposite. `IF key = value` (also written `key=value`, `key= value` or `key =value`; the word after `=` is always the value, even one spelled like a keyword, such as `pass`): the output is a JSON object whose top-level `key` (case-insensitive) equals `value`; `true` and `false` compare with booleans and with those strings, `null` and `none` with null, a number numerically, anything else as case-insensitive text; no JSON object or no such key: false |
| `A AMONG B C D` [بين] | A decides: the candidate named first in A's output (by number, id or title) is the only one that runs; if none is named and exactly one candidate's `expects` accepts the output, that one runs |
| `A AMONG B C D BY key` [حسب] | A's output must be a JSON object; the value of its top-level `key` (case-insensitive) must equal, ignoring case, a candidate's id, number, title or alias, and that one runs. Otherwise the step fails with the reason; nothing is guessed from the rest of the text |
| `A TO D THROUGH B C` [عبر] | A → B → C → D |
| `A TO B IF word ELSE C` [وإلا، والا] | also `A TO C IF NOT word` |
| `A LINK B` [ارتبط، اربط] | an order-only connection: B runs after A, is skipped if A is, and does not receive A's output; a file fed only by LINKs receives the original request. LINK takes no `IF` or `THROUGH` |
| `A PASS B` [مرر] | `A TO B` |
| `CALL name` [نفذ] | in place of a file: the saved path `name` is built (from its file, every time) and spliced in. Connections into it go to its files that nothing feeds; connections out of it leave from its final files. A file may not be both in the path and in the called path; a path may not call itself, directly or not. `AMONG` chooses among files only |
| `EACH A` [لكل] | A's `each` is `yes` in this path |
| `APPROVE A` [اعتمد] | A is approved before it runs in this path (with `approve: after` in the file: both) |
| `A USING m` [بواسطة] | A's model in this path: `--model` > `USING` > `model:` > `TAWJIH_MODEL` |
| `A RETRY n` [أعد، اعد] | n from 1 to 20: A's retries in this path (`--retries` wins) |
| `A UNTIL [NOT] w` [حتى] | needs `RETRY n` on A: while the condition (as for `IF`, including `key = value`) does not hold for A's output, A runs again, at most n more times, with its previous answer and the reason appended to its prompt; then the step fails (and its fallback, if any, runs) |
| `A PICK k` [انتق] | after A's contract and approval, only the value of key `k` of A's JSON object output is kept and passed on (a string as is, anything else as JSON); without that key the step fails. Contract checks skip connections out of A |

Statements chain left to right: `A TO B AND C TO D` connects A to B and C,
and both to D. `IF` applies to the connections of the `TO` just before it;
`A TO B IF w ELSE C TO D` continues from both B and C. `EACH` and `APPROVE`
come before a file, `USING`, `RETRY`, `UNTIL` and `PICK` after it, in any
order; they are saved with the path and belong to that file in that path
only, and the same setting given twice differently is an error. A path whose
statements hold settings or `CALL` but no connecting word is a sequence.
Statements are separated by `,`, `/`, `؛` or `،`; the Arabic tatweel (ـ) is
ignored in words. A statement `ITERATE n [UNTIL [NOT] word | key = value]`
[كرر، حتى، ليس] (n from 1 to 50, at most once in a path, never in a path
that is CALLed) makes the whole path run up to n times: round 1 on the
request, each later round with the previous round's result in place of the
request for the files nothing feeds (the original request is still given
to every model call); it stops when the result meets the condition, as
for `IF`, and fails after round n if it never does; without `UNTIL` it
runs exactly n rounds. Each round is a run of its own, with its own log
and record. `ITERATE` inside another statement is an error naming the
comma. `PARALLEL` [توازي، بالتوازي] and `CACHE` [مخزن] are
reserved and refused with the reason. An alias cannot be one of these
words.
A file listed without any connecting word in a file that has none is the
old form: a plain sequence, one file per line.

A path must not contain a loop, and a file appears in it once. `check` and
`generate` note an `IF` word that is not one of the keys its source file
promises in a `json(...)` contract, since it will be looked for in the text.

## Production

1. Files run one at a time, each after every file that feeds it. With
   `--max-workers N` (1 by default, at most 32), the files whose feeders
   have all finished (done, skipped or failed) run at the same time, up to
   N, in threads; a file whose step runs code (with `--allow-code`, its
   fallback included) or asks for approval on the terminal runs alone
   first. The log lists the steps in the plan's order. Outputs, the
   choices of `AMONG` and the skipping rule are the same as one at a time.
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
   names the step to ask again with `--resume --redo`. With `--on-failure
   continue`, a step that fails and has no fallback is recorded as failed,
   passes nothing on (the files that need it are skipped), and the run goes
   on; the record's `status` is `partial` and `failed_steps` lists them.

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
  input and its digest, and where the log and result are; `by`, who ran
  it (see Identity), and per step `approvals`: `when` (before, after),
  `mode` (ask, auto, deny), `approved`, `at`, `edited` when a person typed
  a replacement, and `by` when the mode was ask.
- `cache/<hash>.json`: one kept model answer, with its prompt.
- `runs/eval-<name>-<time>.json`: an evaluation, case by case.
- `state.json`: the digests used by `sync` and `status`.
- `index/embeddings.json`: the semantic search index.
- `history/<id>-<time>.md`: the previous version of a file, saved whenever
  a command or the Linux windows change it (see Versions).
- `paths/<name>.proposed`: a path proposed by `generate`, until accepted.

## Identity

`whoami set` keeps `{id, name, email, record}` in `identity.json` in the
person's settings folder (`$XDG_CONFIG_HOME/tawjih`, `%APPDATA%\tawjih` on
Windows, else `~/.config/tawjih`; `TAWJIH_IDENTITY` names another file),
never in the project. `id` is the one given (letters, digits, `. - _ :`,
up to 64) or `u-` and eight random hex digits, kept across changes.

Records note `by`: with `record` `name`, `{id, name}`; `id`, `{id}`;
`email`, `{id, name, email}`; `none`, nothing (`null`). Without an
identity, `{name, from: "git"}` from `git config user.name`, else
`{name, from: "system"}`; an email is never taken from git or the system.
`by` is in run records, evaluation records, dataset manifests, release
manifests, schedules and triggers, runtime `job.json`, and as
`decided_by` in decided assistant requests. `reviewed_by` in a file's
header is `name (id)`, `name` or `id`. A trace's
`langfuse.user.id` is `by.id`, else `by.name`, and never the email.

## Failures, approval, redaction and usage

Three optional header lines, set with `contract`:

- `retries: N` (0-9): a model answer that is empty or breaks `produces` is
  asked for again, up to N more times, with the prompt followed by a section
  "Your previous answer was rejected" that gives the reason. `--retries N`
  on `produce`, `run`, `eval` or `serve` overrides every file's value. Only
  an accepted answer is cached, under the original prompt.
- `fallback: ID`: when the step fails (input contract, code, model call,
  contract, no output), the fallback file's machine side runs on the same
  input, and its output, checked against the step's `produces`, is the
  step's output. A failure of the fallback stops the run. `check` reports a
  fallback that is missing, has no machine side, or whose `produces` does
  not fit the step's.
- `approve: before | after | both`: before the step, a person allows it to
  run; after it, a person allows its output to pass on, or types a
  replacement (checked against `produces`). The mode is `--approve ask`
  (read the answer from standard input), `auto` or `deny`. The default is
  `ask` when standard input is a terminal, otherwise `deny`; `eval` and
  `serve` default to `deny`. Refusal stops the run.

**Redaction**: when the project has `redact.txt`, every model call
(production, `edit`, `filter`, `generate`) passes through it. Lines are `@`
and a built-in pattern (email, phone, card, iban, national_id, ip), a word
or name (case-insensitive), or `re:` and a regular expression; `#` starts a
comment. Words and expressions apply first, then iban, card, email, ip,
national_id, phone; a card number must pass the Luhn check. Each distinct
match becomes `[LABEL_k]`; the placeholders found in the answer are
replaced by the originals. Logs and the cache keep the original text, since
they stay on the computer.

**Served model**: when the provider's answer names the model that
answered (`model` in Ollama, OpenAI-compatible and Anthropic answers), it
is kept as `served`: per step in `steps[].usage.served`, the distinct
names in the run's `usage.served`, in evaluation records, and as
`gen_ai.response.model` in traces.

**Usage**: each model call records the input and output tokens the provider
reports (Ollama `prompt_eval_count`/`eval_count`, OpenAI `usage`, Anthropic
`usage`), the seconds waited, and the characters sent and received. The run
record holds them per step (`steps[].usage`) and in total (`usage`: in, out,
seconds, calls, retries, fallbacks, redacted). `stats` sums records per path
and multiplies tokens by `prices.json` (per million, per model) when it has
the model.

## Examples

An optional fifth section, `=== examples ===` (a section marked
`=== anti_examples ===` is read into it), is written only when it has
content. It holds blocks: a line `Q:` (also `س:`, `question:`) starts one;
`A:` (`ج:`, `answer:`) the answer, `E:` (`expect:`, `متوقع:`) a text a
correct answer must contain (`;` separates several), `BAD:` (`خطأ:`) a wrong
answer, repeatable; a line without a key continues the previous field.
Every example needs `A:` or `E:`. It is build material: never sent at run
time.

Header lines: `split: train | eval | ignore` (absent means train);
`weight: W` (a number of 0 or more, absent means 1); `examples_sha`, the
digest of the human and reading sides when the examples were written through
`edit --side examples`, `eval promote` or `confirm --examples`; when the
digest differs the examples are old.

`check` reports unreadable blocks and a question (compared by its words,
case-insensitively) found in a `train` and an `eval` file as errors; the
same question with different answers, and duplicates, as warnings. Files
with `split: ignore` are left out of these checks and of data.

`dataset --source examples` writes one line per example of the files of a
split (default train): prompt the question (after the file's instructions
with `--with-rules`), answer `A:`; `--repeat-by-weight` repeats a line
round(W) times; `--dpo F` writes `{prompt, chosen: A, rejected: BAD}` per
`BAD:`. Every dataset is written with `F.manifest.json`: version, time,
source, split, counts, the SHA-256 of the data, and of each file or
evaluation record it came from.

## Evaluation

`eval NAME` runs the cases (see Managing a project) and records, per case,
the output, pass, the reason, the run's log and record, seconds and tokens.
`match: judge` asks the `--judge MODEL` (through `--judge-backend`, default
the run's backend) with a fixed grading prompt; the first word PASS or FAIL
decides; without `--judge` such a case is skipped and not counted.
`--direct` sends each input to the model without the path and records
`mode: direct`. `--repeat N` runs each case N times (never from the
answer cache, so not with `--resume`) and `--min-pass K` (1 to N, default
N) sets how many must pass; such a case records `passes` and `runs` (each
run's output, pass, reason, log, record, seconds and tokens), its `pass`
is `passes >= K`, and the record has `repeat`, `min_pass` and `served`,
the model names the provider reported. When `paths/NAME.baseline.json` exists, a case that passed
there and fails now is a regression, and the exit status is 1. Actions:

- `init NAME`: writes a commented template cases file; refuses to replace.
- `extract NAME [--branch REF] [--split eval]`: appends a case per example
  of the files of that split whose question is not already a case: expect
  the `E:` texts, or `A:` with `match: judge`.
- `synthesize NAME --backend B [--count N] [--branch REF]`: sends the human
  sides and examples of the path's files (never of `eval` files) and asks
  for N cases as JSON lines; readable lines go to
  `paths/NAME.cases.proposed.jsonl`; `--accept` appends them.
- `compare NAME [A B]`: A and B are record files or `path`, `direct`,
  `baseline` (default `path` and `direct`), case by case by name.
- `baseline NAME [--save]`: shows or saves the latest path evaluation.
- `diff NAME [--baseline]`: the latest against the one before (or the
  baseline): cases now failing and now passing; exit 1 if any fails.
- `blame NAME [--case C]`: per failed case, its run's steps with status,
  cache and whether each output holds all expected texts; then the failed
  step, or the first step that held them, or none.
- `promote NAME --to REF [--case C]`: appends `Q:` input, `A:`/`E:`
  expected, `BAD:` output to REF's examples (refused for an `eval` file),
  and moves those cases to `paths/NAME.promoted.jsonl`.
- `report NAME [-o F]`: Markdown summary of the latest evaluation, with the
  direct and baseline results when present.

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
  machine sides without contracts; machine sides no saved path uses; empty
  human sides; edits outside Tawjih; folders left in `work/`.
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
- `ingest [--with TOOL]` writes Markdown. `auto` takes the first installed
  of: for `.pdf` docling, marker (`marker_single`), pdftotext; for `.docx`
  pandoc when the document holds Office Math, else the built-in reader;
  for `.epub` pandoc, the built-in reader; `.odt` and `.rtf` pandoc;
  `.pptx` docling. Pandoc writes Markdown without spans, divs or
  attributes and formulas as `$...$`. The built-in `.docx` reader takes
  headings from the style's outline level or a style named "heading N" (or
  "عنوان N"), list paragraphs as `- `, tables as `| a | b |`, Office Math
  as its flattened text between `$`; the built-in `.epub` reader uses the
  HTML reader in spine order.
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

## Governance

**Lock.** `locked: yes` makes `save` refuse any change to a file's content
key: title, sides, examples, contracts, model, retries, fallback, approve,
split, weight, each, tags and links. Parent, level and order may change, so
a locked file can be moved. `rm`, `archive` and `rollback` (without
`--force`) refuse locked files. Only `unlock` writes a locked file.

**Aliases.** `resolve` takes an id, then a number, then an alias
(case-insensitive, one match). An alias is one word, not an id and not a
number, and unique in the project.

**Review.** `reviewed_sha` is the SHA-256 of the content key when the file
was reviewed; a file is "changed since reviewed" when the two differ.

**Stage.** `stage/<id>.<side>.json` holds `id`, `side`, `text`, `base` (the
side when staged), `why` and `staged`. `--apply` writes `text` as `edit`
would (a version in `history/`, `built_sha` or `examples_sha` updated) only
when the side still equals `base`.

**Mounts.** `mounts.json` maps a name to a project folder, relative to this
one. Mounted files are loaded with `mount` set, are never saved, recorded in
`state.json` or removed, and their roots sort after the project's own. An id
used in both projects is an error.

**FOR EACH.** A step with `each` takes a JSON list: the input when `each`
is `yes`, else the list under that key of the input object. Its machine
side runs once per item (an item that is not a string is passed as JSON);
`expects` and `produces` apply to each item. Its output is
`{"items": [...]}`, each answer parsed as JSON when it is JSON.

**Sandbox.** With `--sandbox` a code side runs in `python -I` with a
temporary working folder and only `PATH` and `PYTHONIOENCODING` in its
environment; it receives `code`, `input`, `origin`, `vars`, `steps`,
`meta`, `linked` (as data) and limits as JSON on standard input and returns
`output` and `vars` on standard output. On POSIX it sets `RLIMIT_AS`
(memory, default 512 MB), `RLIMIT_CPU` (the timeout) and `RLIMIT_FSIZE`
(16 MB).

**Releases.** `releases/NAME.zip` holds `release.json` (`name`, `created`,
`tawjih`, `note`, `files`: path → SHA-256, `digest`: SHA-256 of the sorted
`path digest` lines) and the files of `files/`, `paths/`, `redact.txt`,
`prices.json` and `mounts.json`. `verify` recomputes both levels.
`rollback` saves `before-rollback-<time>` first, deletes files of `files/`
and `paths/` not in the release, writes the release's files, and records the
state. `snapshot` is `release create snapshot-<time>`. `seal.json` has the
same `files` and `digest`; `seal --sign` writes `seal.json.asc` with
`gpg --detach-sign --armor`.

**Archive.** `archive/` holds the files of archived branches unchanged;
nothing reads them. `--restore ID` brings back ID and the archived files
below it, under its old parent when that still exists, else as a new tree.

**Templates.** `templates/NAME.json` holds `from` and `files`: for each, the
index of its parent in the list, its level relative to the first, and its
text. `scaffold` gives every file a new id.

**gc.** Versions in `history/` are grouped by id and sorted newest first;
those beyond `--keep` (or older than `--days`) are removed, except the
newest version of a file no longer in the project.

**format.** Each side: trailing spaces removed, three or more newlines
reduced to two, leading and trailing blank lines removed; examples
separated by one blank line; the title's spaces collapsed; repeated tags
removed.

**Schedules.** `schedules.json` maps a name to a job: `kind` (`every` with
`seconds`, `at` with `HH:MM`, or `change` with `files` and `seen`, the digest
of their content keys), `path`, `input` or `input_file`, `backend`, `model`,
`allow_code`, `paused`, `last_run`, `last_run_ts`, `last_status`.
`schedule --run-due` runs each job that is due once, with approval denied,
and records the outcome. An `at` job is due once a day, after its time.

## Management statements

`do` reads statements separated by new lines, `/`, `;` or `؛` (outside
quotes, which shlex rules govern; a line starting with `#` is skipped).
Each starts with a verb and becomes one command:

| Statement | Command |
|---|---|
| `CREATE t [UNDER\|INTO r] [FROM tpl]` [أنشئ، انشئ؛ تحت، في؛ من] | `new t --parent r`, or `scaffold tpl --title t --parent r` |
| `EDIT r SIDE s [USING x] request` [عدّل، عدل؛ جانب؛ بواسطة] | `edit r --side s --ask request --backend B [--model x] --stage` (`--yes` instead of `--stage` with `do --yes`). x names a provider when it is one, else a model. s: HUMAN [بشري], MACHINE or INSTRUCTIONS [آلي، الي، تعليمات], CODE [كود], READING [قرائي], EXAMPLES [أمثلة] |
| `MOVE r UNDER\|INTO d`, `COPY r UNDER\|INTO d` [انقل، انسخ] | `move r --to d`, `copy r --to d` |
| `MERGE b INTO a` [ادمج] | `merge a b` |
| `DELETE r` [احذف] | `rm r` (`--yes` with `do --yes`) |
| `CHECK [r]` [افحص] | `check [--branch r]` |
| `GC` [نظّف، نظف] | `gc` (`--yes` with `do --yes`) |
| `PATH name ...` [مسار] | `path name ...` |

Every statement is translated before any runs; one that is not understood
stops all. Without `--run` only the translation is shown. With it, the
commands run in order and the first failure stops the rest; `MERGE` is
skipped without `--yes`. These words are a vocabulary of their own, not
reserved in the path language.

## MCP server

`mcp` speaks JSON-RPC 2.0, one message per line, over standard input and
output (MCP stdio transport; protocol versions 2025-06-18, 2025-03-26 and
2024-11-05, the client's choice when supported). Methods: `initialize`,
`ping`, `tools/list`, `tools/call`; notifications get no reply; a batch, an
unknown method or a malformed message gets a JSON-RPC error. A tool that is
not allowed is a JSON-RPC error (-32602); a refused or failed call is a
result with `isError: true` and the reason as text.

Tools take only the arguments of their `inputSchema`: strings with a
pattern and length, enums and booleans. A file is a string matching
`^\w[\w.\-]{0,63}$`, resolved as `resolve` does. Free text is set on the
command's arguments after parsing, never parsed itself. Commands run with
standard input empty, standard output and error captured, and approval
`deny`. Before serving, the process duplicates its standard output for the
protocol and points file descriptor 1 at standard error.

**Approval.** With `--human-approval always` (the default) every tool of
the write level except `tawjih_propose`, and every tool of the run level,
needs a person; with `run` only the run level; with `never` none. First the
arguments are checked and the action is tested for a sure refusal (an
unknown file in `ref`, `parent`, `to`, `src`, `dst` or `fallback`; an
existing path without `replace`; a missing path; a path with code steps
when the server has no `--allow-code`), which is returned at once. Then,
when the client declared the `elicitation` capability at `initialize`, the
server sends `elicitation/create` with a message describing the action
(for runs: provider, model, steps with instructions, characters, a token
estimate of characters / 4, model calls from steps x (1 + retries), and
code) and the schema `{approve: boolean}`; only `action: accept` with
`approve: true` runs it. Messages arriving meanwhile are handled after the
answer. Without elicitation, or when the client answers it with an error,
the action is written to `requests/rYYYYMMDD-HHMMSS-XXXX.json` (`tool`,
`args`, `summary`, `status: pending`, `backend`, `model`, `allow_code`) and
the tool returns `PENDING` with the id. `approve ID --yes` runs it with
those settings and stores `result` and `status` (`done` or `failed`);
`--no` sets `declined`; `tawjih_request` reads it.

**Data notice.** `tawjih_show`, `tawjih_search`, `tawjih_staged` and the
result of `tawjih_request` begin with a notice that the text is data;
lines matching `INJECTION_RE` (phrases such as "ignore previous
instructions", "تجاهل التعليمات") are listed in a warning.

`tawjih_propose` calls `stage_change` (Governance, Stage) with `why`
prefixed `(proposed through MCP)`; it refuses locked and mounted files.
`tawjih_run` and `tawjih_eval` use `--backend` and `--model` from the
command line, `allow_code` and `sandbox` both equal to `--allow-code`.
`manual` is refused as a backend, and `--allow-code` needs `--allow-run`.
**Durable runs.** `tawjih_run` with `key` (matching `[\w.:-]{1,128}`)
and `wait` (0-600 seconds, default 30): the key's fingerprint is the
SHA-256 of the path, input, backend, model and `allow_code`. A key file
`mcp-runs/keys/<sha256(key)[:32]>.json` (`{run, fingerprint}`) is created
exclusively, so two clients cannot start two runs. Then
`mcp-runs/<id>.json` (id `mYYYYMMDD-HHMMSS-XXXXXXXX`) is written with
`status: working`, before the id is returned, and a separate process
(its own session, or a detached process on Windows) sets `pid` and
`started`, runs the path once as `tawjih_run` would, and replaces the file
with `status: completed` and `output`, or `failed` and `error`. Files are
replaced in one step. A key already used with the same fingerprint
returns its run, with no approval and no new run; with another, an error
"idempotency conflict", and nothing starts. A request refused by the
checks (an unknown path, code without `--allow-code`) leaves the key
unused. `tawjih_run_status` (listed with the run level, but read-only and
never approved) returns `run ID: STATUS` and the output or error; a run
whose process has ended while `working` becomes `failed`, and so does one
whose process never started within 120 seconds. Without `key`,
`tawjih_run` is unchanged.

Each call appends `{time, tool, args, status}` to
`history/mcp-YYYYMMDD.jsonl`, with `title`, `text`, `input`,
`statements`, `why` and `query` replaced by `<N characters>`.

## Architectures

An architecture is a node tagged `architecture` whose machine face's
instructions hold a JSON object, the specification:

| Key | Value |
|---|---|
| `schema` | 1 |
| `family` | `decoder` |
| `layers`, `hidden`, `heads`, `kv_heads`, `ffn_hidden`, `vocab`, `context` | whole numbers above 0 |
| `ffn` | `swiglu`, `gelu` or `relu` |
| `norm` | `rmsnorm` or `layernorm` |
| `positions` | `rope` (with `rope_theta` above 0), `learned` or `none` |
| `tie_embeddings`, `bias` | true or false |
| `moe` | null, or `{"experts": E, "top_k": K}` with 2 ≤ E and 1 ≤ K ≤ E |
| `dropout` | from 0 up to, not including, 1 |
| `name`, `notes`, `derived_from` | optional; `derived_from` is the id of the source |

Any other key is an error. `hidden` must divide by `heads`; `kv_heads` must
divide `heads` (equal: MHA, 1: MQA, else GQA); with `rope` the head size
`hidden / heads` must be even. Parameters, with h = hidden, d = h / heads,
k = kv_heads · d, f = ffn_hidden, b = 1 with bias else 0:

- embedding: vocab · h, plus context · h with learned positions;
- attention per layer: 2h² + 2hk + b(2h + 2k);
- MLP per layer: 3hf + b(2f + h) for swiglu, 2hf + b(f + h) otherwise; with
  `moe`, E times that plus h · E for the router (per token: K times plus
  the router);
- norms: h each for rmsnorm, 2h for layernorm; two per layer and one final;
- head: vocab · h unless `tie_embeddings`.

`arch derive` and `arch mutate --yes` make the new node a child of the
source (its level plus one, at most 6) with a `derived-from` link.
`arch export` maps rmsnorm + swiglu + rope to a Llama configuration
(Mixtral with `moe`, which allows no bias), and layernorm + gelu + learned
positions with bias, tied embeddings and MHA to GPT-2; any other
combination is refused. `arch build` writes a job (see Runtimes) whose
`out/validation.json` holds the parameters counted by the library, the
logits' shape and the loss for two random sequences, and whether the count
matches; it is kept in `models/architectures/ID/`.

## Runtimes

Runtimes are read from `$TAWJIH_RUNTIMES`, else
`$XDG_CONFIG_HOME/tawjih/runtimes.json`, `%APPDATA%\tawjih\runtimes.json`
on Windows, or `~/.config/tawjih/runtimes.json`, never from a project:
`{"default": NAME, "runtimes": {NAME: {"kind": ..., ...}}}`. `local` exists
without an entry. Kinds: `local`; `ssh` with `host` (user@machine),
optional `python` and `workdir` (default `tawjih-jobs`); `command` with
`submit` and optional `status`, `logs`, `cancel`, `collect`, `check`
command lines, in which `{job}` is the job folder (quoted), `{id}` its id,
`{name}` its name and `{workdir}` the setting. A setting that looks like a
key or password is refused.

A job is `models/jobs/ID/` in the project: `run.sh` (it changes to its own
folder, writes `pid`, runs the command with output to `log.txt`, then
writes its exit status to `exit`), the files the command needs at relative
paths, `out/` for its results, and `job.json` (id, name, runtime, command,
where results go, status). A local job runs here and is waited for; an ssh
job is copied with scp to `workdir/ID` and started with nohup; a command
job runs `submit`. A status of `0` is done and another number failed.
`collect` brings `out/` back (scp, or the `collect` command) and copies it
to where the job's results go. Starting a job whose runtime is not this
computer needs `--yes`, and is refused when the project's policy has
`allow_network` other than true.

## Program layout

`tawjih.py` adds its folder to `sys.path`, imports the package `tawjihlib`,
copies every name of its modules into its own namespace and runs `main()`.
The modules, in order: `core`, `providers`, `tracing`, `engine`, `commands`,
`library`, `manage`, `outputs`, `evaluate`, `governance`, `runtimes`,
`arch`, `script`, `mcpserver`, `home`, `monitor`, `cli`. A module
imports only from those before it, by name, and only the standard library
from outside; each lists its names in `__all__`.

## Live runs

While a path runs, the run keeps `history/live/PID-NAME-STAMP.json`: the
record that goes to `history/` at the end, without the input text. It is
written when the run starts, when each step starts and when each step
ends, by replacing the file whole, and removed when the run ends (done,
partial or failed). Each step carries `status` (`waiting`, `running`,
`done`, `failed`, `skipped`, or `stopped` if the step was interrupted),
`started_at` (seconds since the epoch) and `seconds` (its wall time). The
record carries `pid`, `t0` and, at the end, `seconds`. A live file whose
process no longer exists belongs to a run that was stopped. `tawjih top`
reads these files, the records in `history/` and `models/jobs/`; it
writes nothing.

## Traces

With `--trace`, a run is sent at its end as one OTLP/HTTP JSON request to
`LANGFUSE_HOST` + `/api/public/otel/v1/traces` (Basic authentication with
`LANGFUSE_PUBLIC_KEY:LANGFUSE_SECRET_KEY`, `x-langfuse-ingestion-version:
4`), or to `TAWJIH_OTLP_ENDPOINT` with `TAWJIH_OTLP_HEADERS`. Spans:

- the root, `path NAME`, `langfuse.observation.type` `chain`, with the input
  and the result;
- one per step that ran, under the root: `NUMBER TITLE`, type `span` (or
  `tool` for a step with code), level `ERROR` when it failed and `WARNING`
  when skipped or stopped, metadata `file_id`, `number`, `tree`,
  `machine_sha`, `status`;
- one per model call, under its step: type `generation`, with
  `langfuse.observation.model.name`, the input and output as sent and
  received (after redaction), and `langfuse.observation.usage_details`
  `{"input", "output"}` when the provider reports tokens.

Every span carries `langfuse.trace.name`, `langfuse.trace.tags` (`tawjih`
and the trees), `langfuse.release` and, when there is one,
`langfuse.session.id`. Texts are cut at 100,000 characters. A request that
fails is written to `history/traces-pending/` as it would have been sent.
