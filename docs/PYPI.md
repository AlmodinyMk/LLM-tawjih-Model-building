# tawjih — توجيه أوامر الآلة

**Direct a language model through small text files that you write and
review yourself, join them into paths, and know where every answer came
from.** Knowledge is kept in trees of small files; each file has a human
side, a machine side (instructions for a model and optional Python code), a
reading side with its sources, and optional worked examples. When a path
runs, only the machine side is sent; every run is saved and can be traced.

Python 3.11 or later, standard library only, Linux and Windows.
**Experimental:** the test suite passes, but no real AI provider has been
used with it yet.

```
pip install tawjih
tawjih                 # the home screen
tawjih home start      # your first project, step by step
tawjih init
tawjih new "Mathematics"
tawjih new "Algebra" --parent 1
tawjih tree
```

- A path language in English and Arabic: `TO`, `AND`, `IF … ELSE` (also `IF key = value`), `AMONG … BY key`,
  `THROUGH`, `LINK`, `CALL`, with per-step `EACH`, `APPROVE`, `USING`,
  `RETRY … UNTIL`, `PICK`.
- Contracts between steps (keys or a JSON Schema), staleness when sources
  change, a cache to resume.
- Named providers (`providers.json`): Ollama, any OpenAI-compatible server,
  Anthropic, with retries, backoff and fallback chains; keys stay in
  environment variables.
- A home screen in the terminal (`tawjih` alone): the project at a glance,
  the commands as cards, and `tawjih home TOPIC` explanations.
- A project policy that can only restrict: no code, local providers only,
  always sandboxed, a least level of human approval.
- Evaluation: cases from the examples, a path compared with the model asked
  directly, baselines, datasets with SHA-256 manifests, lineage with `trace`.
- Governance: locks, review, staged proposals, sealed releases, schedules.
- `tawjih mcp`: an MCP server for AI assistants, where a person approves
  every write and every run.
- `tawjih do`: management statements in English or Arabic.

Full documentation, the example project, the Linux windows and the optional
training bridge are in the repository:
<https://github.com/AlmodinyMk/LLM-tawjih-Model-building>
([العربية](https://github.com/AlmodinyMk/LLM-tawjih-Model-building/blob/main/README.ar.md)).

![How Tawjih is built](https://raw.githubusercontent.com/AlmodinyMk/LLM-tawjih-Model-building/main/docs/images/architecture-en.png)

Licensed CC BY-NC-SA 4.0: noncommercial use with credit and the same
license for adaptations; commercial use needs permission from the author.
