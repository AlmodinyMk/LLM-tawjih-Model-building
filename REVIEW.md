# Request for Independent Scientific and Architectural Review — Tawjih

**Repository:** `AlmodinyMk/LLM-tawjih-Model-building`  
**Package:** `tawjih` (PyPI)  
**Current Version:** `v0.17.1` (Pre-1.0 Experimental)  
**Specification:** [`SPEC.md`](./SPEC.md)  

---

## 1. Purpose of This Document

`tawjih` (توجيه أوامر الآلة) is a zero-dependency, file-based framework for structuring knowledge and directing Large Language Models (LLMs) through auditable, deterministic workflows. Implemented exclusively with the Python standard library, it explores an alternative to code-heavy agent frameworks and opaque conversational state.

Before freezing the specification for **`v1.0.0`** and submitting the software for formal peer review (e.g., to the *Journal of Open Source Software* or Research Software Engineering forums), we invite independent researchers, systems architects, and ML engineers to critically evaluate the project's underlying architectural assumptions, identify blind spots, and guide improvements to its execution and verification model.

---

## 2. System Context & Current Verification Status

To ensure transparent evaluation, reviewers should note the exact boundaries of what has and has not been verified in `v0.17.0`:

* **Verified:**
  * Core file model, tree hierarchy (up to 6 levels), SHA-256 machine-side digesting, and staleness detection.
  * Bilingual declarative path parser and DAG/bounded-loop execution engine (`tawjih.py` and `tawjihlib/`).
  * Schema validation (`contract`), process-level code isolation (`sandbox`), restrictive project policies (`policy`), pre-flight redaction (`redact`), caching (`--resume`), and lineage tracing (`trace`).
  * Automated unit and workflow test suites (`tests/`) passing on both **Linux** and **Windows** with an in-process stand-in model and a local HTTP stand-in for an OpenAI-compatible server; live provider tests (`tests/test_live.py`) are skipped unless `TAWJIH_LIVE_TESTS=1`.
  * Since `v0.16.0`: identities in run, approval and review records (`whoami`); durable MCP runs with idempotency keys, tested with a real stdio client killed mid-run against a stand-in model held on a signal; `eval --repeat N --min-pass K`; the MoE router's load-balancing loss checked on a real Mixtral build.
  * Since `v0.17.0`: model servers (`tawjih server`: ten engines and any command, health, tuning, benchmarks), `batch` across servers, size conditions and a provider per step in paths, provider options with Ollama's `num_ctx`, the Gemini provider, training pass-through (`--train-arg`, `--model-arg`, `--gpus`, `script`), and the board of `top` with servers, GPU free memory, CPU and RAM, also as a native Tkinter window (`dashboard`, Arabic or English). Engines, `nvidia-smi` and `lmdeploy` were exercised against stand-ins only.
  * Every change is analysed with CodeQL, the path language, contracts, examples and redaction are fuzzed (ClusterFuzzLite with Atheris on pull requests, random inputs in the tests), and releases are signed with Sigstore.
* **Not Yet Verified (Open Milestones for `v1.0.0`):**
  * End-to-end integration benchmarks against live commercial and local LLM providers (e.g., live Ollama instances, OpenAI-compatible APIs, Anthropic) under non-deterministic generation variance.
  * Human usability and ergonomics testing of the experimental Linux desktop interface (`linux/`).
  * The optional training bridge (`plugins/neural_bridge.py`) on a real GPU, the ssh runtime against another computer, uploads to the real Hugging Face Hub, vLLM serving, and traces to a real Langfuse server (each was exercised only locally or against stand-ins).
* **Explicit Non-Goals:**
  * `tawjih` is **not** designed for open-ended conversational chatbots, high-throughput vector RAG over millions of unstructured documents, or complex imperative agent loops where code-first frameworks (such as LangGraph or LlamaIndex) are better suited.

---

## 3. Core Architectural Axes for Review

We specifically seek critical feedback across the following four questions:

### Axis 1: The Three-Sided File Representation (`human` / `machine` / `reading`)
* **Architectural Premise:** Each knowledge node (a folder `files/<id>/` with `head.json`, `human.md`, `machine.md` and `science.md`) separates conceptual human intent (`human`), executable LLM instructions and optional Python transformations (`machine`: `instructions` + `code`), and source passages (`science` face, also called `reading`) extracted from structured academic documents (`reading`: `.tex`, `.md`, `.xml`/PreTeXt, `.html`, `.txt`, while deliberately rejecting `.pdf` to preserve mathematical notation). A SHA-256 digest tracks whether `human` or `reading` content has drifted from the `machine` specification (`staleness`).
* **Questions for Reviewers:**
  1. Does this tri-partite separation offer a meaningful architectural abstraction for reducing instruction drift and maintaining provenance in knowledge-intensive LLM workflows?
  2. What scaling or maintenance bottlenecks emerge when organizing domain knowledge into six-level file trees (`general` → `part`) rather than relational or vector stores?

### Axis 2: Strict Separation of Build Time and Run Time
* **Architectural Premise:** `tawjih` enforces a compiler-like boundary between **Build Time** (`open`, `edit`, `deposit`—where sources are synthesized into machine instructions) and **Run Time** (`produce`—where the LLM receives *only* the node's `machine` side, the original request, and contract-validated outputs from upstream nodes). `human` and `reading` sides are never transmitted at run time.
* **Questions for Reviewers:**
  1. Does strict context isolation at run time measurably improve reproducibility, token efficiency, and auditability (`runs/`, `history/`) compared to accumulative message-passing architectures?
  2. Under what problem classes does withholding broader tree context become overly restrictive, and how might cross-tree handoffs (`A TO B`) be structured to mitigate context starvation without reintroducing context pollution?

### Axis 3: Bilingual Declarative Path DSL, Contracts, and Runtime Controls
* **Architectural Premise:** Workflows are expressed in a declarative path language supporting symmetrical English and Arabic keywords (`TO`/`إلى`, `AND`/`و`, `IF`/`إذا`, `AMONG`/`بين`, `EACH`/`لكل`, `APPROVE`/`اعتمد`, `RETRY`/`أعد`, `ITERATE`/`كرر`) rather than imperative Python glue code. Execution safety and structural reliability are governed via post-generation schema checks (`contract` with JSON keys or JSON Schema), bounded retries (`RETRY` / `ITERATE n UNTIL`), separate-process execution with memory, time and file-size limits but no network isolation (`sandbox`), project-level constraints (`policy`), and pre-transmission masking (`redact`).
* **Questions for Reviewers:**
  1. Does combining a human-readable bilingual DSL with contract-checked handoffs provide practical governance advantages for domain experts reviewing workflows in `git diff`?
  2. From a security and reliability standpoint, what are the primary weaknesses in relying on post-generation contract validation + retries (as opposed to token-level constrained decoding) and OS-process sandboxing for untrusted node code?

### Axis 4: Positioning Relative to Prior Art and Related Systems
* **Architectural Premise:** `tawjih` intersects several established paradigms: Literate Programming (*WEB*, *Org-mode Babel*, *Quarto*), programmatic prompt optimization (*DSPy*), constrained decoding (*LMQL*, *Outlines*, *Guidance*), and evaluation/orchestration engines (*Inspect AI*, *LangGraph*).
* **Questions for Reviewers:**
  1. Which existing research frameworks or production systems already address context governance and reproducible prompt chaining in a comparable or superior way that we should benchmark against?
  2. What minimal, standardized evaluation suite (e.g., multi-step mathematical reasoning, LaTeX-grounded synthesis, or schema-constrained extraction) would provide the strongest empirical validation for `v1.0.0`?

---

## 4. Five-Minute Reviewer Quickstart

Reviewers can inspect the specification and run the local verification suite without external dependencies or API accounts:

```bash
git clone https://github.com/AlmodinyMk/LLM-tawjih-Model-building.git
cd LLM-tawjih-Model-building

# 1. Inspect the formal specification and security model
cat SPEC.md
cat SECURITY.md

# 2. Run the automated test suite (Python stdlib unittest)
python -m unittest discover -s tests

# 3. Inspect the bundled multi-tree example project
#    (the first command asks your name once; TAWJIH_NO_SETUP=1 skips it)
python tawjih.py -p examples/quadratic trees
python tawjih.py -p examples/quadratic check
python tawjih.py -p examples/quadratic produce branch --plan
```

### Key Files to Inspect
* [`SPEC.md`](./SPEC.md) — Formal contract for file headers, machine digests, path semantics, and execution records.
* [`tawjih.py`](./tawjih.py) & `tawjihlib/` — CLI entry point, parser, graph planner, contract validator, and runner implementation.
* `examples/quadratic/` — Reference two-tree project demonstrating cross-tree handoffs and contracts.
* `tests/` — Cross-platform test suite covering path parsing, caching, staleness, contracts, sandboxing, the MCP server and durable runs.
* `fuzz/` — Fuzz targets for the path language, contracts, examples and redaction.

---

## 5. How to Contribute Feedback

We welcome critique at any level of depth—from brief architectural notes on a single axis to detailed code or specification reviews:

1. **GitHub Issues / Discussions:** Open an issue tagged `architecture-review` referencing one or more of the four axes above.
2. **Pull Requests:** Propose concrete refinements to `SPEC.md`, edge-case tests in `tests/`, or hardening improvements to `sandbox` and `contract` validation.
3. **Citation & Attribution:** Substantive architectural reviews and benchmark contributions will be acknowledged in the repository's release notes and [`CITATION.cff`](./CITATION.cff) prior to the `v1.0.0` release.
