---
title: 'Tawjih: A Zero-Dependency, File-Based Workflow Engine for Auditable Large Language Model Orchestration'
tags:
  - Python
  - large language models
  - prompt engineering
  - workflow orchestration
  - reproducibility
  - bilingual DSL
authors:
  - name: Mohammed Khalid Almodiny
    orcid: 0009-0003-5696-1724 # Replace with your ORCID
    affiliation: 1
affiliations:
  - name: Independent Researcher  # Update if needed
    index: 1
date: 22 November 2026
bibliography: paper.bib
---

# Summary

`tawjih` (توجيه أوامر الآلة) is an open-source, zero-dependency Python framework for structuring domain knowledge and orchestrating Large Language Model (LLM) workflows through auditable, deterministic execution paths. Implemented exclusively using the Python standard library, `tawjih` organizes knowledge into hierarchical trees of small Markdown files—each holding three distinct faces (`human`, `machine`, and `science`, the reading side)—and connects them via a declarative, bilingual Domain-Specific Language (DSL) supporting symmetrical English and Arabic syntax. The framework enforces strict separation between build-time knowledge synthesis and run-time execution, providing JSON Schema contracts, separate-process execution with resource limits, pre-flight sensitive-data redaction, deterministic caching, and OpenTelemetry/Langfuse lineage tracing.

# Statement of Need

As Large Language Models are increasingly integrated into scientific research, mathematical reasoning, and technical documentation workflows, practitioners face three recurring engineering challenges:

1. **Context Pollution and Instruction Drift:** Conversational and agentic pipelines often accumulate intermediate reasoning, background literature, and human notes inside a single prompt context, inflating token costs and degrading model adherence. Moreover, when source documents or conceptual notes are updated, dependent prompts frequently drift out of sync without detection.
2. **Opaque Execution and Dependency Bloat:** Many modern orchestration frameworks embed prompt logic inside imperative Python code and rely on deep external dependency trees. This complicates version-control auditing (`git diff`) by domain experts and threatens long-term reproducibility in restricted or air-gapped environments.
3. **Structural Unreliability and Execution Risk:** Integrating LLMs with local data transformations requires deterministic schema validation between steps, isolated execution environments for generated or node-embedded scripts, and privacy safeguards prior to transmitting data to external model providers.

`tawjih` addresses these needs by treating prompt engineering as a transparent, compiler-like workflow over plain-text files that remain inspectable, reproducible, and verifiable without third-party runtime dependencies.

# State of the Field

Several established paradigms and software tools address adjacent aspects of LLM programming and reproducible computation:

* **Literate Programming and Computational Documents:** Systems such as WEB [@Knuth:1984], Emacs Org-mode Babel, and Quarto unify human narrative and executable code within plain-text documents. However, they are designed for document rendering and general code execution rather than governing LLM context boundaries or validating inter-step model contracts.
* **Programmatic Prompt Optimization:** Frameworks such as DSPy [@Khattab:2023] compile declarative signatures into optimized prompts using metric-driven search. While effective for automated prompt tuning, DSPy abstracts away the underlying prompt text, whereas `tawjih` prioritizes explicit human authoring, source provenance, and manual auditability.
* **Constrained Decoding and Query Languages:** Tools such as LMQL [@BeurerKellner:2023] enforce structural constraints during token generation. `tawjih` complements these approaches by operating agnostic to the inference backend—supporting local models via Ollama as well as OpenAI-compatible and Anthropic APIs—and enforcing post-generation JSON Schema contracts with bounded retries (`RETRY` and `ITERATE n UNTIL`).

# Software Architecture

`tawjih` is distributed as a lightweight command-line tool (`tawjih`) and library (`tawjihlib/`) built solely on Python's standard library, ensuring cross-platform compatibility across Linux and Windows. Its architecture rests on four core mechanisms:

1. **Three-Sided Knowledge Nodes and Staleness Detection:** Each node (a folder `files/<id>/` with one file per face) resides in a six-level hierarchy (`general` to `part`) and separates conceptual notes (`human`), executable LLM instructions and optional Python code (`machine`), and structured source excerpts (`science`, the reading side). A staged document ingestion command (`deposit`) extracts passages from `.tex` (expanding custom macros and preserving LaTeX math), `.md`, `.html`, `.xml`/PreTeXt, and `.txt` sources while deliberately rejecting `.pdf` files to prevent mathematical symbol corruption. A SHA-256 digest of the `machine` side tracks synchronization; subsequent edits to `human` or `reading` automatically flag the node as stale.
2. **Build-Time vs. Run-Time Isolation:** During build time (`open`, `edit`), authors synthesize `reading` passages into `human` concepts and `machine` instructions. During run time (`produce`), `tawjih` transmits *only* the node's `machine` side, the initial request, and contract-validated outputs from upstream nodes, guaranteeing zero leakage of background notes into the model's context window.
3. **Bilingual Declarative Path DSL and Governance:** Workflows are defined as directed acyclic graphs or bounded iterative loops using human-readable statements in English or Arabic (e.g., `TO`/`إلى`, `AND`/`و`, `IF`/`إذا`, `AMONG`/`بين`, `EACH`/`لكل`, `APPROVE`/`اعتمد`). Execution is governed by pre-flight `check` validation, `contract` schema enforcement, separate-process `sandbox` isolation for node scripts, project-level `policy` restrictions, and pre-transmission PII `redact` filtering.
4. **Reproducibility, Tracing, and Ecosystem Integration:** Every execution records step-by-step prompts, machine digests, inputs, outputs, and token usage in `runs/` and `history/`, supporting resumption from cached step outputs (`--resume`) and export to Langfuse or OpenTelemetry collectors. `tawjih` also exposes a Model Context Protocol (`mcp`) server interface and remote execution over argument-safe SSH/SCP (`runtime`).

# Acknowledgements

The author thanks the open-source research software engineering community for architectural feedback and verification benchmarks conducted prior to the `v1.0.0` release.

# References
