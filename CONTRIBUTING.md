# Contributing to Tawjih

Thank you for considering contributing to tawjih! We welcome contributions from researchers, software engineers, prompt designers, and domain experts.

This document provides guidelines and instructions for contributing to the repository. Following these helps maintain project quality, reproducibility, and smooth review processes.

## 1. Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please maintain a respectful, constructive, and inclusive environment in all discussions, issues, and pull requests.

## 2. Core Architectural Principles

When contributing code, documentation, or workflows to tawjih, please keep the following core design constraints in mind:

- **Zero External Runtime Dependencies:** The core execution engine (`tawjih.py` and `tawjihlib/`) must strictly rely only on the Python standard library. External packages are strictly disallowed for core functionality to ensure long-term reproducibility, portability, and zero-maintenance execution in air-gapped environments.
- **Three-Faced File Model:** All domain knowledge and step nodes are structured around the three faces: human (concepts/notes), machine (instructions, optional embedded code, examples), and science (source passages extracted from LaTeX, PreTeXt, HTML, or Markdown).
- **Bilingual Symmetry:** Declarative path syntax, CLI options, and reporting maintain symmetrical support for both English and Arabic language keywords (e.g., `TO`/`إلى`, `IF`/`إذا`, `EACH`/`لكل`, `RETRY`/`أعد`, `ITERATE`/`كرر`).
- **Context Window Isolation:** tawjih strictly enforces build-time vs. run-time isolation. Only the machine side is transmitted to language models during execution.

## 3. How You Can Contribute

### A. Reporting Bugs

If you encounter a bug or unexpected behavior:

1. Search the [Issues](https://github.com/AlmodinyMk/LLM-tawjih-Model-building/issues) tab to ensure it hasn't already been reported.
2. Open a new issue with the **Bug report** form, which asks for:
   - your operating system (Linux distribution, Windows build, etc.);
   - your Python version (`python3 --version`) and tawjih version (`tawjih --version`);
   - clear, reproducible step-by-step instructions or a minimal example;
   - expected vs. actual behavior.

If you tried tawjih with a real model or on a remote machine, the **Test report** form collects your results, failures included. Questions go to [Discussions](https://github.com/AlmodinyMk/LLM-tawjih-Model-building/discussions), and security problems are reported privately as described in [SECURITY.md](SECURITY.md).

### B. Architectural & Scientific Feedback

For high-level design questions, DSL proposals, or scientific application reviews:

- Participate in the architectural reviews outlined in [REVIEW.md](REVIEW.md).
- Open or join threads in the GitHub Discussions tab.

### C. Submitting Code or Documentation Fixes

1. Fork the repository and create a feature branch off `main`:

   ```
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, adhering to the code style and the standard-library constraint.
3. Add or update unit tests in `tests/` covering your changes.
4. Run the test suite locally (see below).
5. Submit a pull request targeting the `main` branch.

`main` is protected: every change arrives through a pull request, and the tests must pass on Linux and Windows (Python 3.11 and 3.13) before it can be merged. CodeQL analyses each pull request, and ClusterFuzzLite fuzzes it when it changes code.

## 4. Local Development & Testing

### Setting Up the Environment

Clone your fork locally:

```
git clone https://github.com/your-username/LLM-tawjih-Model-building.git
cd LLM-tawjih-Model-building
```

No pip install of external dependencies is needed for the core project.

### Running the Test Suite

tawjih uses Python's built-in `unittest` framework. Run all tests with verbose output:

```
python -m unittest discover -s tests -v
```

The tests include fuzz targets (`fuzz/targets.py`) run with random inputs; set `TAWJIH_FUZZ_RUNS` for more. The tests use an identity file of their own, so they never write into your settings.

On Linux systems, you can also try the desktop interface (`linux/tawjih_desk.py`):

```
python3 linux/tawjih_desk.py -p examples/quadratic
```

## 5. Pull Request Guidelines

Before submitting your PR, please verify the following checklist:

- [ ] **Standard Library Compliance:** no external libraries were added to `pyproject.toml` or imported into `tawjihlib/`.
- [ ] **Tests Pass:** all unit tests in `tests/` pass without errors or warnings, and new behavior has tests.
- [ ] **Documentation Updates:** if CLI options, path syntax keywords, or core behavior changed, `README.md`, `README.ar.md` and `SPEC.md` are updated accordingly.
- [ ] **Clear Commit Messages:** concise and descriptive commit messages summarizing the changes.
- [ ] **Bilingual Consistency:** user-facing keywords and messages keep symmetry in English and Arabic where applicable.

## 6. Licensing & AI Disclosure

- **License:** By contributing to tawjih, you agree that your contributions will be licensed under the project's license, CC BY-NC-SA 4.0 (see [LICENSE](LICENSE)).
- **Generative AI Transparency:** If you use AI assistants (such as Claude, ChatGPT, or Copilot) to generate or refine code or documentation, please state this transparently in your pull request description and keep full human oversight over correctness and test coverage.

## 7. Contributors

Contributors are listed in the README (in both languages) through the [all-contributors](https://allcontributors.org) bot. Substantive contributions are acknowledged in the release notes.
