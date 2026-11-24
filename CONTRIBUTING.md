Contributing to Tawjih

Thank you for considering contributing to tawjih! We welcome contributions from researchers, software engineers, prompt designers, and domain experts.

This document provides guidelines and instructions for contributing to the repository. Following these helps maintain project quality, reproducibility, and smooth review processes.

1. Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please maintain a respectful, constructive, and inclusive environment in all discussions, issues, and pull requests.

2. Core Architectural Principles

When contributing code, documentation, or workflows to tawjih, please keep the following core design constraints in mind:

Zero External Runtime Dependencies: The core execution engine (tawjih.py and tawjihlib/) must strictly rely only on the Python standard library. External packages are strictly disallowed for core functionality to ensure long-term reproducibility, portability, and zero-maintenance execution in air-gapped environments.

Three-Faced File Model: All domain knowledge and step nodes are structured around the three faces: human (concepts/notes), machine (instructions, optional embedded code, examples), and science (source passages extracted from LaTeX, PreTeXt, HTML, or Markdown).

Bilingual Symmetry: Declarative path syntax, CLI options, and reporting maintain symmetrical support for both English and Arabic language keywords (e.g., TO/إلى, IF/إذا, EACH/لكل, RETRY/أعد, ITERATE/كرر).

Context Window Isolation: tawjih strictly enforces build-time vs. run-time isolation. Only the machine side is transmitted to language models during execution.

3. How You Can Contribute

A. Reporting Bugs

If you encounter a bug or unexpected behavior:

Search the Issues tab to ensure it hasn't already been reported.

Open a new Issue providing:

Your operating system (Linux distribution, Windows build, etc.).

Python version (python3 --version).

Clear, reproducible step-by-step instructions or a minimal code example.

Expected vs. actual behavior.

B. Architectural & Scientific Feedback

For high-level design questions, DSL proposals, or scientific application reviews:

Participate in existing architectural reviews outlined in REVIEW.md.

Open or join threads in the GitHub Discussions tab.

C. Submitting Code or Documentation Fixes

Fork the repository and create a feature branch off main:

git checkout -b feature/your-feature-name


Make your changes adhering to code style and standard-library constraints.

Add or update unit tests in tests/ covering your changes.

Verify all tests pass locally across both Linux and Windows environments.

Submit a Pull Request targeting the main branch.

4. Local Development & Testing

Setting Up the Environment

Clone your fork locally:

git clone https://github.com/your-username/LLM-tawjih-Model-building.git
cd LLM-tawjih-Model-building


No pip install of external dependencies is needed for the core project.

Running the Test Suite

tawjih uses Python’s built-in unittest framework. Run all tests with verbose output:

python -m unittest discover -s tests -v

The tests include fuzz targets (fuzz/targets.py) run with random inputs;
set TAWJIH_FUZZ_RUNS for more. On pull requests, ClusterFuzzLite also
fuzzes them with Atheris, and CodeQL analyses the code.


On Linux systems, you can also test the desktop interface (linux/tawjih_desk.py) or run virtual display tests:

python3 linux/tawjih_desk.py -p examples/quadratic


5. Pull Request Guidelines

Before submitting your PR, please verify the following checklist:

[ ] Standard Library Compliance: Ensure no external libraries were added to pyproject.toml or imported into tawjihlib/.

[ ] Tests Pass: All unit tests in tests/ execute successfully without errors or warnings.

[ ] Documentation Updates: If CLI options, path syntax keywords, or core behavior changed, update README.md and SPEC.md accordingly.

[ ] Clear Commit Messages: Use concise and descriptive commit messages summarizing the changes.

[ ] Bilingual Consistency: Ensure any user-facing keywords or error messages maintain symmetry in both English and Arabic where applicable.

6. Licensing & AI Disclosure

License: By contributing to tawjih, you agree that your contributions will be licensed under the project's open-source license (see LICENSE).

Generative AI Transparency: If you use AI assistants (such as Claude, ChatGPT, or Copilot) to generate or refine code/documentation, please state this transparently in your Pull Request description and maintain full human oversight over correctness and test coverage
