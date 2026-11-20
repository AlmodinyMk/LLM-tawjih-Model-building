#!/usr/bin/env python3
"""tawjih - personal LLM command-directing (توجيه أوامر الآلة).

A project is a folder of three-sided files arranged as trees:
    human side   : the expressive human concept (context)
    machine side : instructions for the model + optional Python code
    reading side : filtered passages of deposited books and research
A path connects files across trees into a directed acyclic workflow (DAG);
running it executes their machine sides step by step.
The human and reading sides are build-time material and never reach the model
at run time.

The program lives in the package tawjihlib/ next to this file; this file is
how it is run (python tawjih.py ...) and imported (import tawjih), with every
name of the package available as tawjih.NAME.

Standard library only. Python 3.11+.

Copyright (c) 2026 Mohammed Khalid Almodiny
SPDX-License-Identifier: CC-BY-NC-SA-4.0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tawjihlib  # noqa: E402
from tawjihlib.cli import main  # noqa: E402

globals().update({k: v for k, v in vars(tawjihlib).items()
                  if not k.startswith("__") or k == "__version__"})

if __name__ == "__main__":
    main()
