"""Atheris entry point for the fuzz target examples (see targets.py)."""
import sys

import atheris

with atheris.instrument_imports():
    import targets


def main():
    atheris.Setup(sys.argv, targets.TARGETS["examples"])
    atheris.Fuzz()


if __name__ == "__main__":
    main()
