"""Atheris entry point for the fuzz target path_language (see targets.py)."""
import sys

import atheris

with atheris.instrument_imports():
    import targets


def main():
    atheris.Setup(sys.argv, targets.TARGETS["path_language"])
    atheris.Fuzz()


if __name__ == "__main__":
    main()
