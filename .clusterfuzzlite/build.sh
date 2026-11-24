#!/bin/bash -eu
# One fuzzer per fuzz/fuzz_*.py; each fuzzes a target in fuzz/targets.py.
cd "$SRC/tawjih"
for fuzzer in fuzz/fuzz_*.py; do
  compile_python_fuzzer "$fuzzer" --paths "$SRC/tawjih" --paths "$SRC/tawjih/fuzz"
done
