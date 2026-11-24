#!/bin/bash -eu
# One fuzzer per fuzz/fuzz_*.py; each fuzzes a target in fuzz/targets.py.
cd "$SRC/tawjih"
# tawjihlib loads its modules by name, which PyInstaller cannot see: name
# each one as a hidden import
hidden=""
for f in tawjihlib/*.py; do
  m=$(basename "$f" .py)
  [ "$m" = "__init__" ] || hidden="$hidden --hidden-import tawjihlib.$m"
done
for fuzzer in fuzz/fuzz_*.py; do
  # shellcheck disable=SC2086
  compile_python_fuzzer "$fuzzer" --paths "$SRC/tawjih" \
    --paths "$SRC/tawjih/fuzz" $hidden
done
