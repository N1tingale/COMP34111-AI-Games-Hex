#!/bin/bash

# 1. AVX-512 Flags are REQUIRED for the new code
# -mavx512f -mavx512dq -mfma: Enable the specific instruction sets used
# -march=native: Optimizes for the A100 processor specifically

# 2. Check filenames
# Input: mcts_hex_optimised.cpp (The file I provided)
# Output: libmcts_hex.so (In the current directory, matching the Python script)

g++ -O3 -Wall -shared -std=c++17 -fPIC -fopenmp \
    -mavx512f -mavx512dq -mfma -march=native \
    src/native/mcts_hex_optimised.cpp -o src/native/libmcts_hex_optimised.so

echo "Compilation finished. Created src/native/libmcts_hex_optimised.so"