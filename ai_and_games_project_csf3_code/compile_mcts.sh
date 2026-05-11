#!/bin/bash

# Ensure the directory exists
mkdir -p src/native

# Compile
g++ -O3 -Wall -shared -std=c++17 -fPIC -fopenmp src/native/mcts_hex.cpp -o src/native/libmcts_hex.so

echo "Compilation finished."