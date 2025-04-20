#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build

# Compile helpers.cpp to object file
g++ -c src/helpers.cpp -o build/helpers.o -Iinclude -I/usr/include/SDL2

# Compile and link main.cu with helpers.o
nvcc -o build/main src/main.cu build/helpers.o -Iinclude -I/usr/include/SDL2 -L/usr/lib -lSDL2

# Run the program
./build/main
