#!/bin/bash

# Create build directory if it doesn't exist
mkdir -p build

# Compile helpers.cpp to object file
g++ -c src/helpers.cpp -o build/helpers.o -Iinclude -I/usr/include/SDL2

# Compile and link multi_agents.cu with helpers.o
nvcc -o build/multi_agents src/multi_agents.cu build/helpers.o -Iinclude -I/usr/include/SDL2 -L/usr/lib -lSDL2

# Run the program
./build/multi_agents
