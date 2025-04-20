Thanks for the clarification! Here’s the updated README with the correct filename:

---

# CUDA + SDL2 Template

This project provides a simple template for running CUDA code and displaying its output in an SDL2-generated window and texture. It is designed to be easy to use and can be run with a single command: `./compile.sh`. The template includes boilerplate code for initializing CUDA and SDL2, and it can be used to quickly prototype CUDA applications with graphical output.

## Prerequisites

Before building and running this project, make sure you have the following dependencies installed:

### System Requirements
- **CUDA** (Tested with CUDA 11.8, but should work with earlier versions)
- **GCC** (Tested with GCC 9.5, should work with older versions)
- **SDL2** (For window and texture management)

### Dependencies
1. **CUDA Toolkit**: Install the CUDA toolkit corresponding to your system and GPU. For this template, CUDA 11.8 is tested.
2. **SDL2**: SDL2 is required for creating windows and rendering textures.

On a **Ubuntu**-based system, you can install SDL2 with the following commands:
```bash
sudo apt update
sudo apt install libsdl2-dev
```

3. **GCC 9.5 or compatible**: The project has been tested with GCC 9.5. Make sure to have an appropriate version installed. You can check your GCC version with:
```bash
gcc --version
```

## Building the Project

To build the project, simply run the `compile.sh` script:

```bash
./compile.sh
```

This script will:
- Compile the CUDA source code.
- Link the CUDA code with the SDL2 library.
- Build the executable.

Once the build process completes successfully, the compiled executable will be generated in the `./build` directory.

### Troubleshooting

If you encounter any issues during compilation, make sure the following:
- CUDA is correctly installed and the `nvcc` compiler is available in your PATH.
- You have the appropriate version of GCC installed.

## Running the Project

Once the project is built, you can run the executable:

```bash
./build/main
```

This will open an SDL2 window, run the CUDA code, and display the output on the window.

## Code Structure

The code is organized into the following files:

- `helpers.h` / `helpers.cuh`: Header and CUDA kernel functions for device-side operations.
- `main.cu`: Main file containing the CUDA code and SDL2 window management.
- `compile.sh`: A shell script that automates the build process.

## Features

- **CUDA Kernel**: The template includes a simple CUDA kernel that sets pixel values in a buffer.
- **SDL2 Integration**: SDL2 is used to manage a window and texture, and the CUDA output is displayed on the texture.
- **Flexible Template**: The project is designed to be a starting point for more complex CUDA-based graphics projects.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
