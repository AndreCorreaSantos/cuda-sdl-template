#ifndef HELPERS_CUH
#define HELPERS_CUH
#include <cuda_runtime.h>





void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " (" << cudaGetErrorString(err) << ")\n";
        exit(-1);
    }
}

bool initCUDA(unsigned char** d_buffer, int width, int height) {
    // allocate device buffer
    cudaError_t err = cudaMalloc(d_buffer, width * height * 4);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: cudaMalloc (" << cudaGetErrorString(err) << ")\n";
        return false;
    }
    return true;
}

void cleanupCUDA(unsigned char* d_buffer) {
    if (d_buffer) {
        checkCuda(cudaFree(d_buffer), "cudaFree");
    }
}

#endif // HELPERS_CUH