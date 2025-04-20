#include "helpers.h"
#include <SDL2/SDL_events.h>
#include <cstdlib>
#include <cuda_device_runtime_api.h> 
#include <driver_types.h>
#include <iostream>
#include <cuda_runtime.h>
#include "helpers.cuh"


#define WIDTH 1920
#define HEIGHT 1080

__global__ void drawTest(unsigned char *d_buffer, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= WIDTH * HEIGHT) return;

    int byteIndex = idx * 4;

    d_buffer[byteIndex + 0] = 255; // A --> for blending
    d_buffer[byteIndex + 1] = 0;   // B
    d_buffer[byteIndex + 2] = 128; // G
    d_buffer[byteIndex + 3] = 0;   // R

}




int main() {
    // init SDL2 context
    SDL2Context sdlContext = {nullptr, nullptr, nullptr};
    if (!initSDL2(sdlContext,WIDTH,HEIGHT)) {
        return -1;
    }

    // Allocate device buffer for pixel data
    unsigned char* d_buffer = nullptr;
    checkCuda(cudaMalloc(&d_buffer, WIDTH * HEIGHT * 4), "cudaMalloc");

    // main loop
    bool running = true;
    bool mouseDown = false;
    SDL_Event event;
    int pixelX = WIDTH / 2;
    int pixelY = HEIGHT / 2;

    while (running) {
        // Handle events
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_MOUSEBUTTONDOWN)
            {
                mouseDown = true;
            }
            if (event.type == SDL_MOUSEBUTTONUP)
            {
                mouseDown = false;
            }
            if (event.type == SDL_QUIT) {
                running = false;
            }

        }

        float t = SDL_GetTicks() / 1000.0f;  // time in seconds as float
        // float t = 1.0;
        int blockSize = 256; // Good default
        int nPixels = WIDTH*HEIGHT;
        int numBlocks = (nPixels + blockSize - 1) / blockSize;


        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaDeviceSynchronize(),"kernel sync");
        
        drawTest<<<numBlocks, blockSize>>>(d_buffer, WIDTH, HEIGHT);

        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaDeviceSynchronize(), "kernel sync");
        // lock texture and copy CUDA buffer to texture
        void* pixels = nullptr;
        int pitch;
        if (SDL_LockTexture(sdlContext.texture, nullptr, &pixels, &pitch) != 0) {
            std::cerr << "SDL_LockTexture failed: " << SDL_GetError() << "\n";
            running = false;
        } else {
            checkCuda(cudaMemcpy(pixels, d_buffer, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost), "cudaMemcpy to texture");
            SDL_UnlockTexture(sdlContext.texture);
        }

        // sdl render
        SDL_RenderClear(sdlContext.renderer);
        SDL_RenderCopy(sdlContext.renderer, sdlContext.texture, nullptr, nullptr);
        SDL_RenderPresent(sdlContext.renderer);
    }

    // cleanup
    checkCuda(cudaFree(d_buffer), "cudaFree");
    cleanupSDL2(sdlContext);
    return 0;
}