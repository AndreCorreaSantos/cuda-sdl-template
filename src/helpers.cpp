#include "helpers.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream> 


// ------------------------------------ SETUP ---------------------------------------------------------------
bool initSDL2(SDL2Context& context, int width, int height) {
    // Initialize SDL2
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
        return false;
    }

    // Create window
    context.window = SDL_CreateWindow(
        "CUDA+SDL2",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        width, height,
        SDL_WINDOW_SHOWN
    );
    if (!context.window) {
        std::cerr << "SDL_CreateWindow failed: " << SDL_GetError() << "\n";
        SDL_Quit();
        return false;
    }

    // Create renderer
    context.renderer = SDL_CreateRenderer(context.window, -1, SDL_RENDERER_ACCELERATED);
    if (!context.renderer) {
        std::cerr << "SDL_CreateRenderer failed: " << SDL_GetError() << "\n";
        SDL_DestroyWindow(context.window);
        SDL_Quit();
        return false;
    }

    // Create texture
    context.texture = SDL_CreateTexture(
        context.renderer,
        SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_STREAMING,
        width, height
    );
    Uint32 format;
    SDL_QueryTexture(context.texture, &format, nullptr, nullptr, nullptr);
    std::cout << "SDL texture format: " << SDL_GetPixelFormatName(format) << std::endl;

    if (!context.texture) {
        std::cerr << "SDL_CreateTexture failed: " << SDL_GetError() << "\n";
        SDL_DestroyRenderer(context.renderer);
        SDL_DestroyWindow(context.window);
        SDL_Quit();
        return false;
    }

    return true;
}

void cleanupSDL2(SDL2Context& context) {
    if (context.texture) SDL_DestroyTexture(context.texture);
    if (context.renderer) SDL_DestroyRenderer(context.renderer);
    if (context.window) SDL_DestroyWindow(context.window);
    SDL_Quit();
}


// -----------------------------------------------------------------------------------------------------

