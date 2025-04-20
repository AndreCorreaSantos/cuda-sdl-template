#ifndef HELPERS_H
#define HELPERS_H

#include <SDL2/SDL.h>

typedef struct
{
    float x,y,value;
} Agent;

// Structure to hold SDL2 objects
struct SDL2Context {
    SDL_Window* window;
    SDL_Renderer* renderer;
    SDL_Texture* texture;
};

// Initialize SDL2, window, renderer, and texture
bool initSDL2(SDL2Context& context, int width, int height);

// Cleanup SDL2 resources
void cleanupSDL2(SDL2Context& context);

// Initialize CUDA device buffer
bool initCUDA(unsigned char** d_buffer, int width, int height);

// Cleanup CUDA resources
void cleanupCUDA(unsigned char* d_buffer);


#endif // HELPERS_H