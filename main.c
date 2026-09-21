#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_TRUETYPE_IMPLEMENTATION

#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stb_image.h"
#include "stb_image_resize.h"
#include "stb_truetype.h"
#include "inference.h"

#define ATLAS_SIZE 512      // size of the bitmap where the characters are baked
#define FIRST_CHAR 32       // first baked ASCII character (space)
#define NUM_CHARS  96       // baked characters: 32..127 (printable ASCII)

// Loads a whole .ttf file into memory. Returns NULL on failure (ONLY NECESSARY FOR WRITTING THE LABEL)
static unsigned char* loadFontFile(const char* path, long* outSize){
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    unsigned char* buffer = (unsigned char*)malloc(size);
    if (!buffer){ fclose(f); return NULL; }

    fread(buffer, 1, size, f);
    fclose(f);

    if (outSize) *outSize = size;
    return buffer;
}

int main(int argc, char** argv){

    // 1. INPUTS

        // Image
        const char* imgPath = "Your image path";     // image path

        // Model
        const char* modelPath = "Your model path";   // model path
        int imageSize = 640;                         // imageSize of the model                                                              
        int runOnGPU = 0;

        const char* fontPath = "/System/Library/Fonts/Supplemental/Arial.ttf";
        float fontPixelHeight = 18.0f;

    //-----------------------------------------------------------------------------------------------------------------

    // 2. INITIALIZE THE INFERENCE STRUCTURE
    Inference inf;
    inference_init(&inf, modelPath, imageSize, imageSize, runOnGPU);

    //-----------------------------------------------------------------------------------------------------------------

    // 3. LOAD IMAGE
    int imgW, imgH, channels;
    unsigned char* img = stbi_load(imgPath, &imgW, &imgH, &channels, 3);
    if (!img){
        printf("Error loading image: %s\n", imgPath);
        inference_free(&inf);
        return 1;
    }

    //-----------------------------------------------------------------------------------------------------------------

    // 4. INFERENCE
    Detection detections[MAX_DETECTIONS];
    int nDetections = 0;

    run_inference(&inf, img, imgW, imgH, detections, &nDetections);

    printf("Number of detections: %d\n", nDetections);

    //-----------------------------------------------------------------------------------------------------------------

    // 5. PREPARE THE FONT (stb_truetype, no external dependencies)

    long fontFileSize;
    unsigned char* fontBuffer = loadFontFile(fontPath, &fontFileSize);
    if (!fontBuffer){
        printf("Error loading font: %s\n", fontPath);
        stbi_image_free(img);
        inference_free(&inf);
        return 1;
    }

    // Bitmap where all the characters are baked (rasterized) at once
    unsigned char* fontAtlasAlpha = (unsigned char*)malloc(ATLAS_SIZE * ATLAS_SIZE);
    stbtt_bakedchar charData[NUM_CHARS];

    int bakeResult = stbtt_BakeFontBitmap(fontBuffer, 0, fontPixelHeight,
                                           fontAtlasAlpha, ATLAS_SIZE, ATLAS_SIZE,
                                           FIRST_CHAR, NUM_CHARS, charData);
    if (bakeResult <= 0){
        printf("Error: font does not fit in the atlas (bakeResult=%d)\n", bakeResult);
    }

    //-----------------------------------------------------------------------------------------------------------------

    // 6. SHOW IMAGE WITH DETECTIONS (SDL2)

    if (SDL_Init(SDL_INIT_VIDEO) != 0){
        printf("Error initializing SDL2: %s\n", SDL_GetError());
        stbi_image_free(img);
        inference_free(&inf);
        free(fontBuffer);
        free(fontAtlasAlpha);
        return 1;
    }

    // Get the screen shape
    SDL_DisplayMode dm;
    SDL_GetCurrentDisplayMode(0, &dm);

    // Calculate the scale keeping the aspect ratio
    float scaleX = (float)dm.w / imgW;
    float scaleY = (float)dm.h / imgH;
    float scale  = (scaleX < scaleY ? scaleX : scaleY) * 0.9f;

    int winW = (int)(imgW * scale);
    int winH = (int)(imgH * scale);

    SDL_Window* window = SDL_CreateWindow(
        imgPath,
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        winW,
        winH,
        0
    );

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    // Image texture
    SDL_Texture* texture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGB24,
        SDL_TEXTUREACCESS_STATIC,
        imgW,
        imgH
    );
    SDL_UpdateTexture(texture, NULL, img, imgW * 3);

    // Convert the font atlas (1 channel, alpha only) to RGBA
    // so it can be created as an SDL texture and tinted with any color.
    unsigned char* fontAtlasRGBA = (unsigned char*)malloc(ATLAS_SIZE * ATLAS_SIZE * 4);
    for (int p = 0; p < ATLAS_SIZE * ATLAS_SIZE; p++){
        fontAtlasRGBA[p * 4 + 0] = 255;                 // R (white; tinted with SDL_SetTextureColorMod)
        fontAtlasRGBA[p * 4 + 1] = 255;                 // G
        fontAtlasRGBA[p * 4 + 2] = 255;                 // B
        fontAtlasRGBA[p * 4 + 3] = fontAtlasAlpha[p];   // A: the shape of the letter
    }

    SDL_Texture* fontTexture = SDL_CreateTexture(
        renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STATIC,
        ATLAS_SIZE,
        ATLAS_SIZE
    );
    SDL_UpdateTexture(fontTexture, NULL, fontAtlasRGBA, ATLAS_SIZE * 4);
    SDL_SetTextureBlendMode(fontTexture, SDL_BLENDMODE_BLEND);

    free(fontAtlasRGBA);
    free(fontAtlasAlpha);
    free(fontBuffer);

    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    SDL_Event e;
    int quit = 0;

    while (!quit){
        while (SDL_PollEvent(&e)){
            if (e.type == SDL_QUIT) quit = 1;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) quit = 1;
        }

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);

        for (int i = 0; i < nDetections; i++){
            Detection* d = &detections[i];

            SDL_Rect rect = {
                (int)(d->x * scale),
                (int)(d->y * scale),
                (int)(d->w * scale),
                (int)(d->h * scale)
            };

            SDL_SetRenderDrawColor(renderer, d->r, d->g, d->b, 255);
            SDL_RenderDrawRect(renderer, &rect);

            // Text (label + score)
            char label[64];
            snprintf(label, sizeof(label), "%s %.2f", d->className, d->confidence);

            // Compute the total text size to position the background rectangle
            float textX = 0.0f, textY = 0.0f;
            float minX = 1e9f, minY = 1e9f, maxX = -1e9f, maxY = -1e9f;

            for (const char* c = label; *c; c++){
                if (*c < FIRST_CHAR || *c >= FIRST_CHAR + NUM_CHARS) continue;

                stbtt_aligned_quad q;
                stbtt_GetBakedQuad(charData, ATLAS_SIZE, ATLAS_SIZE,
                                    *c - FIRST_CHAR, &textX, &textY, &q, 1);

                if (q.y0 < minY) minY = q.y0;
                if (q.y1 > maxY) maxY = q.y1;
                if (q.x0 < minX) minX = q.x0;
                if (q.x1 > maxX) maxX = q.x1;
            }

            float textHeight = maxY - minY;
            float textWidth  = maxX - minX;

            // Text position: right above the box (or below if it doesn't fit)
            float baseX = rect.x;
            float baseY = rect.y - textHeight - 4;
            if (baseY < 0) baseY = rect.y + 4;

            // Semi-transparent background for readability
            SDL_Rect bgRect = { (int)baseX - 1, (int)baseY - 1,
                                 (int)textWidth + 2, (int)textHeight + 2 };
            SDL_SetRenderDrawColor(renderer, d->r, d->g, d->b, 255);
            SDL_RenderFillRect(renderer, &bgRect);

            // Draw each character using the baked atlas
            SDL_SetTextureColorMod(fontTexture, 255, 255, 255);
            
            textX = baseX;
            textY = baseY - minY; // adjust baseline so the text starts at baseY

            for (const char* c = label; *c; c++){
                if (*c < FIRST_CHAR || *c >= FIRST_CHAR + NUM_CHARS) continue;

                stbtt_aligned_quad q;
                stbtt_GetBakedQuad(charData, ATLAS_SIZE, ATLAS_SIZE,
                                    *c - FIRST_CHAR, &textX, &textY, &q, 1);

                SDL_Rect srcRect = {
                    (int)(q.s0 * ATLAS_SIZE), (int)(q.t0 * ATLAS_SIZE),
                    (int)((q.s1 - q.s0) * ATLAS_SIZE), (int)((q.t1 - q.t0) * ATLAS_SIZE)
                };
                SDL_Rect dstRect = {
                    (int)q.x0, (int)q.y0,
                    (int)(q.x1 - q.x0), (int)(q.y1 - q.y0)
                };

                SDL_RenderCopy(renderer, fontTexture, &srcRect, &dstRect);
            }
        }

        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    //-----------------------------------------------------------------------------------------------------------------

    // 7. RELEASE MEMORY
    SDL_DestroyTexture(fontTexture);
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    stbi_image_free(img);
    inference_free(&inf);

    return 0;
}

/*clang "main.c" "inference.c"  -I/usr/local/Cellar/sdl2/2.32.10/include   -I"/Users/usuario/Downloads/onnxruntime-osx-x86_64-1.11.1/include"   -L/usr/local/Cellar/sdl2/2.32.10/lib   -L"/Users/usuario/Downloads/onnxruntime-osx-x86_64-1.11.1/lib"   -lSDL2 -lonnxruntime   -o "main"   -Wl,-rpath,"/Users/usuario/Downloads/onnxruntime-osx-x86_64-1.11.1/lib"
*/
