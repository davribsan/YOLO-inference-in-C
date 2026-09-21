#ifndef INFERENCE_H
#define INFERENCE_H

#include "onnxruntime_c_api.h"

#define MAX_DETECTIONS 100

// Number of classes the model detects
#define NUM_CLASSES 3

// Class names, indexed by class_id (must match the training order of the model!)
static const char* CLASS_NAMES[NUM_CLASSES] = {"UAV", "Plane", "Bird"};

// One display color per class (R, G, B), indexed by class_id
static const unsigned char CLASS_COLORS[NUM_CLASSES][3] = {
    {255, 0,   0},   // UAV   -> red
    {0,   128, 255}, // Plane -> blue
    {0,   200, 0}    // Bird  -> green
};

typedef struct {
    float x, y, w, h;
    float confidence;
    int class_id;
    char className[10]; // For UAV only 4 are necessary but 10 in case it is used for different objects
    unsigned char r, g, b;
} Detection;

typedef struct {
    char* modelPath;
    int inputWidth;
    int inputHeight;
    int cudaEnabled;
    const OrtApi* api; 
    OrtEnv* env;
    OrtSession* session;
    OrtSessionOptions* sessionOptions;
} Inference;

// Initialization
void inference_init(Inference* inf, const char* modelPath, int w, int h,int useCuda);
                                
// Convert an image into CHW normalized tensor 
float* preprocess_image(unsigned char* img, int imgW, int imgH, int targetW, 
                                 int targetH, int* squareSize);

// Inference
void run_inference(Inference* inf, unsigned char* img, int imgW, int imgH,
                   Detection* outputDets, int* nDetections);

// NMS
void nms(Detection* dets, int n, float iouThreshold, int* keep, int* keepCount);

// Release resources
void inference_free(Inference* inf);

#endif