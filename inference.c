#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "inference.h"
#include "stb_image_resize.h"
#include "stb_image.h"
#include "onnxruntime_c_api.h"

#define MAX_DETECTIONS 100

// Initialization
void inference_init(Inference* inf, const char* modelPath, int w, int h,int useCuda){
    
    // Basic configuration 
    inf->modelPath = strdup(modelPath);
    inf->inputWidth = w;
    inf->inputHeight = h;
    inf->cudaEnabled = useCuda;

    // Get teh ONXX Runtime API 
    inf->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    const OrtApi* api = inf->api;

    // Create runtime environment
    api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "YOLOv8_C", &inf->env);

    // Create session options
    api->CreateSessionOptions(&inf->sessionOptions);

    // Load the ONXX model
    api->CreateSession(inf->env, inf->modelPath, inf->sessionOptions, &inf->session);

}

// Convert an image into CHW normalized tensor 
float* preprocess_image(unsigned char* img, int imgW, int imgH, int targetW, 
                                 int targetH, int* squareSize){
    
    // Determine the size of the square (select the largest side to keep proportions)
    int maxSize = imgW > imgH ? imgW : imgH;
    *squareSize = maxSize;

    // Create a black square image (padding)
    unsigned char* square = (unsigned char*)calloc(maxSize * maxSize * 3, 1);
    if (!square) return NULL;

    // Copy the original image at the upper-left corner        
    for(int y = 0; y < imgH; y++)
        memcpy(square + (y * maxSize * 3), img + (y * imgW * 3), imgW * 3);
    

    // Resize to the input model's shape
    unsigned char* resized = (unsigned char*)malloc(targetW * targetH * 3);
    if (!resized) {
        free(square);
        return NULL;
    }

    stbir_resize_uint8(square, maxSize, maxSize, 0,         // Input image
                       resized, targetW, targetH, 0, 3);    // Output image
    
    // Release resources from the image with padding before resizing    
    free(square);

    // Convert to tensor CHW [channel, height, width]
    float* tensor = (float*)malloc(3 * targetW * targetH * sizeof(float));
    if (!tensor) {
        free(resized);
        return NULL;
    }

    // RGB channels
    for(int c = 0; c < 3; c++){
        // Rows
        for(int h = 0; h < targetH; h++){
            // Columns
            for(int w = 0; w < targetW; w++){
                int idx = (h * targetW + w) * 3; // index in HWC
                // Normalized tensor in CHW (ONXX format)
                tensor[c * targetW * targetH + h * targetW + w] = resized[idx + c] / 255.0f;
            }
        }
    }

    // Release resources from residez image 
    free(resized);

    return tensor;
}

// Run inference in ONXX model 
void run_inference(Inference* inf, unsigned char* img, int imgW, int imgH, 
                   Detection* outputDets, int* nDetections){
            
    // Preprocess 
    int squareSize; 
    float* inputTensor = preprocess_image(img, imgW, imgH, inf->inputWidth, inf->inputHeight, &squareSize);
    // Security check
    if(!inputTensor){
        *nDetections = 0;
        return;
    }

    // Get teh ONXX Runtime API 
    const OrtApi* api = inf->api;
    // Define dimenssions of the input tensor [batch, channels, height, width]
    int64_t dims[4] = {1, 3, inf->inputHeight, inf->inputWidth};
    size_t tensor_size = 1 * 3 * inf->inputHeight * inf->inputWidth;

    OrtMemoryInfo* memInfo = NULL;
    OrtValue* inputOrt = NULL;
    OrtValue* outputOrt = NULL;

    // Create memory in CPU for the input tensor
    api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memInfo);
    api->CreateTensorWithDataAsOrtValue(memInfo, inputTensor, tensor_size*sizeof(float),
                                        dims, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputOrt);
    
    // Names of the input/output nodes to the net
    const char* inputNames[] = {"images"};
    const char* outputNames[] = {"output0"};

    // Inference 
    OrtStatus* status = NULL;
    status = api->Run(inf->session, NULL, inputNames, (const OrtValue* const*)&inputOrt,
             1, outputNames, 1, &outputOrt);
    
    if (status != NULL){
        printf("ONNX Runtime error: %s\n", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        *nDetections = 0;
        goto cleanup;
    }
       
    // Obtain a pointer to the output data
    float* outData = NULL;
    api->GetTensorMutableData(outputOrt, (void**)&outData);

    // Obtain dimenssions of the output tensor
    OrtTensorTypeAndShapeInfo* shapeInfo = NULL;
    api->GetTensorTypeAndShape(outputOrt, &shapeInfo);

    int64_t outDims[3];
    api->GetDimensions(shapeInfo, outDims, 3);

    api->ReleaseTensorTypeAndShapeInfo(shapeInfo);

    // Escalate the output coordenates to the original size
    float x_factor = (float)squareSize / inf->inputWidth;
    float y_factor = (float)squareSize / inf->inputHeight;

    // Total number of predictions 
    int nPred = outDims[2];    
    
    // Number of parameters per prediciton (x, y, w, h, score_class0, ..., score_classN-1)
    int nChannels = outDims[1]; // 4 + NUM_CLASSES

    // Process every prediction
    float* data = outData;
    int count = 0;

    for(int i = 0; i < nPred; i++){

        float x = data[0 * nPred + i];
        float y = data[1 * nPred + i];
        float w = data[2 * nPred + i];
        float h = data[3 * nPred + i];

        // Find the class with the highest score for this prediction (argmax)
        int bestClass = 0;
        float bestScore = data[(4 + 0) * nPred + i];
        for (int c = 1; c < NUM_CLASSES; c++){
            float score = data[(4 + c) * nPred + i];
            if (score > bestScore){
                bestScore = score;
                bestClass = c;
            }
        }

        // Filter poor predictions 
        if(bestScore < 0.25f)
            continue;

        float left   = (x - 0.5f*w) * x_factor;
        float top    = (y - 0.5f*h) * y_factor;
        float width  = w * x_factor;
        float height = h * y_factor;

        // Save detection
        outputDets[count].x = left;
        outputDets[count].y = top;
        outputDets[count].w = width;
        outputDets[count].h = height;
        outputDets[count].confidence = bestScore;
        outputDets[count].class_id = bestClass;

        strcpy(outputDets[count].className, CLASS_NAMES[bestClass]);
        outputDets[count].r = CLASS_COLORS[bestClass][0];
        outputDets[count].g = CLASS_COLORS[bestClass][1];
        outputDets[count].b = CLASS_COLORS[bestClass][2];

        count++;
        if(count >= MAX_DETECTIONS)
            break;
    }

    // Apply NMS to remove duplicated boxes 
    int keep[MAX_DETECTIONS];
    int keepCount = 0;

    nms(outputDets, count, 0.7f, keep, &keepCount);

    Detection tmp[MAX_DETECTIONS];
    for(int i = 0; i < keepCount; i++)
        tmp[i] = outputDets[keep[i]];

    for(int i = 0; i < keepCount; i++)
        outputDets[i] = tmp[i];

    *nDetections = keepCount;

    goto cleanup;
    // Release resources 
    cleanup:
        api->ReleaseValue(inputOrt);
        api->ReleaseValue(outputOrt);
        api->ReleaseMemoryInfo(memInfo);
        free(inputTensor);
}

// Remove overlapping boxes keeping the best one
void nms(Detection* dets, int n, float iouThreshold, int* keep, int* keepCount){
    
    // Sort detections by confidence (descending)
    for (int i = 0; i < n - 1; i++){
        for (int j = i + 1; j < n; j++){
            if (dets[j].confidence > dets[i].confidence){
                Detection tmp = dets[i];
                dets[i] = dets[j];
                dets[j] = tmp;
            }
        }
    }

    // Create an array to register removed detections 
    int* suppressed = (int*)calloc(n, sizeof(int));
    if (!suppressed){
        *keepCount = 0;
        return;
    }

    *keepCount = 0;

    // Iterate on each detection
    for (int i = 0; i < n; i++){

        // Ignore if it has been already removed
        if (suppressed[i])
            continue;

        // Otherwhise, keep the detection
        keep[(*keepCount)++] = i;

        // Security check
        if (*keepCount >= MAX_DETECTIONS)
            break;

        // Compare with following detections
        for (int j = i + 1; j < n; j++){

            if (suppressed[j])
                continue;

            // Compute intersection between two boxes
            float xx1 = fmaxf(dets[i].x, dets[j].x);
            float yy1 = fmaxf(dets[i].y, dets[j].y);
            float xx2 = fminf(dets[i].x + dets[i].w,
                              dets[j].x + dets[j].w);
            float yy2 = fminf(dets[i].y + dets[i].h,
                              dets[j].y + dets[j].h);
            
            // Intersection width and height 
            float w = fmaxf(0.0f, xx2 - xx1);
            float h = fmaxf(0.0f, yy2 - yy1);

            float inter = w * h;
            
            // COmpute infividual areas
            float area_i = dets[i].w * dets[i].h;
            float area_j = dets[j].w * dets[j].h;
            
            // Union area
            float union_ = area_i + area_j - inter;

            // Compute IoU
            if (union_ > 0.0f){
                float iou = inter / union_;

                // If the overlap is larger than a threshold AND it's the same class,
                // the boxes are detecting the same object
                if (iou > iouThreshold && dets[i].class_id == dets[j].class_id)

                    // Remove the box with less confidence
                    suppressed[j] = 1;
            }
        }
    }

    //  Release resources
    free(suppressed);
}

// Release resources associated with inference
void inference_free(Inference* inf){

    // Security check
    if (!inf){printf("Error: Releasing memory\n"); return;}

    // Get ONNX Runtime API
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    // Release ONNX session
    if (inf->session){
        api->ReleaseSession(inf->session);
        inf->session = NULL;  
    }

    // Release session options
    if (inf->sessionOptions) {
        api->ReleaseSessionOptions(inf->sessionOptions);
        inf->sessionOptions = NULL;
    }

    // Release ONNX environment
    if (inf->env) {
        api->ReleaseEnv(inf->env);
        inf->env = NULL;
    }

    // Release resources from the string of the model path 
    if (inf->modelPath) {
        free(inf->modelPath);
        inf->modelPath = NULL;
    }
}