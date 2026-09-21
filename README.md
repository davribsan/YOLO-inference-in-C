# YOLO in C

C inference of a pretrained YOLOv8n model for UAV detection (two extra classes included in case it is used for multiclass detection).

This project is part of a Master's Thesis.

## Description

The program loads a pretrained YOLOv8n model exported in **ONNX** format, runs inference on an image using **ONNX Runtime** (C API), and displays the result on screen using **SDL2**, drawing the bounding box for each detection on top of the image, together with its label and confidence score.

The model currently detects 3 classes:

| class_id | Class   | On-screen colour |
|----------|---------|--------------------|
| 0        | UAV     | Red                |
| 1        | Plane   | Blue               |
| 2        | Bird    | Green              |

> The `Plane` and `Bird` classes are included in the code to support a future multiclass retraining. However, the provided model only detects UAVs.

## Results

![Detection result](Images/Expected%20results/example%201%20result.png)

Example output of the programme: the image is shown in a window with bounding boxes, class labels and confidence scores overlaid on each detection.

## Project structure

```
.
├── main.c                          # Entry point: loads the image, runs inference and displays the result
├── inference.c / inference.h       # Inference logic using ONNX Runtime (preprocessing, forward pass, NMS)
├── stb_image.h                     # Image loading (single-header, stb)
├── stb_image_resize.h              # Image resizing (single-header, stb)
├── stb_image_resize2.h             # Version 2 of stb_image_resize (as used in the project)
├── stb_truetype.h                  # Text rendering (labels and scores) with no external dependencies
├── onnxruntime-osx-x86_64-1.11.1/  # ONNX Runtime distribution (headers + libraries)
├── Images/
│   ├── Test/                       # Samples of images for trying out the programme
│       └── example 1.jpg
│       └── example 2.jpg
│       └── example 3.jpg
│       └── example 4.jpg
│   ├── Expected results            # Screenshots of the expected results after inference (approx.) 
│       └── example 1 result.jpg
│       └── example 2 result.jpg
│       └── example 3 result.jpg
└──     └── example 4 result.jpg
```

## Requirements

Tested exclusively on:

- **macOS 12.7.6** (not tested on Linux or Windows)
- **CPU** only (GPU/CUDA execution has not been tested, although the `runOnGPU` flag exists in the code)

### Dependencies

| Dependency     | Version used | Installation                          |
|----------------|---------------|----------------------------------------|
| C compiler     | clang (Xcode Command Line Tools) | `xcode-select --install` |
| SDL2           | 2.32.10       | `brew install sdl2`                  |
| ONNX Runtime   | 1.11.1        | Included in `onnxruntime-osx-x86_64-1.11.1/` (manually downloaded from the corresponding release) |
| stb_image / stb_image_resize / stb_image_resize2 / stb_truetype | single-header, no formal version | Included directly in the repository |

There is no need to install `SDL2_ttf` or any other font library: text rendering is handled with `stb_truetype.h`, which has no external dependencies.

### Model

The programme **does not train or export models**. It assumes a pretrained YOLOv8 model **exported in ONNX format** (e.g. `model_size_640_batch_32.onnx`), obtained beforehand with Ultralytics/YOLO and the corresponding ONNX export command.

## Building

Adjust the include/lib paths according to where SDL2 and ONNX Runtime are installed on your system, then build with:

```bash
clang "main.c" "inference.c" \
  -I/usr/local/Cellar/sdl2/2.32.10/include \
  -I"/Users/usuario/Downloads/onnxruntime-osx-x86_64-1.11.1/include" \
  -L/usr/local/Cellar/sdl2/2.32.10/lib \
  -L"/Users/usuario/Downloads/onnxruntime-osx-x86_64-1.11.1/lib" \
  -lSDL2 -lonnxruntime \
  -o "main" \
  -Wl,-rpath,"/Users/usuario/Downloads/onnxruntime-osx-x86_64-1.11.1/lib"
```

## Usage

The model and image paths **are edited directly in the code**, inside `main.c`:

```c
// Image
const char* imgPath = "images/test.jpg"; // sample image included in the repository

// Model
const char* modelPath = "/path/to/your/model.onnx";
int imageSize = 640;      // model input size
int runOnGPU = 0;         // 0 = CPU, 1 = GPU (untested)
```

Once built, run:

```bash
./main
```

A window will open showing the image with bounding boxes, class labels and confidence scores for each detected object. Press **ESC** or close the window to exit.

## Notes

- The order of `CLASS_NAMES` in `inference.h` must exactly match the class order used when training the model (typically defined in YOLO's `data.yaml`). If it does not match, the labels shown will not correspond to the actual detected class.
- The minimum confidence threshold for a detection to be shown is set to `0.25` inside `run_inference` (`inference.c`).
- The Non-Maximum Suppression (NMS) used has an IoU threshold of `0.7` and only suppresses overlapping boxes that belong to the same class.

## Licence

This project does not include an explicit licence. All rights are reserved by the author, as part of a Master's Thesis.
