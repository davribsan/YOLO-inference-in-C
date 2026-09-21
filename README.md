# YOLO-inference-in-C
The program loads a pretrained YOLOv8n model exported in **ONNX** format, runs inference on an image using **ONNX Runtime** (C API), and displays the result on screen using **SDL2**, drawing the bounding box for each detection on top of the image, together with its label and confidence score.
