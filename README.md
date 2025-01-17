# Classroom Dispersion Detector

## Summary
The Classroom Dispersion Detector is a project designed to analyze classroom dynamics and seating arrangements using computer vision techniques. By leveraging a YOLOv8 pretrained model, this project aims to detect human faces and persons in the classroom scene. The key objective is to determine student engagement by identifying individuals who are distracted. If a person is detected without their face visible, they are classified as "Distracted," and vice versa. Such a tool can provide insights into classroom engagement, social distancing compliance, or spatial organization. This application detects distraction in images of classrooms, video footage of classrooms, and in real-time classroom scenarios
---

## Dataset

The dataset for this project was collected using a custom-built tool adapted from the [OIDv4_ToolKit](https://github.com/EscVM/OIDv4_ToolKit). The Open Image Dataset V7 was chosen for its comprehensive annotations and extensive variety, making it ideal for training object detection models like YOLO. Below are some details about the dataset:

### Open Image Dataset V7 Highlights:
- **15,851,536** bounding boxes across **600 classes**.
- **2,785,498** instance segmentations for **350 classes**.
- **3,284,280** relationship annotations for **1,466 relationships**.
- **675,155** localized narratives.
- **66,391,027** point-level annotations for **5,827 classes**.
- **61,404,966** image-level labels across **20,638 classes**.
- Extension: **478,000** crowdsourced images with over **6,000 classes**.

[Open Image Dataset V7](https://storage.googleapis.com/openimages/web/index.html)

### Data Collection Process
The dataset used in this project specifically focuses on images and labels from the following classes:
- **Person**
- **Human Face**

Using the custom-built tool, **1900 images** and their corresponding labels were downloaded. This subset provides a robust foundation for training the YOLO model to detect people and faces in classroom.

### Command to Gather Dataset:
To fetch the dataset, the following command was used:
```bash
python main.py downloader --classes Person Human face --type_csv train --limit 1900
```
This command downloads 1900 training images for the class `Person` & `Human face`.

### Data Splitting
The dataset was split into two parts:
- **1500 images** for training.
- **400 images** for validation.

---

## Data Preprocessing

To prepare the dataset for fine-tuning the YOLOv8 model, the label text files were converted into the YOLO format. The YOLO 1.1 format requires normalized values and the assignment of numeric class labels. Below are the preprocessing steps followed:

1. **Normalization:**
   - The bounding box coordinates were normalized to the range [0, 1] by dividing the absolute coordinates by the width and height of the images.
   - The format for each label entry is: `class x_center y_center width height`.

2. **Class Assignment:**
   - In our dataset, the following numeric labels were assigned:
     - `0` for the "Person" class.
     - `1` for the "Human Face" class.

3. **Conversion Script:**
   - A script was written to read the original label files, normalize the bounding box coordinates, and convert them into the YOLO format. This ensured compatibility with the YOLOv8 model.

The processed label files should be like this:
```
0 0.508437 0.499582 0.981875 0.999163
1 0.374687 0.543515 0.598125 0.676987
```

---

## Environment Setup

To ensure a consistent and reproducible environment for this project, an Anaconda environment YAML file is included. This file specifies all the dependencies and their versions required for the project.

Create the environment from the YAML file:
```bash
conda env create -f human_detection_yolo_env.yaml
```

---

## Training the YOLOv8 Model

For the training process, we utilized the YOLOv8n model, which is known for its efficiency and speed. Here are some specifics about the YOLOv8n model:
- **Model:** YOLOv8n
- **Input Size:** 640 pixels
- **mAP (mean Average Precision) 50-95:** 37.3
- **Speed on A100 TensorRT:** 0.99 ms
- **Speed on CPU ONNX:** 80.4 ms
- **Parameters:** 3.2 million
- **FLOPs:** 8.7 billion

### Training Process
Given the relatively small size of our dataset due to limited resources, the images were split into **1500 training images** and **400 validation images**. This split ensures that the model has enough data to learn effectively while also having a separate set to validate its performance.

1. **Model Initialization:**
   - Loaded the YOLOv8n model with pretrained weights.

2. **Configuration:**
   - Set the training parameters, including the number of epochs (100 epochs).

---

## Results

After training the YOLOv8n model for **100 epochs** on the fine-tuned dataset, the following results were obtained:

- **Training Duration:** The training process took approximately 2 days to complete 100 epochs.
- **Model Size:** The final weights of the model were stripped of the optimizer, resulting in a size of 6.2MB for both `last.pt` and `best.pt`.
- **Model Summary:** The YOLOv8n model consists of 168 layers with 3,005,843 parameters and 8.1 GFLOPs (Giga Floating Point Operations).

### Validation Performance
The model was validated using 100 images, resulting in the following performance metrics:
- **Precision (P):** 0.65192
- **Recall (R):** 0.50636
- **mAP 50:** 0.50636
- **mAP 50-95:** 0.29997

These metrics indicate that the model achieves a good balance between precision and recall, with a mean average precision (mAP) of 50.0% at an IoU threshold of 0.5, and 29.9% across multiple IoU thresholds from 0.5 to 0.95.

### Speed
The inference speed of the model on the validation set was as follows:
- **Pre-process:** 1.4ms per image
- **Inference:** 60.5ms per image
- **Post-process:** 0.5ms per image

### Final Validation Results
A final validation of the best model weights (`best.pt`) confirmed the following metrics on the validation dataset:
- **Precision (P):** 0.65192
- **Recall (R):** 0.50636
- **mAP 50:** 0.50636
- **mAP 50-95:** 0.29997

The results demonstrate that the YOLOv8n model effectively detects people in the validation images, achieving high precision and recall scores. These results are saved in the `runs\detect\train` directory for further analysis and potential deployment in real-time detection scenarios.

