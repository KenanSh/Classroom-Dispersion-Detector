# Classroom Dispersion Detector

## Summary
The Classroom Dispersion Detector is a project designed to analyze classroom dynamics and seating arrangements using computer vision techniques. By leveraging a YOLOv8 pretrained model, this project aims to detect human faces and persons in the classroom scene. The key objective is to determine student engagement by identifying individuals who are distracted. If a person is detected without their face visible, they are classified as "Distracted," and vice versa. Such a tool can provide insights into classroom engagement, social distancing compliance, or spatial organization.

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

## Environment Setup

To ensure a consistent and reproducible environment for this project, an Anaconda environment YAML file is included. This file specifies all the dependencies and their versions required for the project.

Create the environment from the YAML file:
```bash
conda env create -f human_detection_yolo_env.yaml
```

