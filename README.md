# Industrial Defect Inspection using YOLOv8 and U-Net

## Overview

This project presents a hybrid computer vision pipeline for automatic industrial defect inspection. It combines **YOLOv8** for object detection with a custom-trained **U-Net (ResNet34)** model for semantic segmentation, followed by post-processing to measure defect size and generate inspection results.

The system is designed to detect defects, segment them at the pixel level, estimate their physical dimensions, and automatically save annotated images and measurement reports.

---

## Project Pipeline

```text
Input Image
      │
      ▼
YOLOv8 Object Detection
      │
      ▼
Region of Interest (ROI) Extraction
      │
      ▼
U-Net Semantic Segmentation
      │
      ▼
Binary Defect Mask
      │
      ▼
Connected Component Analysis
      │
      ▼
Defect Measurement (Pixel & mm²)
      │
      ▼
Visualization & Report Generation
```

---

## Repository Contents

### 1. YOLO Training

The object detector was trained using **Ultralytics YOLOv8** on a custom industrial defect dataset.

The training workflow was initially based on the excellent Google Colab tutorial by **EdjeElectronics**.

Reference:
https://github.com/EdjeElectronics/Train-and-Deploy-YOLO-Models

The trained detector (`best.pt`) is used as the first stage of the inspection pipeline.

---

### 2. U-Net Training (`train_unet.ipynb`)

This notebook trains a semantic segmentation model using **PyTorch** and **segmentation_models_pytorch**.

Features include:

* Custom PyTorch Dataset
* Grayscale image processing
* Data augmentation using Albumentations
* Transfer learning with ResNet34 encoder
* Dice Loss optimization
* Adam optimizer
* Automatic checkpoint saving
* Google Colab support

The best-performing model is saved as:

```
best_defect_unet.pth
```

---

### 3. Defect Inspection Pipeline (`Main_code.ipynb`)

This notebook combines the trained YOLO detector and U-Net segmentation model into a complete inspection system.

Pipeline features:

* Load trained YOLOv8 detector
* Detect defect regions
* Extract Regions of Interest (ROI)
* Perform semantic segmentation
* Generate binary defect masks
* Connected component analysis
* Calculate defect area
* Convert pixel measurements into mm²
* Save annotated images
* Export CSV and summary reports
* Benchmark CPU, GPU, RAM and inference time

---

## Technologies Used

* Python
* PyTorch
* YOLOv8 (Ultralytics)
* segmentation_models_pytorch
* OpenCV
* Albumentations
* NumPy
* Pillow
* Google Colab

---

## Model Architecture

### Detection

* YOLOv8

### Segmentation

* U-Net
* ResNet34 encoder (ImageNet pretrained)
* Binary segmentation

---

## Output

The pipeline produces:

* Detected defects
* Segmented defect masks
* Annotated inspection images
* Defect area measurements
* CSV reports
* Summary text reports
* Performance statistics

---

## Future Improvements

* Separate training, inference and utility modules into Python scripts
* Implement train/validation/test dataset split
* Add IoU and Dice evaluation metrics
* Improve visualization of segmentation results
* Package the project for deployment
* Add support for batch inference

---

## Acknowledgments

The initial YOLOv8 training workflow was based on the Google Colab tutorial developed by EdjeElectronics.

This project extends that workflow by:

* Training a custom defect detector
* Developing a custom U-Net segmentation model
* Integrating object detection and semantic segmentation
* Implementing post-processing for defect measurement
* Generating automated inspection reports
* Benchmarking computational performance

---

## Author

Developed as a computer vision project focused on industrial defect detection, semantic segmentation, and automated image analysis using deep learning.
