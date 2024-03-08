# Image Processing and Object Detection Pipeline

This repository contains a Python script for an end-to-end image processing pipeline, utilizing TensorFlow Hub for image stylization and PyTorch YOLOv5 for object detection.

# Installation

```bash
# Install virtualenv
pip install virtualenv
```
## Create and activate virtual environment (Windows)

```bash
virtualenv venv
venv\Scripts\activate
```

## Install requirements
```bash
pip install -r requirements.txt
```
## Dataset Link for style_images

[Best Artwork of All Time -- ICARDO](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time)


# Usage

Place your content images in the 'test_images' directory and style images in the 'style_images' directory

## Run script

```bash
python final_test.py
```
Input images will be picked from the test_images and style_images directories respectively, and the generated output will be stored in their respective directories. Detected objects will be placed alongside styled generated images, and an output_data.csv file will be generated in the root directory of the project containing paths and content loss of the outputs.
