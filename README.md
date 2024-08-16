# Face Recognition and Detection Project

## Overview

This project is a simple face recognition system using OpenCV. It detects faces in real-time from a webcam feed and identifies them based on a trained model. The project is implemented in Python using OpenCV's `cv2` module and the Local Binary Patterns Histograms (LBPH) face recognizer.

## Features

- Real-time face detection using Haar cascades.
- Real-time eyes , nose , mouth detechtion using Haar cascades.
- Face recognition using LBPH algorithm.
- Easily trainable with new faces.

## Requirements

- Python 
- OpenCV 
- NumPy
- PIL (Pillow)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shalini_1008/face-recognition.git
   cd face-recognition
2. **Download dependencies:**
   ```bash
   pip install python
   pip install opencv-python-headless numpy pillow
3. **Train the model:**
   Place your training images in the data directory. Ensure that the images are labeled correctly by their IDs.
   Run the following script to train your model:
   ```bash
   python train_classifier.py
   This will generate a classifier.yml file used for recognition.

## ScreenShots and Clips

![Screenshot 2024-08-16 135204](https://github.com/user-attachments/assets/a6b97502-d3a4-47cb-a6b5-5d3f6e7a8b2c)
![Screenshot 2024-08-15 231742](https://github.com/user-attachments/assets/e89ebb73-69b6-46c3-bc90-beb045c13305)





      
