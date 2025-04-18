# Real-Time Object Detection, Depth Estimation and Speech Synthesis.

This project demonstrates real-time object detection and depth estimation using **YOLOv5** for object detection and **MiDaS** for monocular depth estimation. The system can detect objects in a video stream, estimate their depth, and provide spoken feedback through **Text-to-Speech (TTS)** regarding the object's position and depth relative to the camera. It is an attempt to provide a tool for the visually impaired.

## Overview

This project brings together state-of-the-art deep learning models to enable real-time understanding of a scene from a single camera input. It not only detects objects but also provides contextual awareness of their position and distance, with intelligent feedback through speech synthesis.

### Key Features

- **Real-Time Object Detection**  
  Utilizes **YOLOv5** to detect multiple objects in a live video stream with high accuracy and speed.

- **Object Position Classification**  
  Once detected, each object's location within the frame is categorized as one of the following zones:
  - **Top Left**
  - **Top Right**
  - **Center**
  - **Bottom Left**
  - **Bottom Right**

- **Depth Estimation**  
  Integrates the **MiDaS** model to estimate the depth of each detected object from a single RGB image, classifying objects as either **Near** or **Far** relative to the camera.

- **Object Tracking & Feedback Optimization**  
  Implements lightweight object tracking to remember previously detected objects. This helps avoid redundant feedback by ensuring that the system does not repeatedly speak about the same object across multiple frames.

- **Temporal Smoothing**  
   Implements temporal smoothing through a sliding window approach with deques, averaging bounding box coordinates across multiple frames, to reduce jitter in detections and provide more consistent feedback.

- **Text-to-Speech (TTS) Feedback**  
  Uses `pyttsx3` to provide real-time spoken feedback, describing each newly detected object's type, position, and depth — for example:  
  _"Person at center, near."_

- **Flexible Input Options**  
  Supports both **webcam** and **IP camera** streams for flexible deployment in various environments.


## Technologies Used

- **Python**: Programming language for implementing the system.
- **PyTorch**: Deep learning framework for loading YOLOv5 and MiDaS models.
- **YOLOv5**: A state-of-the-art model for real-time object detection.
- **MiDaS**: A deep learning model for monocular depth estimation.
- **OpenCV**: Computer vision library for video streaming, image processing, and drawing bounding boxes.
- **pyttsx3**: Text-to-speech engine to provide feedback.
- **NumPy**: Library for numerical computations.
- **PIL (Pillow)**: Image processing library for depth map generation.

## Installation

### Steps to Install

1. Clone the repository to your local machine.
    ```bash
    git clone https://github.com/meghahaa/ObjectDepthVoice.git
    cd ObjectDepthVoice
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Make sure to have a webcam or an IP camera ready for real-time video feed.
   
## Usage

### Running the Application

1. **Run the application**  
   Simply execute the script using:
   ```bash
    python main.py
    ```
2. **Choose the video source**  
   After running the script, you’ll be prompted to select your preferred video source:
   - Type `webcam` to use your device’s built-in camera.
   - Type `ipcam` to use a stream from an IP camera (e.g. : the IP Webcam Android app).  
      You will then be prompted to enter the stream URL (e.g. : http://192.168.1.100:8080/video).

## Future Enhancements
Although the current system combines object detection, depth estimation, speech synthesis effectively, there are several areas that could be improved to enhance accuracy, performance, speed, and usability.

- **Improve Detection Accuracy**  
  Fine-tune the object detection model or experiment with different YOLO variants for better precision and recall.

- **Enhanced Object Tracking**  
  Implement a more robust tracking algorithm to maintain consistent identities and reduce repeated feedback.

- **Better Depth Classification**  
  Calibrate depth estimation more accurately for improved categorization between "near" and "far."

- **Optimized Voice Feedback**  
  Add smoother and more natural speech with options to adjust verbosity or language.

- **User-Friendly Input Options**  
  Add a simple interface for switching between webcam, IP camera, or video file inputs.

