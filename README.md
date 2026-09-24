# 🏸 Badminton AI — Footwork Detection & Movement Analysis

An AI-powered computer vision project for analyzing badminton player movement and identifying footwork patterns on a singles badminton court using **YOLOv8** and video-based movement analysis.

## 🚀 Project Overview

Badminton requires fast and accurate movement across different areas of the court. This project aims to use computer vision and deep learning to automatically detect a badminton player from video footage, analyze their movement, and classify their position into one of **six court zones/corners**.

The system processes badminton video footage and converts player movement into meaningful court-position information.

## 🎯 Objective

The main objectives of this project are:

* Detect the badminton player from video frames.
* Track the player's movement across the court.
* Analyze movement patterns during gameplay.
* Divide the singles court into six movement zones.
* Classify the player's position based on their detected location.
* Generate movement-related output that can be used for further performance analysis.

## 🧠 System Architecture

```text
Badminton Video
       ↓
Frame Extraction
       ↓
Annotated Dataset
       ↓
YOLOv8 Training
       ↓
Trained YOLOv8 Model
       ↓
New Video
       ↓
Player Detection
       ↓
Movement Analysis
       ↓
6-Corner Classification
       ↓
Output
```

## 🔍 How It Works

### 1. Video Input

A badminton match or training video is used as the input.

### 2. Frame Extraction

Frames are extracted from the video at selected intervals to create images suitable for dataset preparation and model training.

### 3. Dataset Annotation

The extracted frames are annotated to identify the badminton player.

The annotated dataset is then organized into training and validation data for YOLOv8.

### 4. YOLOv8 Training

The annotated dataset is used to train a YOLOv8 object-detection model.

The trained model learns to detect the badminton player in different positions and movements.

### 5. Player Detection

The trained YOLOv8 model is applied to a new badminton video.

For each frame, the model detects the player and obtains their bounding-box coordinates.

### 6. Movement Analysis

The detected player coordinates are analyzed across consecutive frames to determine how the player moves around the court.

Movement information can be used to identify changes in court position and movement patterns.

### 7. Six-Corner Classification

The badminton singles court is divided into six movement zones:

```text
        BACK LEFT        BACK RIGHT

             ┌───────────────┐
             │
```
