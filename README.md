# AI Assistance System for Basketball Referees

An AI-powered computer vision system designed to assist basketball referees by analyzing gameplay footage and identifying player movements and potential rule violations such as **double dribbling** and **travelling**.

The system combines **YOLOv8 pose estimation, basketball detection, body keypoint tracking, motion analysis, and rule-based logic** to analyze basketball gameplay and provide visual and audio alerts when a potential violation is detected.

---

## Project Overview

Basketball referees need to make fast and accurate decisions while continuously tracking player movements, ball possession, and rule violations.

This project explores how **computer vision can be used as an intelligent assistant for referees** by automatically analyzing gameplay footage.

The system processes video frames, detects players and the basketball, tracks important body keypoints, analyzes movement patterns, and applies basketball-specific rules to identify potential violations.

> **Note:** The system is designed as an assistive computer vision tool and does not replace human referees or official officiating systems.

---

## Problem Statement

Manual detection of certain basketball violations can be challenging because referees must simultaneously monitor:

* Player movements
* Ball movement
* Dribbling patterns
* Player possession
* Foot and body positioning
* Multiple players on the court

The goal of this project is to develop a computer vision-based assistant that can automatically analyze gameplay footage and highlight potential rule violations.

---

# System Architecture

```text
                 ┌─────────────────────────┐
                 │     Basketball Video    │
                 │        Input            │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │     Video Processing   │
                 │   Frame Extraction     │
                 │   Frame Resizing       │
                 └────────────┬────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
      ┌──────────────────┐       ┌──────────────────┐
      │ YOLOv8 Pose      │       │ Basketball       │
      │ Estimation       │       │ Detection Model  │
      └────────┬─────────┘       └────────┬─────────┘
               │                          │
               ▼                          ▼
      ┌──────────────────┐       ┌──────────────────┐
      │ Body Keypoints   │       │ Ball Position &  │
      │ & Wrist Tracking │       │ Player Movement  │
      └────────┬─────────┘       └────────┬─────────┘
               │                          │
               └────────────┬─────────────┘
                            ▼
                 ┌─────────────────────────┐
                 │ Action Recognition      │
                 │                         │
                 │ • Dribble Detection     │
                 │ • Hold Action Detection │
                 │ • Player Movement(steps)│
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Basketball Rule Logic   │
                 │                         │
                 │ • Double Dribble        │
                 │ • Travelling            │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Violation Detection     │
                 └────────────┬────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
        ┌──────────────────┐      ┌──────────────────┐
        │ Visual Alert     │      │ Audio Alert      │
        │ & Annotation     │      │ / Buzzer         │
        └──────────────────┘      └──────────────────┘
```

---
### Action Recognition & Rule Logic

The system combines **player movement and ball movement** to identify action sequences.

For example:

```text
Dribble → Hold → Dribble
             ↓
       Double Dribble Rule
             ↓
      Violation Detected
```

and:

```text
Hold → Step Taken
          ↓
   Travelling Rule
          ↓
   Violation Detected
```

These action sequences are evaluated against predefined basketball rule logic to determine potential violations.

---

## Key Features

-  **Player & Pose Detection** using YOLOv8 Pose
-  **Basketball Detection & Tracking**
-  **Player Movement Analysis**
-  **Action Recognition** from ball and player movement
-  **Double Dribble Detection**
-  **Travelling Detection**
-  **Visual & Audio Alerts**
-  **Annotated Video Output**

---

## Technologies

- **Python**
- **YOLOv8**
- **YOLOv8 Pose Estimation**
- **OpenCV**
- **NumPy**
- **Pygame**

# Installation

### 1. Clone the repository

```bash
git clone https://github.com/Devika1027/-AI-assistance-to-basketball-referees.git
cd -AI-assistance-to-basketball-referees
```

### 2. Install dependencies

```bash
pip install ultralytics opencv-python numpy pygame
```

> A `requirements.txt` file can also be used if provided with the project.

### 3. Model Setup

The YOLOv8 Pose model is included in the repository.

The basketball detection model is not included because of its file size.
You can download the pre-trained ball detection model from the following link:
Download Ball Detection Model
 https://drive.google.com/file/d/1_CIPPY2FFcikNVwCFXK8e_QujTjNI6Ya/view?usp=drive_link

Place the required model in the expected project directory before running the detection scripts.

---

# 📊 Results

The system produces annotated outputs showing the detected players, pose keypoints, basketball position, movement information, and identified violations.

## Pose & Basketball Detection
![alt text](<Results/Model A/object detection results.jpeg>)

## Player Tracking
![alt text](<Results/Player Tracking/Player Tracking.jpeg>)

## Step Detection
![alt text](<Results/Action recognition/step action detection.png>)

## Hold Detection
![alt text](<Results/Action recognition/Hold action detection.png>)

## Double Dribble Detection
![alt text](<Results/Violation Detection/Double dribble detection.png>)

## Travelling Detection
![alt text](<Results/Violation Detection/travel detection.png>)

# 🎥 Demo

A complete demonstration of the system is available in the repository:

<video controls src="demo/Demo Video.mp4" title="Title"></video>

The demo demonstrates the system processing basketball gameplay footage and identifying player movements and potential rule violations.

---

# ⚠️ Limitations

The current implementation is primarily designed for analyzing recorded basketball footage under the conditions represented in the available test videos.

Detection performance can vary depending on factors such as:

* Camera angle
* Video quality
* Player occlusion
* Lighting conditions
* Ball visibility
* Player overlap
* Movement speed

The system should therefore be considered an **AI-assisted analysis tool**, rather than a replacement for official basketball refereeing systems.

---


