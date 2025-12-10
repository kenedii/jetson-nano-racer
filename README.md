# Jetson Nano Racer – Autonomous RC Car with RealSense D435i

This project deploys a deep learning lane-following model on an
**NVIDIA Jetson Nano** mounted on a **LaTrax 1/18 RC car** using an
**Intel RealSense D435i** depth camera. The model predicts steering
commands from the RGB stream in real time; control is sent to the
PCA9685 servo driver to steer the car.

## Hardware

- Jetson Nano 4GB Developer Kit with ARM Cortex-A57 CPU, Ubuntu + JetPack
- LaTrax Rally 1/18 RC car
- Intel RealSense D435i (RGBD + IMU)
- TP-Link TL-WN725N USB WiFi adapter
- PCA9685 16-channel servo driver
- Pololu 4-Channel RC Servo Multiplexer
- Fan-4020-PWM-5V
- XBOX Controller
- Batteries, mounts, cabling

## Software

- Ubuntu 18.04.6 LTS
- Jetpack 4.6.1 SDK
- Python 3.6.9

## Model Architecture

- Supports several Resnet variants: ```Resnet18, Resnet34, Resnet50, Resnet101```

## Core Features

- Data collection from teleoperated driving
- ResNet-based steering model (PyTorch + TensorRT)
- Autonomous lane following on indoor track
- RealSense center-depth measurement for safety / debugging
- REST API for live predictions (`deployment/api_server.py`)
- Streamlit demo UI (`deployment/streamlit_app.py`)
- Dockerfile for Jetson deployment (`deployment/Dockerfile`)

## 1. Setup

### 1.1 Jetson Base Setup

1. Flash JetPack and boot Nano.
2. Install system packages:
   ```bash
   sudo apt-get update
   sudo apt-get install python3-pip git
