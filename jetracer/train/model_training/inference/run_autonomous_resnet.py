# run_autonomous_resnet.py
# Fully working autonomous drive script to run the resnet model for Jetson Nano + RealSense + LaTrax
# Uses TensorRT and correct PCA9685 control to move the car.
# Will automatically default to using PyTorch to run the model if TensorRT can't be used.
# Adjust MODEL_TRT_PATH or MODEL_PYTORCH_PATH if needed.
# FIXED: Added ESC arming sequence for LaTrax/HobbyWing ESC to ensure it responds to throttle.

import os
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import time
import signal
import sys
import subprocess
from smbus2 import SMBus



# ------------------- TRY TO LOAD TensorRT -------------------
try:
    from torch2trt import TRTModule
    HAS_TRT = True
    print("[OK] TensorRT support loaded")
except ImportError:
    HAS_TRT = False
    print("[WARNING] torch2trt not found → will be very slow")

# ------------------- CONFIGURATION -------------------
MODEL_ARCHITECTURE = 'resnet101'  # Set to 'resnet18', 'resnet34', 'resnet50', or 'resnet101'
MODEL_TRT_PATH   = f"checkpoints/model_8_resnet101/best_model_trt.pth"
MODEL_PYTORCH_PATH = f"checkpoints/model_8_resnet101/best_model.pth"

FIXED_THROTTLE_NORM = 0.3        # Set to 1.0 for full throttle; adjust lower if too fast
STEERING_GAIN = 1.0
STEERING_OFFSET = 0.0

TARGET_FPS = 15
FRAME_SKIP = 1                    # With TRT you can do every frame!

CAMERA_WIDTH = 848
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ------------------- PCA9685 RAW INIT (ADDED) -------------------
def init_pca9685_raw():
    """
    Runs the i2cset initialization sequence you normally paste manually.
    Ensures PCA9685 is in correct state after Jetson reboot.
    """
    print("[INIT] Running raw i2c warm-up init for PCA9685...")

    cmds = [
        ["i2cset", "-y", "1", "0x40", "0x00", "0x21"],
        ["i2cset", "-y", "1", "0x40", "0xFE", "0x65"],
        ["i2cset", "-y", "1", "0x40", "0x00", "0xA1"],

        # Steering (channel 0)
        ["i2cset", "-y", "1", "0x40", "0x08", "0x00", "0x06"],
        ["sleep", "2"],
        ["i2cset", "-y", "1", "0x40", "0x08", "0x00", "0x09"],
        ["sleep", "2"],
        ["i2cset", "-y", "1", "0x40", "0x08", "0x00", "0x06"],
        ["sleep", "1"],

        # Throttle (channel 1)
        ["i2cset", "-y", "1", "0x40", "0x0C", "0x00", "0x09"],
        ["sleep", "4"],
        ["i2cset", "-y", "1", "0x40", "0x0C", "0x00", "0x06"]
    ]

    for cmd in cmds:
        if cmd[0] == "sleep":
            time.sleep(float(cmd[1]))
        else:
            subprocess.run(["sudo"] + cmd, check=False)

    print("[INIT] PCA9685 warm-up complete!\n")

# ------------------- PCA9685 (CORRECTED & TESTED) -------------------
class PCA9685:
    def __init__(self, bus=1, address=0x40):
        self.bus = SMBus(bus)
        self.address = address
        self.set_pwm_freq(50)

    def set_pwm_freq(self, freq_hz=50):
        prescaleval = 25000000.0
        prescaleval /= 4096.0
        prescaleval /= float(freq_hz)
        prescaleval -= 1.0
        prescale = int(prescaleval + 0.5)

        oldmode = self.bus.read_byte_data(self.address, 0x00)
        newmode = (oldmode & 0x7F) | 0x10
        self.bus.write_byte_data(self.address, 0x00, newmode)
        self.bus.write_byte_data(self.address, 0xFE, prescale)
        self.bus.write_byte_data(self.address, 0x00, oldmode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, 0x00, oldmode | 0x80)

    def set_pwm(self, channel, on, off):
        self.bus.write_byte_data(self.address, 0x06 + 4*channel, on & 0xFF)
        self.bus.write_byte_data(self.address, 0x07 + 4*channel, on >> 8)
        self.bus.write_byte_data(self.address, 0x08 + 4*channel, off & 0xFF)
        self.bus.write_byte_data(self.address, 0x09 + 4*channel, off >> 8)

    def set_us(self, channel, microseconds):
        pulse = int(microseconds * 4096 * 50 / 1000000 + 0.5)
        self.set_pwm(channel, 0, pulse)

# ------------------- HARDWARE LIMITS -------------------
STEERING_CHANNEL = 0
THROTTLE_CHANNEL = 1

STEERING_CENTER = 1500
STEERING_MIN    = 1000
STEERING_MAX    = 2000

THROTTLE_CENTER = 1500
THROTTLE_MIN    = 1200
THROTTLE_MAX    = 1900

# ------------------- LOAD MODEL -------------------
DEVICE = torch.device("cuda")

def load_model():
    if HAS_TRT and os.path.exists(MODEL_TRT_PATH):
        print("[FAST] Loading TensorRT model (10-14 FPS expected)")
        model = TRTModule()
        model.load_state_dict(torch.load(MODEL_TRT_PATH))
        return model

    print("[SLOW] Loading PyTorch model (2-4 FPS)")
    from torchvision import models
    import torch.nn as nn

    class ControlModel(nn.Module):
        def __init__(self):
            super().__init__()
            if MODEL_ARCHITECTURE == 'resnet18':
                backbone = models.resnet18(pretrained=False)
                feature_dim = 512
            elif MODEL_ARCHITECTURE == 'resnet34':
                backbone = models.resnet34(pretrained=False)
                feature_dim = 512
            elif MODEL_ARCHITECTURE == 'resnet50':
                backbone = models.resnet50(pretrained=False)
                feature_dim = 2048
            elif MODEL_ARCHITECTURE == 'resnet101':
                backbone = models.resnet101(pretrained=False)
                feature_dim = 2048
            else:
                raise ValueError(f"Unsupported architecture: {MODEL_ARCHITECTURE}")

            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            self.head = nn.Sequential(
                nn.Linear(feature_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.4),
                nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.3),
                nn.Linear(128, 1), nn.Tanh()
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.head(x)

    model = ControlModel().to(DEVICE)
    ckpt = torch.load(MODEL_PYTORCH_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt if not isinstance(ckpt, dict) else ckpt.get('model_state_dict', ckpt))
    model.eval()
    return model

# ------------------- CAMERA -------------------
pipeline = None
align = None
pipeline_started = False

def start_camera():
    global pipeline, align, pipeline_started
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.rgb8, CAMERA_FPS)
    try:
        profile = pipeline.start(config)
        align = rs.align(rs.stream.color)
        for _ in range(15):
            pipeline.wait_for_frames()
        print(f"[Camera] {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS} FPS → OK")
        pipeline_started = True
    except Exception as e:
        print("Camera failed:", e)
        print("Fix: unplug/replug, killall rs-*, or reboot")
        sys.exit(1)

def get_frame():
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color = aligned.get_color_frame()
    if not color: return None
    return np.asanyarray(color.get_data())

# ------------------- FAST PREPROCESS -------------------
def preprocess(frame):
    img = cv2.resize(frame, (160, 120))
    img = img.transpose(2, 0, 1)
    tensor = torch.from_numpy(img).float().div(255.0)
    return tensor.unsqueeze(0).to(DEVICE)

# ------------------- ARM ESC -------------------
def arm_esc(pca):
    print("[ESC] Arming sequence...")
    pca.set_us(THROTTLE_CHANNEL, THROTTLE_CENTER)
    time.sleep(3.0)
    pca.set_us(THROTTLE_CHANNEL, THROTTLE_MIN)
    time.sleep(2.0)
    pca.set_us(THROTTLE_CHANNEL, THROTTLE_CENTER)
    time.sleep(2.0)
    print("[ESC] Armed!")

# ------------------- MAIN -------------------
def main():
    print("\n" + "="*60)
    print("   JETRACER AUTONOMOUS DRIVE – TENSORRT + WORKING PCA9685")
    print("="*60 + "\n")

    # ------------------ ADDED LINE: AUTO INIT PCA9685 ------------------
    init_pca9685_raw()

    # Init hardware
    pca = PCA9685()
    pca.set_us(STEERING_CHANNEL, STEERING_CENTER)
    pca.set_us(THROTTLE_CHANNEL, THROTTLE_CENTER)
    time.sleep(1.0)

    # Arm ESC
    arm_esc(pca)

    start_camera()
    model = load_model()

    print("Warming up model...")
    dummy = torch.ones((1, 3, 120, 160), device=DEVICE)
    for _ in range(10):
        model(dummy)
    print("Warm-up complete!\n")

    throttle_us = THROTTLE_CENTER + int(FIXED_THROTTLE_NORM * (THROTTLE_MAX - THROTTLE_CENTER))
    throttle_us = np.clip(throttle_us, THROTTLE_MIN, THROTTLE_MAX)

    print(f"Starting autonomous drive...")
    print(f"Throttle locked → {throttle_us}µs")
    print("Press Ctrl+C to stop\n")

    pca.set_us(THROTTLE_CHANNEL, throttle_us)
    time.sleep(1.5)

    frame_count = infer_count = 0
    last_report = time.time()

    def stop_car(signum=None, frame=None):
        print("\n\nSTOPPING CAR...")
        pca.set_us(THROTTLE_CHANNEL, THROTTLE_CENTER)
        pca.set_us(STEERING_CHANNEL, STEERING_CENTER)
        if pipeline_started:
            pipeline.stop()
        print("Car stopped safely. Bye!")
        sys.exit(0)

    signal.signal(signal.SIGINT, stop_car)
    signal.signal(signal.SIGTERM, stop_car)

    try:
        while True:
            frame = get_frame()
            if frame is None:
                continue

            frame_count += 1
            if frame_count % FRAME_SKIP == 0:
                tensor = preprocess(frame)
                with torch.no_grad():
                    pred = model(tensor).item()

                steer_norm = np.clip(pred * STEERING_GAIN + STEERING_OFFSET, -1.0, 1.0)
                steer_us = STEERING_CENTER + int(steer_norm * (STEERING_MAX - STEERING_CENTER))
                steer_us = int(np.clip(steer_us, STEERING_MIN, STEERING_MAX))

                pca.set_us(STEERING_CHANNEL, steer_us)
                infer_count += 1

            pca.set_us(THROTTLE_CHANNEL, throttle_us)

            now = time.time()
            if now - last_report > 2.0:
                cam_fps = frame_count / (now - last_report)
                inf_fps = infer_count / (now - last_report)
                print(f"FPS: {cam_fps:5.1f} cam | {inf_fps:5.1f} inf | "
                      f"Steer → {steer_us:4d}µs | Throttle → {throttle_us}µs")
                frame_count = infer_count = 0
                last_report = now

    except KeyboardInterrupt:
        pass
    finally:
        stop_car()

if __name__ == "__main__":
    main()
