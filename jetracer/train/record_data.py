# record_data.py
# Python Script to record training data to the Jetson Nano's MicroSD card.
# After xbox controller config is complete and configuration is complete in this script,
# This script allows the user to drive the car around the track using the XBOX controller.
# It will save the flattened RGB, IR, and depth data as independent variables, and the steering and acceleration values are the target variables of the dataset.
# This data can be used to predict steering and acceleration values when given an image of what the car sees.
"""
Human driving dataset collection for LaTrax + PCA9685 + RealSense
Uses Xbox controller and working RealSense pipeline.
Stores flattened RGB, IR, and depth images as features.
"""

import numpy as np
import os
import csv
import time
import threading
from datetime import datetime
import pygame
from smbus2 import SMBus
import cv2
import realsense_full  # Use your verified RealSense pipeline

# ================= CONFIG =================
class Config:
    PCA_ADDR = 0x40
    STEERING_CHANNEL = 0
    THROTTLE_CHANNEL = 1
    STEERING_AXIS = 0
    THROTTLE_AXIS = 1
    PWM_FREQ = 50

    TARGET_FPS = 15          # Samples per second
    RUN_DIR = f"runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    DELETE_N_FRAMES = 50

cfg = Config()
os.makedirs(cfg.RUN_DIR, exist_ok=True)

# ================= PCA9685 =================
class PCA9685:
    def __init__(self, bus=1, address=cfg.PCA_ADDR):
        self.bus = SMBus(bus)
        self.address = address
        self.set_pwm_freq(cfg.PWM_FREQ)

    def set_pwm_freq(self, freq_hz):
        prescaleval = 25000000.0 / 4096 / freq_hz - 1
        prescale = int(prescaleval + 0.5)
        self.bus.write_byte_data(self.address, 0x00, 0x10)
        self.bus.write_byte_data(self.address, 0xFE, prescale)
        self.bus.write_byte_data(self.address, 0x00, 0x80)

    def set_pwm(self, channel, on, off):
        self.bus.write_byte_data(self.address, 0x06 + 4*channel, on & 0xFF)
        self.bus.write_byte_data(self.address, 0x07 + 4*channel, on >> 8)
        self.bus.write_byte_data(self.address, 0x08 + 4*channel, off & 0xFF)
        self.bus.write_byte_data(self.address, 0x09 + 4*channel, off >> 8)

    def set_us(self, channel, microseconds):
        pulse_length = 1000000 / cfg.PWM_FREQ / 4096
        pulse = int(microseconds / pulse_length)
        self.set_pwm(channel, 0, pulse)

pca = PCA9685()
STEERING_CENTER = 1500
THROTTLE_CENTER = 1500
STEERING_MAX = 2000
STEERING_MIN = 1000
THROTTLE_MAX = 2000
THROTTLE_MIN = 1000
pca.set_us(cfg.STEERING_CHANNEL, STEERING_CENTER)
pca.set_us(cfg.THROTTLE_CHANNEL, THROTTLE_CENTER)

# ================= PYGAME =================
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise Exception("No joystick detected! Connect Xbox controller.")
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Joystick detected: {joystick.get_name()}")

# ================= HELPERS =================
def pwm_to_norm(us):
    return (us - 1500) / 500.0

def get_flattened_images():
    rgb = realsense_full.get_rgb_image()
    ir = realsense_full.get_ir_image()
    depth = realsense_full.get_depth_image()
    if rgb is None or ir is None or depth is None:
        return None, None, None
    rgb_flat = cv2.resize(rgb, (224,224)).flatten()
    ir_flat = cv2.resize(ir, (224,224)).flatten()
    depth_flat = cv2.resize(depth, (224,224)).flatten()
    return rgb_flat, ir_flat, depth_flat

# ================= DATASET SETUP =================
csv_path = os.path.join(cfg.RUN_DIR, "dataset.csv")
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)
header = ["timestamp","steer_us","throttle_us","steer_norm","throttle_norm","rgb_flat","depth_flat","ir_flat"]
writer.writerow(header)

# Instructions file
with open(os.path.join(cfg.RUN_DIR,"README.txt"),"w") as f:
    f.write(
        "Dataset instructions:\n"
        "- Features: flattened RGB (224x224x3), depth (224x224), IR (224x224)\n"
        "- Targets: steering (us & normalized), throttle (us & normalized)\n"
        "- Left stick: horizontal=steering, vertical=forward/reverse\n"
        "- Pause collection: ENTER\n"
        "- Delete last 50 frames: BACKSPACE\n"
        "- Exit: Ctrl+C\n"
        f"- CSV saved at: {csv_path}\n"
    )

# ================= INPUT THREAD =================
frame_idx = 0
recording = False
last_save_time = 0
MIN_CHANGE_US = 15
last_steer = STEERING_CENTER
last_throttle = THROTTLE_CENTER

def delete_last_n(n):
    global frame_idx
    if frame_idx == 0:
        print("\nNothing to delete")
        return
    n = min(n, frame_idx)
    with open(csv_path,'r') as f:
        lines = f.readlines()
    with open(csv_path,'w') as f:
        f.writelines(lines[:-n])
    frame_idx -= n
    print(f"\nDeleted last {n} frames -> now at {frame_idx}")

def input_thread():
    global recording
    while True:
        key = input()
        if key == "":
            recording = not recording
            print(f"\n>>> {'RECORDING' if recording else 'PAUSED'} ({cfg.TARGET_FPS} FPS)")
        elif key in ("\x08","\x7f","\b"):
            delete_last_n(cfg.DELETE_N_FRAMES)

threading.Thread(target=input_thread, daemon=True).start()

# ================= TEST CAMERA & CONTROLS =================
print("\n--- TESTING CAMERA & CAR CONTROLS ---")
rgb_flat, ir_flat, depth_flat = get_flattened_images()
if rgb_flat is not None:
    print(f"RGB flat length: {len(rgb_flat)}, IR flat length: {len(ir_flat)}, Depth flat length: {len(depth_flat)}")
else:
    print("ERROR: Could not retrieve camera frames!")

# Test car controls
pygame.event.pump()
steer = -joystick.get_axis(cfg.STEERING_AXIS)
throttle = -joystick.get_axis(cfg.THROTTLE_AXIS)
steer_us = int(STEERING_CENTER + steer*(STEERING_MAX - STEERING_CENTER))
throttle_us = int(THROTTLE_CENTER + throttle*(THROTTLE_MAX - THROTTLE_CENTER))
pca.set_us(cfg.STEERING_CHANNEL, steer_us)
pca.set_us(cfg.THROTTLE_CHANNEL, throttle_us)
print("OK – Car controls working\n")
print("--- CAMERA AND CONTROLS TEST COMPLETE ---\n")

# ================= MAIN DATA COLLECTION LOOP =================
print("Controls:\nENTER -> Start/Pause\nBACKSPACE -> Delete last 50 frames\nCtrl+C -> Quit\n")
try:
    while True:
        pygame.event.pump()
        if not recording:
            time.sleep(0.1)
            continue
        now = time.time()
        if now - last_save_time < (1.0 / cfg.TARGET_FPS):
            time.sleep(0.01)
            continue

        steer = -joystick.get_axis(cfg.STEERING_AXIS)
        throttle_axis = -joystick.get_axis(cfg.THROTTLE_AXIS)
        steer_us = int(STEERING_CENTER + steer*(STEERING_MAX - STEERING_CENTER))
        throttle_us = int(THROTTLE_CENTER + throttle_axis*(THROTTLE_MAX - THROTTLE_CENTER))
        pca.set_us(cfg.STEERING_CHANNEL, steer_us)
        pca.set_us(cfg.THROTTLE_CHANNEL, throttle_us)

        if abs(steer_us-last_steer)<MIN_CHANGE_US and abs(throttle_us-last_throttle)<MIN_CHANGE_US:
            continue
        last_steer = steer_us
        last_throttle = throttle_us

        rgb_flat, ir_flat, depth_flat = get_flattened_images()
        if rgb_flat is None:
            continue

        writer.writerow([
            time.time(), steer_us, throttle_us,
            pwm_to_norm(steer_us), pwm_to_norm(throttle_us),
            rgb_flat.tolist(), depth_flat.tolist(), ir_flat.tolist()
        ])
        csv_file.flush()
        frame_idx += 1
        last_save_time = now
        print(f"\rFrame {frame_idx:05d} | S {pwm_to_norm(steer_us):+0.3f} | T {pwm_to_norm(throttle_us):+0.3f}", end="")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    csv_file.close()
    pca.set_us(cfg.STEERING_CHANNEL, STEERING_CENTER)
    pca.set_us(cfg.THROTTLE_CHANNEL, THROTTLE_CENTER)
    pygame.quit()
    if realsense_full.pipeline:
        realsense_full.pipeline.stop()
        print("[RealSense] Pipeline stopped.")
    print(f"\nDATA SAVED -> {cfg.RUN_DIR}")
