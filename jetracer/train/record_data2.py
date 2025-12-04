"""
Minimal RC car data collection with session management:
- Uses only RGB camera and a single front depth value
- Immediate PWM output, low-latency
- Supports multi-session recording and deletion of last N frames or entire session
- Throttle capped at 30% of full speed
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
import realsense_full  # Your verified RealSense pipeline

# ================= CONFIG =================
class Config:
    PCA_ADDR = 0x40
    STEERING_CHANNEL = 0
    THROTTLE_CHANNEL = 1
    STEERING_AXIS = 0
    THROTTLE_AXIS = 1
    PWM_FREQ = 50
    IMG_WIDTH = 160
    IMG_HEIGHT = 120
    TARGET_FPS = 3
    DELETE_N_FRAMES = 50
    THROTTLE_MAX_SCALE = 0.25  # Max 30% of full speed

cfg = Config()

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

# ================= MULTI-SESSION SETUP =================
BASE_RUN_DIR = "runs_rgb_depth"
os.makedirs(BASE_RUN_DIR, exist_ok=True)

def create_new_session():
    session_dir = os.path.join(BASE_RUN_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(session_dir, exist_ok=True)
    return session_dir

RUN_DIR = create_new_session()
csv_path = os.path.join(RUN_DIR, "dataset.csv")
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["timestamp","steer_us","throttle_us","steer_norm","throttle_norm","depth_front","rgb_path"])
frame_idx = 0

# ================= HELPERS =================
def pwm_to_norm(us):
    return (us - 1500) / 500.0

def get_rgb_and_front_depth():
    rgb = realsense_full.get_rgb_image()
    depth = realsense_full.get_depth_image()
    if rgb is None or depth is None:
        return None, None
    rgb_small = cv2.resize(rgb, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
    h, w = depth.shape
    center_depth = float(depth[h//2, w//2])
    return rgb_small, center_depth

def delete_last_n(n):
    global frame_idx
    if frame_idx == 0:
        print("\nNothing to delete")
        return
    n = min(n, frame_idx)
    with open(csv_path,'r') as f:
        lines = f.readlines()
    # Delete last n data rows
    with open(csv_path,'w') as f:
        f.writelines(lines[:1] + lines[-frame_idx:-frame_idx+n] if frame_idx > n else lines[:1])
    # Delete images
    for i in range(frame_idx-n, frame_idx):
        rgb_file = os.path.join(RUN_DIR, f"rgb_{i:05d}.png")
        if os.path.exists(rgb_file):
            os.remove(rgb_file)
    frame_idx -= n
    print(f"\nDeleted last {n} frames -> now at {frame_idx}")

def delete_current_session():
    global frame_idx, csv_file, writer, RUN_DIR, csv_path
    confirm = input(f"\nDelete current session '{RUN_DIR}'? [y/N]: ").strip().lower()
    if confirm == 'y':
        csv_file.close()
        for fname in os.listdir(RUN_DIR):
            os.remove(os.path.join(RUN_DIR, fname))
        os.rmdir(RUN_DIR)
        print(f"Session '{RUN_DIR}' deleted!")
        RUN_DIR = create_new_session()
        csv_path = os.path.join(RUN_DIR, "dataset.csv")
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["timestamp","steer_us","throttle_us","steer_norm","throttle_norm","depth_front","rgb_path"])
        frame_idx = 0

# ================= INPUT THREAD =================
recording = False
def input_thread():
    global recording
    while True:
        key = input().strip().lower()
        if key == "":
            recording = not recording
            print(f"\n>>> {'RECORDING' if recording else 'PAUSED'}")
        elif key in ("\x08","\x7f","\b"):
            delete_last_n(cfg.DELETE_N_FRAMES)
        elif key in ("del","d"):
            delete_current_session()

threading.Thread(target=input_thread, daemon=True).start()

# ================= STARTUP INSTRUCTIONS =================
print("\n--- RC CAR DATA COLLECTION ---")
print("Controls:")
print("  ENTER -> Start/Pause recording")
print("  BACKSPACE -> Delete last 50 frames")
print("  DEL/d -> Delete current session")
print("  Ctrl+C -> Quit")
print("\n>>> RECORDING will start after pressing ENTER\n")

# ================= MAIN LOOP =================
last_steer = STEERING_CENTER
last_throttle = THROTTLE_CENTER
MIN_CHANGE_US = 15

try:
    last_save_time = 0
    while True:
        pygame.event.pump()
        steer = -joystick.get_axis(cfg.STEERING_AXIS)
        throttle_axis = -joystick.get_axis(cfg.THROTTLE_AXIS)

        # Cap throttle to 30% of max
        throttle_axis = max(min(throttle_axis, cfg.THROTTLE_MAX_SCALE), -cfg.THROTTLE_MAX_SCALE)

        steer_us = int(STEERING_CENTER + steer*(STEERING_MAX - STEERING_CENTER))
        throttle_us = int(THROTTLE_CENTER + throttle_axis*(THROTTLE_MAX - THROTTLE_CENTER))

        pca.set_us(cfg.STEERING_CHANNEL, steer_us)
        pca.set_us(cfg.THROTTLE_CHANNEL, throttle_us)

        now = time.time()
        if recording and now - last_save_time >= 1.0 / cfg.TARGET_FPS:
            if abs(steer_us-last_steer) >= MIN_CHANGE_US or abs(throttle_us-last_throttle) >= MIN_CHANGE_US:
                last_steer, last_throttle = steer_us, throttle_us
                rgb, depth_front = get_rgb_and_front_depth()
                if rgb is not None:
                    rgb_path = os.path.join(RUN_DIR, f"rgb_{frame_idx:05d}.png")
                    cv2.imwrite(rgb_path, rgb)
                    writer.writerow([time.time(), steer_us, throttle_us,
                                     pwm_to_norm(steer_us), pwm_to_norm(throttle_us),
                                     depth_front, rgb_path])
                    csv_file.flush()
                    frame_idx += 1
                    last_save_time = now
                    print(f"\rFrame {frame_idx:05d} | S {pwm_to_norm(steer_us):+0.3f} | "
                          f"T {pwm_to_norm(throttle_us):+0.3f} | Depth {depth_front:.2f}", end="")

        time.sleep(0.01)

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
    print(f"\nDATA SAVED -> {RUN_DIR}")
