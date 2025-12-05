#!/usr/bin/env python3
"""
Minimal RC car data collection with session management:
- Uses only RGB camera and a single front depth value (via realsense_full)
- Immediate PWM output via PCA9685 (smbus2)
- Supports multi-session recording and deletion of last N frames or entire session
- Throttle capped at configurable percent of full speed
- All 3 control modes supported (sticks + triggers)
"""

import os
import csv
import time
import threading
import atexit
import queue
from datetime import datetime
import pygame
from smbus2 import SMBus
import cv2
import realsense_full  # RealSense pipeline (your module)

# ================= ASYNC SAVER =================
class AsyncSaver:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            try:
                item = self.queue.get(timeout=0.1)
                if item is None:
                    continue
                
                # Unpack
                func, args = item
                try:
                    func(*args)
                except Exception as e:
                    print(f"[Saver Error] {e}")
                
                self.queue.task_done()
            except queue.Empty:
                continue

    def save(self, func, *args):
        self.queue.put((func, args))

    def stop(self):
        self.running = False
        self.thread.join()

saver = AsyncSaver()

# ================= CONFIG =================
class Config:
    PCA_ADDR = 0x40
    STEERING_CHANNEL = 0
    THROTTLE_CHANNEL = 1

    STEERING_AXIS = 0
    THROTTLE_AXIS = 1        # Default to left Y, will be overridden if mode 2
    RIGHT_THROTTLE_AXIS = 4  # Right stick Y for separate mode
    LEFT_TRIGGER_AXIS = 2    # LT (original working)
    RIGHT_TRIGGER_AXIS = 5   # RT (original working)

    PWM_FREQ = 50
    IMG_WIDTH = 160
    IMG_HEIGHT = 120
    TARGET_FPS = 2
    DELETE_N_FRAMES = 12

    # Safety scaling (set to safe defaults; change if needed)
    THROTTLE_MAX_SCALE = 0.30  # 30% of full travel
    STEERING_MAX_SCALE = 1.00  # 100% steering
    
    # Steering Gamma for fine control (1.0 = linear, 2.0 = quadratic)
    STEERING_GAMMA = 2.5

    # Deadzone to ignore tiny stick/trigger noise
    AXIS_DEADZONE = 0.03

cfg = Config()

# ================= PCA9685 (SMBus) =================
class PCA9685:
    def __init__(self, bus=1, address=cfg.PCA_ADDR):
        self.bus = SMBus(bus)
        self.address = address
        self.set_pwm_freq(cfg.PWM_FREQ)

    def set_pwm_freq(self, freq_hz):
        prescaleval = 25000000.0 / 4096 / freq_hz - 1
        prescale = int(prescaleval + 0.5)
        # reset + set prescale + restart
        self.bus.write_byte_data(self.address, 0x00, 0x10)
        self.bus.write_byte_data(self.address, 0xFE, prescale)
        self.bus.write_byte_data(self.address, 0x00, 0x80)

    def set_pwm(self, channel, on, off):
        self.bus.write_byte_data(self.address, 0x06 + 4*channel, on & 0xFF)
        self.bus.write_byte_data(self.address, 0x07 + 4*channel, (on >> 8) & 0xFF)
        self.bus.write_byte_data(self.address, 0x08 + 4*channel, off & 0xFF)
        self.bus.write_byte_data(self.address, 0x09 + 4*channel, (off >> 8) & 0xFF)

    def set_us(self, channel, microseconds):
        pulse_length = 1000000.0 / cfg.PWM_FREQ / 4096.0
        pulse = int(microseconds / pulse_length)
        self.set_pwm(channel, 0, pulse)

pca = PCA9685()

# servo/esc pulse parameters (μs)
STEERING_CENTER = 1500
THROTTLE_CENTER = 1500
STEERING_MAX = 2000
STEERING_MIN = 1000
THROTTLE_MAX = 2000
THROTTLE_MIN = 1000

# Neutralize helper and safe exit
def neutralize():
    try:
        pca.set_us(cfg.STEERING_CHANNEL, STEERING_CENTER)
        pca.set_us(cfg.THROTTLE_CHANNEL, THROTTLE_CENTER)
        # short delay to ensure ESC/servo sees neutral
        time.sleep(0.02)
    except Exception as e:
        print(f"[neutralize] PCA error: {e}")

# immediate neutral on startup and register for exit
neutralize()
atexit.register(neutralize)

# ================= PYGAME =================
pygame.init()
pygame.joystick.init()
if pygame.joystick.get_count() == 0:
    raise Exception("No joystick detected! Connect Xbox controller.")
joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Joystick detected: {joystick.get_name()}")

# ================= CONTROL MODE SELECTION =================
print("\nChoose control mode:")
print("1: Left joystick only (horizontal: steer, vertical: throttle)")
print("2: Left joystick for steer (horizontal), Right joystick for throttle (vertical)")
print("3: Left joystick for steer (horizontal), Right trigger for accelerate, Left trigger for brake/reverse")
while True:
    mode_choice = input("Enter 1, 2, or 3: ").strip()
    if mode_choice == '1':
        cfg.THROTTLE_AXIS = cfg.THROTTLE_AXIS
        mode = 1
        print("Selected Mode 1: Left joystick controls both steering and throttle.")
        break
    elif mode_choice == '2':
        cfg.THROTTLE_AXIS = cfg.RIGHT_THROTTLE_AXIS
        mode = 2
        print("Selected Mode 2: Left joystick for steering, Right joystick for throttle.")
        break
    elif mode_choice == '3':
        mode = 3
        print("Selected Mode 3: Left joystick for steering, RT/LT for accelerate/brake (normalized).")
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")

# ================= STEERING LIMIT SELECTION =================
steering_limit_input = input("\nEnter steering limit % (1-100, enter for 100): ").strip()
if steering_limit_input:
    try:
        pct = float(steering_limit_input)
        if 1 <= pct <= 100:
            cfg.STEERING_MAX_SCALE = pct / 100.0
            print(f"Steering limited to {pct}% of maximum.")
        else:
            print("Invalid range. Using 100%.")
    except ValueError:
        print("Invalid input. Using 100%.")
else:
    print("Using 100% steering.")

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
def save_task(rgb, rgb_path, row_data, writer_obj, csv_file_obj):
    try:
        cv2.imwrite(rgb_path, rgb)
        writer_obj.writerow(row_data)
        csv_file_obj.flush()
    except Exception as e:
        print(f"Save error: {e}")

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
    global frame_idx, csv_file, writer
    if frame_idx == 0:
        print("\nNothing to delete")
        return
    n = min(n, frame_idx)
    # read existing rows and keep header + remaining rows (without last n)
    with open(csv_path, 'r', newline='') as f:
        rows = list(csv.reader(f))
    header = rows[:1]
    data = rows[1:]
    remaining = data[:-n] if n < len(data) else []
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerows(header + remaining)
    # delete images
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
        try:
            csv_file.close()
        except Exception:
            pass
        for fname in os.listdir(RUN_DIR):
            try:
                os.remove(os.path.join(RUN_DIR, fname))
            except Exception:
                pass
        try:
            os.rmdir(RUN_DIR)
        except Exception:
            pass
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
        try:
            key = input().strip().lower()
        except EOFError:
            return
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
print(f"  BACKSPACE -> Delete last {cfg.DELETE_N_FRAMES} frames")
print("  DEL/d -> Delete current session")
print("  Ctrl+C -> Quit")
print("\n>>> RECORDING will start after pressing ENTER\n")

# ================= UTIL: deadzone =================
def apply_deadzone(value, threshold=cfg.AXIS_DEADZONE):
    return 0.0 if abs(value) < threshold else value

# ================= MAIN LOOP =================
last_steer_sent = -1
last_throttle_sent = -1
last_steer_rec = STEERING_CENTER
last_throttle_rec = THROTTLE_CENTER
MIN_CHANGE_US = 15

try:
    # Ensure neutral at start
    neutralize()
    last_save_time = 0
    while True:
        pygame.event.pump()

        # STEERING (left stick X) - invert to match mapping
        raw_steer = -joystick.get_axis(cfg.STEERING_AXIS)
        raw_steer = apply_deadzone(raw_steer)
        
        # Apply Gamma Curve for fine control
        # steer = sign(raw) * (|raw| ^ gamma)
        sign = 1 if raw_steer >= 0 else -1
        steer_curved = sign * (abs(raw_steer) ** cfg.STEERING_GAMMA)
        
        steer = max(min(steer_curved, cfg.STEERING_MAX_SCALE), -cfg.STEERING_MAX_SCALE)

        # THROTTLE selection by mode
        if mode == 3:
            # Correct trigger math using explicit RT & LT axes
            rt = (joystick.get_axis(cfg.RIGHT_TRIGGER_AXIS) + 1.0) / 2.0  # 0 -> 1
            lt = (joystick.get_axis(cfg.LEFT_TRIGGER_AXIS)  + 1.0) / 2.0  # 0 -> 1
            # apply trigger deadzone
            if rt < cfg.AXIS_DEADZONE:
                rt = 0.0
            if lt < cfg.AXIS_DEADZONE:
                lt = 0.0
            throttle_axis = rt - lt  # -1 -> +1
        else:
            raw_thr = -joystick.get_axis(cfg.THROTTLE_AXIS)
            raw_thr = apply_deadzone(raw_thr)
            throttle_axis = raw_thr

        # Clamp throttle to safety scale
        throttle_axis = max(min(throttle_axis, cfg.THROTTLE_MAX_SCALE), -cfg.THROTTLE_MAX_SCALE)

        # Convert to PWM µs
        steer_us = int(STEERING_CENTER + steer * (STEERING_MAX - STEERING_CENTER))
        throttle_us = int(THROTTLE_CENTER + throttle_axis * (THROTTLE_MAX - THROTTLE_CENTER))

        # Send to PCA ONLY if changed (reduces I2C bus congestion)
        if steer_us != last_steer_sent or throttle_us != last_throttle_sent:
            try:
                pca.set_us(cfg.STEERING_CHANNEL, steer_us)
                pca.set_us(cfg.THROTTLE_CHANNEL, throttle_us)
                last_steer_sent = steer_us
                last_throttle_sent = throttle_us
            except Exception as e:
                print(f"[PCA ERROR] {e}")

        # Recording & saving at TARGET_FPS
        now = time.time()
        if recording and now - last_save_time >= 1.0 / cfg.TARGET_FPS:
            if abs(steer_us - last_steer_rec) >= MIN_CHANGE_US or abs(throttle_us - last_throttle_rec) >= MIN_CHANGE_US:
                last_steer_rec, last_throttle_rec = steer_us, throttle_us
                rgb, depth_front = get_rgb_and_front_depth()
                if rgb is not None:
                    rgb_path = os.path.join(RUN_DIR, f"rgb_{frame_idx:05d}.png")
                    
                    # Prepare data for async save
                    row_data = [time.time(), steer_us, throttle_us,
                                pwm_to_norm(steer_us), pwm_to_norm(throttle_us),
                                depth_front, rgb_path]
                    
                    # Offload to thread
                    saver.save(save_task, rgb, rgb_path, row_data, writer, csv_file)
                    
                    frame_idx += 1
                    last_save_time = now
                    print(f"\rFrame {frame_idx:05d} | S {pwm_to_norm(steer_us):+0.3f} | "
                          f"T {pwm_to_norm(throttle_us):+0.3f} | Depth {depth_front:.2f}", end="")

        time.sleep(0.001) # Reduced sleep to 1ms for higher polling rate

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    saver.stop()
    try:
        csv_file.close()
    except Exception:
        pass
    neutralize()
    pygame.quit()
    try:
        realsense_full.stop_pipeline()
    except Exception as e:
        print(f"[RealSense stop error] {e}")
    print(f"\nDATA SAVED -> {RUN_DIR}")
