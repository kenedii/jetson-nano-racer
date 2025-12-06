#!/usr/bin/env python3
"""
Network Xbox controller sender.
Run this on your laptop with the Xbox controller plugged in.
It reads the controller via pygame and sends normalized axes over UDP to the Jetson.

Packet format (JSON over UDP): {"s": steer_norm, "t": throttle_norm, "ts": epoch_seconds}
- steer_norm: -1.0 (left) .. +1.0 (right)
- throttle_norm: -1.0 (reverse) .. +1.0 (forward)
"""
import os
import time
import json
import socket
import pygame

# ===== CONFIGURE THESE =====
JETSON_HOST = os.environ.get("JETSON_HOST", "192.168.1.50")  # set to your Jetson IP
JETSON_PORT = int(os.environ.get("JETSON_PORT", "5007"))
# Hotspot-friendly: slightly lower rate to reduce bursts
SEND_HZ = 40.0
AXIS_DEADZONE = 0.03
STEERING_AXIS = 0          # left stick X
THROTTLE_AXIS = 1          # left stick Y (invert below)
USE_TRIGGERS_MODE3 = os.environ.get("USE_TRIGGERS_MODE3", "False").lower() in ("1", "true", "yes")
RIGHT_TRIGGER_AXIS = int(os.environ.get("RIGHT_TRIGGER_AXIS", "5"))
LEFT_TRIGGER_AXIS = int(os.environ.get("LEFT_TRIGGER_AXIS", "2"))
DEBUG_TRIGGERS = os.environ.get("DEBUG_TRIGGERS", "0").lower() in ("1", "true", "yes")


def apply_deadzone(v, dz=AXIS_DEADZONE):
    return 0.0 if abs(v) < dz else v


def main():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise SystemExit("No joystick detected. Plug in controller.")
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Joystick: {js.get_name()}")
    print(f"Sending to udp://{JETSON_HOST}:{JETSON_PORT} at ~{SEND_HZ} Hz")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interval = 1.0 / SEND_HZ
    next_send = time.time()

    counter = 0
    next_debug = time.time()
    try:
        while True:
            pygame.event.pump()
            # Steering
            raw_steer = -js.get_axis(STEERING_AXIS)  # invert to match car mapping
            steer = apply_deadzone(raw_steer)
            # Throttle: default to left stick Y; triggers optional via USE_TRIGGERS_MODE3
            if USE_TRIGGERS_MODE3:
                rt = (js.get_axis(RIGHT_TRIGGER_AXIS) + 1.0) / 2.0
                lt = (js.get_axis(LEFT_TRIGGER_AXIS) + 1.0) / 2.0
                if rt < AXIS_DEADZONE:
                    rt = 0.0
                if lt < AXIS_DEADZONE:
                    lt = 0.0
                throttle = rt - lt  # -1..1
                # if DEBUG_TRIGGERS and time.time() >= next_debug:
                #     print(f"[TRIG] rt_raw={js.get_axis(RIGHT_TRIGGER_AXIS):+.3f} lt_raw={js.get_axis(LEFT_TRIGGER_AXIS):+.3f} rt={rt:+.2f} lt={lt:+.2f} throttle={throttle:+.2f}")
                #     next_debug = time.time() + 0.5
            else:
                raw_thr = -js.get_axis(THROTTLE_AXIS)
                throttle = apply_deadzone(raw_thr)

            # Clamp
            steer = max(min(steer, 1.0), -1.0)
            throttle = max(min(throttle, 1.0), -1.0)

            now = time.time()
            if now >= next_send:
                pkt = {"s": steer, "t": throttle, "ts": now}
                try:
                    sock.sendto(json.dumps(pkt).encode("utf-8"), (JETSON_HOST, JETSON_PORT))
                    counter += 1
                    # print(f"[NET TX] #{counter} -> {JETSON_HOST}:{JETSON_PORT} s={steer:+.2f} t={throttle:+.2f} ts={now:.2f}")
                except Exception as e:
                    print(f"send error: {e}")
                next_send = now + interval

            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nStopping sender...")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
