#!/usr/bin/env python3
"""
Simple RealSense D435i + IMU hello world for Jetson Nano.

- Streams color + depth at modest resolution.
- Streams accelerometer + gyro from the built-in IMU.
- Prints a few samples to the terminal.
"""

import time

import numpy as np
import pyrealsense2 as rs


def main():
    # Create pipeline and config
    pipeline = rs.pipeline()
    config = rs.config()

    # Use modest resolutions / frame rates to be gentle on Jetson Nano
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

    # IMU streams (accelerometer + gyro)
    # Typical D435i rates are ~63 Hz (accel) and ~200/400 Hz (gyro)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    print("Starting RealSense pipeline...")
    profile = pipeline.start(config)

    # Depth scale is handy if you want distances in meters
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale:.6f} meters per unit")

    last_print = 0.0

    try:
        while True:
            # Wait for the next set of frames
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            accel_frame = frames.first_or_default(rs.stream.accel)
            gyro_frame = frames.first_or_default(rs.stream.gyro)

            if not depth_frame or not color_frame:
                continue

            # Throttle prints so we don't spam the terminal
            now = time.time()
            if now - last_print < 0.25:
                continue
            last_print = now

            # Extract a single depth pixel in the center of the image
            depth_width = depth_frame.get_width()
            depth_height = depth_frame.get_height()
            center_distance = depth_frame.get_distance(
                depth_width // 2, depth_height // 2
            )

            # IMU data (accelerometer ≈ m/s^2, gyro ≈ rad/s)
            accel = gyro = None

            if accel_frame:
                accel_data = accel_frame.as_motion_frame().get_motion_data()
                accel = np.array([accel_data.x, accel_data.y, accel_data.z])

            if gyro_frame:
                gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                gyro = np.array([gyro_data.x, gyro_data.y, gyro_data.z])

            print("---- RealSense D435i sample ----")
            print(f"Depth center distance: {center_distance:.3f} m")

            if accel is not None:
                print(
                    f"Accel (m/s^2): x={accel[0]: .3f}, "
                    f"y={accel[1]: .3f}, z={accel[2]: .3f}"
                )
            if gyro is not None:
                print(
                    f"Gyro (rad/s):  x={gyro[0]: .3f}, "
                    f"y={gyro[1]: .3f}, z={gyro[2]: .3f}"
                )

    except KeyboardInterrupt:
        print("\nStopping RealSense pipeline...")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
