#!/usr/bin/env python3
"""
IMU-based crash & rollover detector for JetRacer (Latrax Rally + RealSense D435i).

- Uses D435i's internal IMU via pyrealsense2.
- Monitors tilt and acceleration magnitude.
- If a crash/rollover is detected, sets car.throttle = 0.0 and latches into a safe state.

Integrate this into your main control loop by treating
    detector.apply_safety(desired_throttle)
as the final value you send to car.throttle.
"""

import time

import numpy as np
import pyrealsense2 as rs
from jetracer.nvidia_racecar import NvidiaRacecar


class ImuCrashRolloverDetector:
    def __init__(
        self,
        rollover_angle_deg: float = 70.0,
        crash_g_mult: float = 2.5,
        consecutive_imu_samples: int = 3,
    ):
        """
        rollover_angle_deg:
            Angle between gravity-at-start and current gravity estimate above which
            we consider the car 'rolled'.

        crash_g_mult:
            Accel magnitude threshold, in multiples of 'g' measured at startup.

        consecutive_imu_samples:
            Number of consecutive "bad" IMU samples required before latching.
        """
        self.rollover_angle_deg = rollover_angle_deg
        self.crash_g_mult = crash_g_mult
        self.consecutive_imu_samples = consecutive_imu_samples

        self.g_dir = None
        self.g_mag = None

        self.latched = False
        self.last_reason = ""
        self._bad_count = 0

    def calibrate(self, pipeline, seconds: float = 2.0):
        """
        Estimate gravity direction and magnitude from the accelerometer
        while the car is at rest in its normal upright pose.
        """
        print("IMU safety: calibrating baseline gravity; keep the car still...")
        samples = []
        start = time.time()
        while time.time() - start < seconds:
            frames = pipeline.wait_for_frames()
            accel_frame = frames.first_or_default(rs.stream.accel)
            if not accel_frame:
                continue
            d = accel_frame.as_motion_frame().get_motion_data()
            samples.append(np.array([d.x, d.y, d.z]))

        if not samples:
            raise RuntimeError("No IMU samples received during calibration")

        mean_accel = np.mean(samples, axis=0)
        mag = float(np.linalg.norm(mean_accel))
        if mag < 1e-6:
            raise RuntimeError("IMU calibration failed: gravity magnitude is ~0")

        self.g_mag = mag
        self.g_dir = mean_accel / mag

        print(f"IMU safety: baseline |g| ≈ {self.g_mag:.3f}, dir = {self.g_dir}")

    def _tilt_and_mag(self, accel_vec):
        """Return (tilt_deg, |a|) relative to the calibrated gravity direction."""
        if self.g_dir is None or self.g_mag is None:
            raise RuntimeError("Detector not calibrated; call calibrate() first")

        a = np.array(accel_vec, dtype=float)
        amag = float(np.linalg.norm(a))
        if amag < 1e-6:
            return 0.0, amag

        a_dir = a / amag
        cosang = float(np.clip(np.dot(a_dir, self.g_dir), -1.0, 1.0))
        tilt_rad = np.arccos(cosang)
        tilt_deg = float(np.degrees(tilt_rad))
        return tilt_deg, amag

    def update(self, accel_vec):
        """
        Update state from a new accelerometer sample.

        Returns (latched, reason_string).
        """
        tilt_deg, amag = self._tilt_and_mag(accel_vec)

        rollover = tilt_deg > self.rollover_angle_deg
        crash = amag > self.crash_g_mult * self.g_mag

        if rollover or crash:
            self._bad_count += 1
        else:
            # small hysteresis so brief spikes don't immediately unlatch
            self._bad_count = max(0, self._bad_count - 1)

        if not self.latched and self._bad_count >= self.consecutive_imu_samples:
            self.latched = True
            if rollover and crash:
                self.last_reason = (
                    f"tilt {tilt_deg:.1f}° and |a|={amag:.2f} > "
                    f"{self.crash_g_mult:.1f} g"
                )
            elif rollover:
                self.last_reason = (
                    f"tilt {tilt_deg:.1f}° exceeds {self.rollover_angle_deg:.1f}°"
                )
            else:
                self.last_reason = (
                    f"|a|={amag:.2f} > {self.crash_g_mult:.1f} × g ({self.g_mag:.2f})"
                )

        return self.latched, self.last_reason

    def apply_safety(self, desired_throttle: float) -> float:
        """
        Given a controller's desired throttle in [-1, 1], return a safe throttle.

        If a crash/rollover has been detected, this will return 0.0.
        """
        if self.latched:
            return 0.0
        return float(desired_throttle)


def main():
    # Connect to JetRacer car (Latrax Rally wiring via PCA9685 and multiplexer)
    car = NvidiaRacecar()
    car.throttle = 0.0
    # Tune as needed; gain scales the output to the ESC
    car.throttle_gain = 0.8
    print("Initialized NvidiaRacecar; throttle is at 0.0")

    # Set up RealSense IMU-only pipeline (no color/depth here)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 63)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    print("Starting RealSense IMU pipeline...")
    pipeline.start(config)

    detector = ImuCrashRolloverDetector(
        rollover_angle_deg=70.0,
        crash_g_mult=2.5,
        consecutive_imu_samples=3,
    )

    # Calibrate gravity baseline
    detector.calibrate(pipeline, seconds=2.0)

    print(
        "IMU safety: running. Integrate this loop with your main control.\n"
        "Press Ctrl+C to exit."
    )

    try:
        while True:
            frames = pipeline.wait_for_frames()
            accel_frame = frames.first_or_default(rs.stream.accel)
            if not accel_frame:
                continue

            d = accel_frame.as_motion_frame().get_motion_data()
            accel = np.array([d.x, d.y, d.z])

            latched, reason = detector.update(accel)

            if latched:
                if car.throttle != 0.0:
                    print(f"*** SAFETY STOP: {reason} ***")
                car.throttle = 0.0
            else:
                # In your real driving code, compute desired_throttle here
                # (e.g., from your lane-following / obstacle avoidance controller),
                # then wrap it with apply_safety():
                desired_throttle = 0.0  # keep 0 here so this demo doesn't drive the car
                safe_throttle = detector.apply_safety(desired_throttle)
                car.throttle = safe_throttle

            # Sleep a bit to reduce CPU usage; IMU runs much faster than this.
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExiting; stopping car and pipeline...")
    finally:
        car.throttle = 0.0
        pipeline.stop()


if __name__ == "__main__":
    main()
