# sensors/realsense_imu.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pyrealsense2 as rs


@dataclass
class ImuSample:
    t: float                 # seconds, device timestamp
    accel: np.ndarray        # shape (3,), m/s^2
    gyro: np.ndarray         # shape (3,), rad/s (if configured that way)


@dataclass
class RgbdFrame:
    t: float                 # seconds, device timestamp (color)
    color: np.ndarray        # H x W x 3, uint8 BGR
    depth: np.ndarray        # H x W, uint16 depth in millimeters


class RealSenseD435i:
    """
    Simple RGB-D + IMU wrapper.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        enable_color: bool = True,
        enable_depth: bool = True,
        enable_imu: bool = True,
    ) -> None:
        self._pipeline = rs.pipeline()
        self._cfg = rs.config()

        if enable_depth:
            self._cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        if enable_color:
            self._cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_imu:
            # IMU streams do not need width/height, just fps
            self._cfg.enable_stream(rs.stream.accel)
            self._cfg.enable_stream(rs.stream.gyro)

        self._profile = None
        self._align = rs.align(rs.stream.color) if enable_depth and enable_color else None

    def start(self) -> None:
        self._profile = self._pipeline.start(self._cfg)

    def stop(self) -> None:
        self._pipeline.stop()

    def get_rgbd(self, timeout_ms: int = 1000) -> Optional[RgbdFrame]:
        """
        Blocks until a frameset is available or timeout.
        Returns None on timeout.
        """
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
        except RuntimeError:
            return None

        if self._align is not None:
            frames = self._align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None

        color = np.asarray(color_frame.get_data())  # H x W x 3, BGR
        depth = np.asarray(depth_frame.get_data())  # H x W

        # Use color frame timestamp as representative
        ts = color_frame.get_timestamp() / 1000.0  # ms -> s

        return RgbdFrame(t=ts, color=color, depth=depth)

    def get_imu_sample(self, timeout_ms: int = 100) -> Optional[ImuSample]:
        """
        Poll for the next IMU sample (both accel and gyro).
        This is a simple implementation: it waits for a frameset and
        tries to extract accel + gyro from it.
        """
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms)
        except RuntimeError:
            return None

        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame = frames.first_or_default(rs.stream.gyro)

        if not accel_frame or not gyro_frame:
            return None

        accel = accel_frame.as_motion_frame().get_motion_data()
        gyro = gyro_frame.as_motion_frame().get_motion_data()

        # convert to numpy arrays
        accel_vec = np.array([accel.x, accel.y, accel.z], dtype=np.float32)
        gyro_vec = np.array([gyro.x, gyro.y, gyro.z], dtype=np.float32)

        # use gyro timestamp (they are very high rate; exact sync isn't critical for safety logic)
        ts = gyro_frame.get_timestamp() / 1000.0

        return ImuSample(t=ts, accel=accel_vec, gyro=gyro_vec)
