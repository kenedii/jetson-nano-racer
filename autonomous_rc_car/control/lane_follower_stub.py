# control/lane_follower_stub.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from car.car_interface import Car
from control.safety import ImuSafetyMonitor
from sensors.realsense_imu import ImuSample, RealSenseD435i, RgbdFrame


@dataclass
class LaneFollowerConfig:
    base_throttle: float = 0.25
    steering_kp: float = 0.8


class LaneFollowerStub:
    """
    Placeholder controller that demonstrates the structure:
    - You will replace `compute_steering` with your ML model or CV pipeline.
    """

    def __init__(self, cfg: LaneFollowerConfig | None = None) -> None:
        self.cfg = cfg or LaneFollowerConfig()

    def compute_steering(self, frame: RgbdFrame) -> float:
        """
        Right now, this just outputs 0 steering (go straight).
        Later, replace with:
          - lane segmentation / centerline detection, or
          - end-to-end network on RGB or RGB-D.
        """
        return 0.0

    def compute_throttle(self, steering: float) -> float:
        # Simple heuristic: slow down when steering hard
        return self.cfg.base_throttle * max(0.2, 1.0 - abs(steering))
