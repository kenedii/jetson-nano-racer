# control/safety.py

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, sqrt
from typing import Tuple

import numpy as np

from sensors.realsense_imu import ImuSample


@dataclass
class SafetyConfig:
    impact_g_threshold: float = 3.0     # ~3g
    tilt_deg_threshold: float = 60.0    # more than 60° pitch/roll -> unsafe
    decay_time_s: float = 0.5           # how long to hold unsafe after event


def accel_to_tilt_deg(accel: np.ndarray) -> Tuple[float, float]:
    """
    Rough estimate of roll and pitch from accelerometer (assuming z points up when level).
    Returns (roll_deg, pitch_deg).
    """
    ax, ay, az = accel
    g = sqrt(ax * ax + ay * ay + az * az) + 1e-6
    # Normalize
    ax /= g
    ay /= g
    az /= g

    pitch = atan2(-ax, sqrt(ay * ay + az * az))
    roll = atan2(ay, az)

    return (roll * 180.0 / 3.14159, pitch * 180.0 / 3.14159)


class ImuSafetyMonitor:
    """
    Simple IMU-based safety monitor:
    - Detects impacts via accel magnitude
    - Detects rollover via tilt angle
    - Exposes a binary 'safe' flag you can use to gate throttle.
    """

    def __init__(self, cfg: SafetyConfig | None = None) -> None:
        self.cfg = cfg or SafetyConfig()
        self._unsafe_until: float = 0.0

    def update(self, sample: ImuSample) -> bool:
        """
        Update with latest IMU sample.
        Returns current safety flag: True if safe, False if unsafe.
        """
        t = sample.t

        # Accel magnitude in g (D435i reports m/s^2)
        acc_mag = np.linalg.norm(sample.accel) / 9.80665

        roll_deg, pitch_deg = accel_to_tilt_deg(sample.accel)

        impact = acc_mag > self.cfg.impact_g_threshold
        tilted = (
            abs(roll_deg) > self.cfg.tilt_deg_threshold
            or abs(pitch_deg) > self.cfg.tilt_deg_threshold
        )

        if impact or tilted:
            # Extend unsafe window
            self._unsafe_until = max(self._unsafe_until, t + self.cfg.decay_time_s)

        safe = t >= self._unsafe_until
        return safe
