# car/car_interface.py

from dataclasses import dataclass

try:
    from jetracer.nvidia_racecar import NvidiaRacecar
except ImportError as e:
    raise ImportError(
        "Could not import NvidiaRacecar from jetracer. "
        "Make sure the jetracer repo is installed on the Jetson."
    ) from e


@dataclass
class CarConfig:
    steering_gain: float = -0.65
    steering_offset: float = 0.0
    throttle_gain: float = 0.8  # global scaling for throttle
    max_throttle: float = 0.5   # safety clamp for early testing


class Car:
    """
    Thin wrapper around NvidiaRacecar that:
    - clamps inputs to [-1,1]
    - applies configurable gains
    - gives you stop() and simple speed limiting
    """

    def __init__(self, config: CarConfig | None = None) -> None:
        self.cfg = config or CarConfig()
        self._car = NvidiaRacecar()

        # Configure internal gains
        self._car.steering_gain = self.cfg.steering_gain
        self._car.steering_offset = self.cfg.steering_offset
        self._car.throttle_gain = self.cfg.throttle_gain

    @staticmethod
    def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, float(value)))

    def set_control(self, steering: float, throttle: float) -> None:
        """
        steering, throttle in [-1, 1].
        throttle is additionally clamped to +/- max_throttle for safety.
        """
        s = self._clamp(steering)
        t = self._clamp(throttle)
        t = max(-self.cfg.max_throttle, min(self.cfg.max_throttle, t))

        self._car.steering = s
        self._car.throttle = t

    def stop(self) -> None:
        self._car.throttle = 0.0

    def close(self) -> None:
        """Call before exiting, just to be safe."""
        self.stop()
