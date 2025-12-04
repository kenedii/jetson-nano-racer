# scripts/run_autonomy_stub.py

from __future__ import annotations

import time

from car.car_interface import Car, CarConfig
from control.lane_follower_stub import LaneFollowerStub, LaneFollowerConfig
from control.safety import ImuSafetyMonitor
from sensors.realsense_imu import RealSenseD435i


def main() -> None:
    car = Car(CarConfig(max_throttle=0.4))
    cam = RealSenseD435i()
    safety = ImuSafetyMonitor()
    controller = LaneFollowerStub(LaneFollowerConfig())

    cam.start()
    print("Autonomy stub running. CTRL+C to exit.")

    try:
        while True:
            # Get sensor data
            rgbd = cam.get_rgbd(timeout_ms=100)
            imu = cam.get_imu_sample(timeout_ms=10)

            if imu is not None:
                safe = safety.update(imu)
            else:
                # If we lose IMU, fail safe
                safe = False

            if rgbd is None:
                # No camera frame -> stop
                car.stop()
                continue

            # Compute steering/throttle from RGBD
            steering = controller.compute_steering(rgbd)
            throttle = controller.compute_throttle(steering)

            if not safe:
                throttle = 0.0

            car.set_control(steering=steering, throttle=throttle)

            # Target ~20 Hz loop
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        car.stop()
        cam.stop()
        car.close()


if __name__ == "__main__":
    main()
