# scripts/test_car_motion.py

import time
from car.car_interface import Car, CarConfig


def main() -> None:
    car = Car(CarConfig())
    try:
        print("Center steering, zero throttle")
        car.set_control(0.0, 0.0)
        time.sleep(1.0)

        print("Sweep steering left/right")
        for s in [0.5, -0.5, 0.0]:
            print(f" steering={s}")
            car.set_control(s, 0.0)
            time.sleep(1.0)

        print("Short gentle forward throttle test")
        car.set_control(0.0, 0.2)
        time.sleep(1.0)
        car.stop()
    finally:
        car.close()


if __name__ == "__main__":
    main()
