# scripts/test_realsense_stream.py

import time

import cv2
from sensors.realsense_imu import RealSenseD435i


def main() -> None:
    cam = RealSenseD435i()
    cam.start()
    print("Started RealSense. Press 'q' in the window to exit.")

    try:
        last_imu = None

        while True:
            rgbd = cam.get_rgbd()
            imu = cam.get_imu_sample()

            if rgbd is not None:
                img = rgbd.color
                # Show a small overlay of depth center value
                h, w, _ = img.shape
                d = rgbd.depth[h // 2, w // 2]
                cv2.putText(
                    img,
                    f"depth center: {d} mm",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("RealSense RGB", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if imu is not None:
                last_imu = imu

            if last_imu is not None and int(time.time()) % 1 == 0:
                print(
                    f"IMU t={last_imu.t:.3f} accel={last_imu.accel} m/s^2 "
                    f"gyro={last_imu.gyro} rad/s",
                    end="\r",
                )

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
