import time
from adafruit_servokit import ServoKit

# 16-channel PCA9685 at the default address 0x40
kit = ServoKit(channels=16, address=0x40)

# We'll treat CH0 as steering (S1 on mux)
steering = kit.continuous_servo[0]

print("Centering steering...")
steering.throttle = 0.0   # neutral pulse
time.sleep(2)

print("Small right turn...")
steering.throttle = 0.3   # adjust sign later if reversed
time.sleep(2)

print("Small left turn...")
steering.throttle = -0.3
time.sleep(2)

print("Back to center and exit.")
steering.throttle = 0.0
time.sleep(1)
