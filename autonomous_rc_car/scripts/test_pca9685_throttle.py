import time
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16, address=0x40)

steering = kit.continuous_servo[0]   # S1 → OUT1 → steering
throttle = kit.continuous_servo[1]   # S2 → OUT2 → ESC

# Always start safe
steering.throttle = 0.0
throttle.throttle = 0.0
time.sleep(2)

print("Arming ESC at neutral...")
time.sleep(2)

print("Very small forward throttle...")
throttle.throttle = 0.2   # if wheels spin backwards, change to -0.2
time.sleep(2)

print("Back to neutral...")
throttle.throttle = 0.0
time.sleep(2)

print("Very small reverse (if your ESC allows it)...")
throttle.throttle = -0.2
time.sleep(2)

print("Stop.")
throttle.throttle = 0.0
