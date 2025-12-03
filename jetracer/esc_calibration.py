#!/usr/bin/env python3
"""
LaTrax + PCA9685 — One-shot calibration → ONLY prints the values you need forever
No code blocks, just clean numbers.
"""

import time
import sys
import smbus

# Config
PCA_ADDR = 0x40
STEERING_CHANNEL = 0
THROTTLE_CHANNEL = 1
PWM_FREQ = 60
MAX_TEST_THROTTLE = 0.4

class PCA9685:
    MODE1 = 0x00
    PRESCALE = 0xFE
    def __init__(self, bus=1, addr=PCA_ADDR):
        self.bus = smbus.SMBus(bus)
        self.addr = addr
        self._write(self.MODE1, 0x00)
        self.frequency = PWM_FREQ
    def _write(self, reg, val): self.bus.write_byte_data(self.addr, reg, val)
    def _read(self, reg): return self.bus.read_byte_data(self.addr, reg)
    def frequency():
        doc = "frequency getter/setter"
        def fget(self): return self._freq
        def fset(self, freq):
            self._freq = freq
            prescale = int(25000000 / 4096 / freq + 0.5)
            old = self._read(self.MODE1)
            self._write(self.MODE1, (old & 0x7F) | 0x10)
            self._write(self.PRESCALE, prescale)
            self._write(self.MODE1, old)
            time.sleep(0.005)
            self._write(self.MODE1, old | 0x80)
        return locals()
    frequency = property(**frequency())
    def set_pwm(self, ch, on, off):
        self.bus.write_byte_data(self.addr, 0x06 + 4*ch, on & 0xFF)
        self.bus.write_byte_data(self.addr, 0x07 + 4*ch, on >> 8)
        self.bus.write_byte_data(self.addr, 0x08 + 4*ch, off & 0xFF)
        self.bus.write_byte_data(self.addr, 0x09 + 4*ch, off >> 8)
    def set_pulse(self, ch, us):
        tick = int(4096 * us / (1_000_000 / self._freq))
        self.set_pwm(ch, 0, tick)

pca = PCA9685()
pca.frequency = PWM_FREQ

def throttle(us): pca.set_pulse(THROTTLE_CHANNEL, us); print(f"   Throttle → {us} µs")
def steering(us): pca.set_pulse(STEERING_CHANNEL, us);   print(f"   Steering → {us} µs")
def stop(): throttle(1500); steering(1500); time.sleep(0.5)

print("\n" + "="*60)
print("   FINAL CALIBRATION → ONLY VALUES (no code)")
print("="*60 + "\n")

input("Car on stand, battery OFF → press Enter when ready...")

# 1. ESC calibration
print("\n1. ESC CALIBRATION")
input("   → Plug battery in now → press Enter...")
throttle(1500); time.sleep(5)
input("   Hold EZ-Set button until RED blink → release → press Enter...")
print("   Choose mode → just remember it (no input needed here)")
throttle(2000); input("   Full forward → wait for 1 red flash → Enter...")
throttle(1000); input("   Full brake → wait for 2 red flashes → Enter...")
throttle(1500); print("   → Should be solid green now"); time.sleep(4)

# 2. Steering center
print("\n2. STEERING CENTER")
steering(1500)
while True:
    v = input("\n   Enter µs (e.g. 1485) or 'done': ").strip()
    if v.lower() in ['done','d','']: break
    try: steering(int(v))
    except: print("   Invalid")
CENTER = int(input("\n   Final center value [1500]: ") or 1500)
steering(CENTER)

# 3. Full left
print("\n3. FULL LEFT")
steering(CENTER - 500)
time.sleep(2)
input("   Adjust linkage if needed → press Enter when max left is reached...")
LEFT = int(input(f"   Enter left pulse [{CENTER-500}]: ") or (CENTER-500))
LEFT = max(1000, min(2000, LEFT))

# 4. Full right
print("\n4. FULL RIGHT")
steering(CENTER + 500)
time.sleep(2)
input("   Adjust linkage if needed → press Enter when max right is reached...")
RIGHT = int(input(f"   Enter right pulse [{CENTER+500}]: ") or (CENTER+500))
RIGHT = max(1000, min(2000, RIGHT))

steering(CENTER)

# 5. Quick test
print("\n5. Quick test sweep...")
for a in [-0.8, -0.4, 0, 0.4, 0.8]:
    steering(int(CENTER + a*(LEFT-CENTER) if a<0 else CENTER + a*(RIGHT-CENTER)))
    time.sleep(0.8)
steering(CENTER); throttle(0.3); time.sleep(3); throttle(0)

# ====================== FINAL VALUES ======================
print("\n" + "="*60)
print("             SAVE THESE VALUES FOREVER")
print("="*60)
print(f"STEERING_CENTER      = {CENTER} µs")
print(f"STEERING_FULL_LEFT   = {LEFT} µs")
print(f"STEERING_FULL_RIGHT  = {RIGHT} µs")
print()
print(f"THROTTLE_NEUTRAL     = 1500 µs")
print(f"THROTTLE_MAX_FORWARD = 2000 µs")
print(f"THROTTLE_MAX_REVERSE = 1000 µs")
print()
print("These 6 numbers are all you will ever need")
print("for teleop, DonkeyCar, ROS2, RL, or anything else.")
print("="*60)

stop()
