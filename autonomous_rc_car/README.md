# Autonomous RC Car with IMU-Enhanced Autonomy — Project Snapshot

This archive is a snapshot of the project context collected so far for the **Autonomous RC Car with IMU-Enhanced Autonomy**.

It is based on the NVIDIA JetRacer project:

- GitHub: https://github.com/NVIDIA-AI-IOT/jetracer
- Hardware setup (Latrax Rally): https://github.com/NVIDIA-AI-IOT/jetracer/blob/master/docs/latrax/hardware_setup.md
- Software setup: https://github.com/NVIDIA-AI-IOT/jetracer/blob/master/docs/software_setup.md

## Custom Hardware Choices

Compared to the reference JetRacer build, this project uses:

- **NVIDIA Jetson Nano (4GB)**
- **Intel RealSense D435i depth camera with IMU** (replacing the CSI camera modules)
- **TP-Link TL-WN725N USB WiFi adapter**
- **Latrax Rally 1/18 RC car (Traxxas 75054-5)**

Key datasheets and manuals are included in the `docs/` folder of this snapshot.

## RC Electronics & Control Path

### RC Receiver → Pololu RC Servo Multiplexer (item 2806)

Wiring as described by the user:

- Traxxas receiver **Channel 1** → Multiplexer **M1** (steering input from the radio)
- Traxxas receiver **Channel 2** → Multiplexer **M2** (throttle/ESC input from the radio)
- Traxxas receiver **Channel 3** → Multiplexer **SEL** (selection channel: RC vs Jetson control)

Outputs:

- Steering **servo** → Multiplexer **OUT1**
- **ESC** → Multiplexer **OUT2**

When SEL is set appropriately, you can switch between manual radio control and computer (Jetson) control.

### Jetson Nano → Adafruit 16‑Channel PWM/Servo Driver (PCA9685, product 815)

- Servo driver **Channel 0** → Multiplexer **S1** (steering signal from Jetson)
- Servo driver **Channel 1** → Multiplexer **S2** (throttle/ESC signal from Jetson)

I²C wiring between Jetson Nano and the servo driver:

- Servo driver **VCC** → Jetson Nano 3.3 V **Pin 1 or 17** (user specified: *Jetson Nano Pin 3.3V*)
- Servo driver **SDA** → Jetson Nano **Pin 3** (I2C SDA)
- Servo driver **SCL** → Jetson Nano **Pin 5** (I2C SCL)
- Servo driver **GND** → Jetson Nano **GND** (e.g., Pin 6, 9, 14, 20, 25, 30, 34, or 39)

> Note: The PCA9685 board usually also has a separate high‑current **V+** rail to power servos; in this snapshot we only record the logic‑side wiring you described. High‑current power for servos should come from an appropriate BEC/receiver pack, *not* from the Jetson’s 3.3 V pin.

### Perception Stack (planned)

- Intel RealSense **D435i** providing:
  - RGB + depth images
  - Onboard **BMI055/BMI085 IMU** data for 6‑DoF motion sensing
- IMU and depth data will be fused on the Jetson Nano to support:
  - Lane following
  - Obstacle detection / avoidance
  - Drift detection & stability control
  - Crash / rollover detection and emergency stop
  - Incline / terrain awareness and throttle modulation

See `docs/Project Management Plan Autonomous RC Car with IMU.docx` for the full project management plan, schedule, and risk register.

## Files in This Snapshot

- `README.md` — This summary
- `wiring_notes.txt` — Plain‑text version of the hardware wiring information
- `docs/` — Datasheets and manuals:
  - Intel RealSense D400 Series datasheet (includes D435i)
  - Jetson Nano Developer Kit User Guide
  - BMI055 IMU datasheet
  - LaTrax / Traxxas 75054‑5 owner’s manual
  - Project Management Plan for this capstone project

Snapshot generated on: 2025-12-04T04:09:20.827608Z
