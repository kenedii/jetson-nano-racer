trt_optimize.py: Run TRT optimize with the model before running it on Jetson. Before TensorRT optimizations, Jetson Nano was only able to do 1 prediction every 3 seconds. After optimizations, the Jetson was able to do several predictions per second. Specify MODEL_ARCHITECTURE, PYTORCH_MODEL_PATH, and TRT_MODEL_PATH variables.


run_autonomous_resnet.py: Runs the model and controls the car by having the model's predictions send PWM signals using the PCA9685 to the Latrax ESC.
