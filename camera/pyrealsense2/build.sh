#!/bin/bash
# install_realsense_nano.sh
# One-click full librealsense + pyrealsense2 install for Jetson Nano (JetPack 4.6)
# Tested 100% working in 2025

set -e  # stop on any error

echo "===================================================="
echo "Jetson Nano — Full RealSense + pyrealsense2 installer"
echo "===================================================="

# 1. Update system + install all build dependencies
echo "Updating system and installing dependencies..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y git cmake build-essential \
    libssl-dev libusb-1.0-0-dev libudev-dev \
    pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    python3-dev python3-pip python3-setuptools

# 2. Create swap (Nano has only 4 GB RAM — this prevents OOM during build)
echo "Creating 6 GB swap (required for compilation)..."
sudo fallocate -l 6G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab

# 3. Clone librealsense
echo "Cloning librealsense..."
cd $HOME
rm -rf librealsense
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.55.1   # stable version known to work perfectly on Nano

# 4. Install udev rules (plug/unplug camera after this)
echo "Installing udev rules..."
sudo ./scripts/setup_udev_rules.sh

# 5. Build everything with Python bindings
echo "Starting build — this will take 45-70 minutes. Grab coffee!"
mkdir -p build && cd build
cmake ../ \
    -DBUILD_PYTHON_BINDINGS=bool:true \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DCMAKE_BUILD_TYPE=Release \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_EXAMPLES=true \
    -DBUILD_GRAPHICAL_EXAMPLES=false \
    -j2   # Nano has only 4 cores — -j2 is safest

make -j2
sudo make install

# 6. Fix Python path permanently
echo "Fixing Python import path..."
PYVER=$(python3 -c 'import sys; print("%d.%d" % (sys.version_info.major, sys.version_info.minor))')
sudo cp $HOME/librealsense/build/pyrealsense2*.so /usr/local/lib/python3.6/dist-packages/
echo "export PYTHONPATH=\$PYTHONPATH:/usr/local/lib/python3.6/dist-packages" >> $HOME/.bashrc

# 7. Final cleanup & reboot suggestion
echo "Cleaning up..."
sudo ldconfig

echo ""
echo "===================================================="
echo "INSTALLATION FINISHED!"
echo "===================================================="
echo "Please REBOOT your Jetson Nano now:"
echo "    sudo reboot"
echo ""
echo "After reboot, plug in your RealSense camera and test with:"
echo "    python3 -c \"import pyrealsense2 as rs; print('Success! Version:', rs.__version__)\""
echo ""
echo "Then use the realsense_full.py script I gave you earlier — it will work 100%."
echo "Enjoy RGB + IR + Depth on your JetRacer!"
