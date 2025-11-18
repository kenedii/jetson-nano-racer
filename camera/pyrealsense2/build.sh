#!/bin/bash
# install_realsense_nano.sh
# Fully working RealSense + pyrealsense2 installer for Jetson Nano 4GB (JetPack 4.6)
# Run with: 
# 1. chmod +x install_realsense_nano.sh
# 2. ./install_realsense_nano.sh

set -e  # stop on any error

echo "===================================================="
echo "Jetson Nano — Full RealSense + pyrealsense2 installer"
echo "===================================================="

# 1. Update system + install dependencies
echo "[1/7] Updating system and installing dependencies..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y git cmake build-essential \
    libssl-dev libusb-1.0-0-dev libudev-dev \
    pkg-config libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    python3-dev python3-pip python3-setuptools

# 2. Create swap (needed for build – Nano only has 4GB RAM)
echo "[2/7] Creating 6 GB swap..."
sudo fallocate -l 6G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo "/swapfile none swap sw 0 0" | sudo tee -a /etc/fstab

# 3. Clone RealSense library
echo "[3/7] Cloning librealsense..."
cd $HOME
rm -rf librealsense
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.55.1   # Known working version

# 4. Install udev rules
echo "[4/7] Installing udev rules (you may plug/unplug camera after this)"
sudo ./scripts/setup_udev_rules.sh

# 5. Build 
echo "[5/7] Starting build — this takes 45–70 min!"
mkdir -p build && cd build
cmake ../ \
    -DBUILD_PYTHON_BINDINGS=bool:true \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DCMAKE_BUILD_TYPE=Release \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_EXAMPLES=true \
    -DBUILD_GRAPHICAL_EXAMPLES=false

make -j2         
sudo make install

# 6. Add Python bindings permanently
echo "[6/7] Installing Python bindings..."
PYVER=$(python3 -c 'import sys; print("%d.%d" % (sys.version_info.major, sys.version_info.minor))')
sudo cp pyrealsense2*.so /usr/local/lib/python${PYVER}/dist-packages/
echo "export PYTHONPATH=\$PYTHONPATH:/usr/local/lib/python${PYVER}/dist-packages" >> $HOME/.bashrc

sudo ldconfig

# 7. Test installation
echo "[7/7] Testing Python install..."
python3 - << EOF
try:
    import pyrealsense2 as rs
    print("pyrealsense2 SUCCESSFULLY INSTALLED! Version:", rs.__version__)
except Exception as e:
    print("ERROR:", e)
EOF

echo "===================================================="
echo "ALL DONE! Now reboot with:"
echo "    sudo reboot"
echo "Then plug in your RealSense and run:"
echo "    python3 -c \"import pyrealsense2 as rs; print(rs.context().devices)\""
echo "===================================================="
