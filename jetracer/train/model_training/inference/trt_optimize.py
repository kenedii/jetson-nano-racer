# trt_optimize.py
# Run this ONCE on your Jetson Nano to convert your trained model to TensorRT for faster inference. 

import os
import torch
import torch.nn as nn
from torchvision import models

# ------------------------------------------------------------
# USER CONFIG
# ------------------------------------------------------------
MODEL_ARCHITECTURE = "resnet18"   # Options: "resnet18", "resnet34", "resnet50", "resnet101"
# Paths
PYTORCH_MODEL_PATH = "checkpoints/model_5_resnet18/best_model.pth"
TRT_MODEL_PATH     = "checkpoints/model_5_resnet18/best_model_trt.pth"

# ------------------------------------------------------------
# 1. Install torch2trt automatically if not present
# ------------------------------------------------------------
try:
    from torch2trt import torch2trt, TRTModule
    print("[OK] torch2trt already installed")
except ImportError:
    print("[INFO] torch2trt not found → installing from source...")
    os.system("cd ~ && git clone https://github.com/NVIDIA-AI-IOT/torch2trt")
    os.system("cd ~/torch2trt && sudo python3 setup.py install")
    from torch2trt import torch2trt, TRTModule
    print("[OK] torch2trt installed successfully!")

# ------------------------------------------------------------
# 2. Exact same model class as in your training script — now dynamic
# ------------------------------------------------------------
class ControlModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Select backbone and feature size based on architecture
        if MODEL_ARCHITECTURE == "resnet18":
            backbone = models.resnet18(pretrained=False)
            feature_dim = 512
        elif MODEL_ARCHITECTURE == "resnet34":
            backbone = models.resnet34(pretrained=False)
            feature_dim = 512
        elif MODEL_ARCHITECTURE == "resnet50":
            backbone = models.resnet50(pretrained=False)
            feature_dim = 2048
        elif MODEL_ARCHITECTURE == "resnet101":
            backbone = models.resnet101(pretrained=False)
            feature_dim = 2048
        else:
            raise ValueError("Unsupported MODEL_ARCHITECTURE. Choose: resnet18, resnet34, resnet50, resnet101")

        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

# ------------------------------------------------------------
# 4. Load model + weights
# ------------------------------------------------------------
device = torch.device("cuda")

print(f"[INFO] Loading PyTorch model ({MODEL_ARCHITECTURE}) from: {PYTORCH_MODEL_PATH}")
model = ControlModel().to(device)

# Handle both plain state_dict and dict with 'model_state_dict'
ckpt = torch.load(PYTORCH_MODEL_PATH, map_location=device)
if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
    model.load_state_dict(ckpt['model_state_dict'])
else:
    model.load_state_dict(ckpt)

model.eval()
print("[OK] Model loaded and set to eval mode")

# ------------------------------------------------------------
# 5. Create dummy input (must match inference size)
# ------------------------------------------------------------
dummy_input = torch.ones((1, 3, 120, 160)).to(device)

# ------------------------------------------------------------
# 6. Convert to TensorRT (FP16 = huge speed boost on Nano)
# ------------------------------------------------------------
print("[INFO] Converting to TensorRT with FP16... (30–90 seconds)")
model_trt = torch2trt(
    model,
    [dummy_input],
    fp16_mode=True,           # ← critical for speed on Jetson Nano
    max_workspace_size=1 << 25,  # 512MB workspace
    use_onnx=False
)

# ------------------------------------------------------------
# 7. Save the optimized engine
# ------------------------------------------------------------
os.makedirs(os.path.dirname(TRT_MODEL_PATH), exist_ok=True)
torch.save(model_trt.state_dict(), TRT_MODEL_PATH)
print(f"[SUCCESS] TensorRT engine saved → {TRT_MODEL_PATH}")
print("")
print("You can now run your autonomous script — it will automatically use the fast TRT version!")
