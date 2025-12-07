import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import time
import sys
from tqdm import tqdm
from datetime import timedelta

# ==================== CONFIGURATION ====================
DATASET_PATH = 'combined_augmented_dataset.csv'   # Change this if needed
IMG_HEIGHT = 120
IMG_WIDTH = 160
NUM_PIXELS = IMG_HEIGHT * IMG_WIDTH
BATCH_SIZE = 32
NUM_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VRAM_ALLOCATION = 0.5
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(VRAM_ALLOCATION, 0)
SAVE_DIR = 'checkpoints'
os.makedirs(SAVE_DIR, exist_ok=True)

# RGB column names
RGB_COLUMNS = [f'{c}{i}' for i in range(1, NUM_PIXELS + 1) for c in ['R', 'G', 'B']]

# Loss weights: steering is 5× more important
LOSS_WEIGHTS = torch.tensor([5.0, 1.0], device=DEVICE)  # [steer_norm, throttle_norm]

# ==================== LOGGING SETUP (console + file) ====================
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

original_stdout = sys.stdout
temp_log_path = 'temp_training_log.txt'
log_file = open(temp_log_path, 'w', encoding='utf-8')
sys.stdout = Tee(sys.stdout, log_file)

# ==================== PRINT SETUP INFO ====================
print("="*65)
print("                JETRACER BEHAVIORAL CLONING TRAINING")
print("="*65)
print(f"Dataset             : {DATASET_PATH}")
print(f"Device              : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU                 : {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory          : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Batch Size          : {BATCH_SIZE}")
print(f"Epochs              : {NUM_EPOCHS}")
print(f"Targets             : steer_norm, throttle_norm")
print(f"Loss Weighting      : Steering ×5 | Throttle ×1")
print(f"Output Activation   : Tanh() → [-1, 1]")
print("="*65 + "\n")

# ==================== DATASET ====================
class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        
        # Reshape flat RGB columns → (3, 120, 160)
        rgb_data = self.df[RGB_COLUMNS].values.astype(np.float32) / 255.0
        images = []
        for row in rgb_data:
            img = row.reshape(NUM_PIXELS, 3).reshape(IMG_HEIGHT, IMG_WIDTH, 3)
            img = np.transpose(img, (2, 0, 1))  # (C, H, W)
            images.append(img)
        self.images = np.array(images, dtype=np.float32)
        
        # Only normalized targets
        self.targets = self.df[['steer_norm', 'throttle_norm']].values.astype(np.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return torch.tensor(self.images[idx]), torch.tensor(self.targets[idx])

# ==================== LOAD DATA ====================
print(f"Loading dataset: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)
print(f"Total samples       : {len(df):,}")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
train_dataset = CustomDataset(train_df)
test_dataset = CustomDataset(test_df)

num_workers = 0 if os.name == 'nt' else 4
pin_memory = torch.cuda.is_available()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory)

print(f"Train batches       : {len(train_loader)}")
print(f"Test batches        : {len(test_loader)}\n")

# ==================== MODEL (ResNet18 + Tanh) ====================
class ControlModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-2])  # Up to last conv
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
            nn.Tanh()  # Ensures output in [-1, 1]
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

model = ControlModel().to(DEVICE)

# Save model architecture
arch_path = os.path.join(SAVE_DIR, 'model_architecture.txt')
with open(arch_path, 'w', encoding='utf-8') as f:
    f.write("ControlModel (ResNet18 backbone + Tanh head)\n")
    f.write("="*50 + "\n")
    f.write(str(model))
print(f"Model architecture saved → {arch_path}\n")

# Loss & optimizer
def weighted_mse_loss(pred, target):
    return torch.mean(LOSS_WEIGHTS * (pred - target) ** 2)

criterion = weighted_mse_loss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

# ==================== TRAINING ====================
history = {'epoch': [], 'train_loss': [], 'val_mae': [], 'val_r2': [], 'epoch_time': []}
best_val_mae = float('inf')
total_start_time = time.time()

print("Starting training...\n")

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()
    
    # Training
    model.train()
    train_loss = 0.0
    for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} [Train]", leave=False):
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for imgs, targets in tqdm(test_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS} [Val]", leave=False):
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.numpy())
    
    preds = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)
    
    val_mae = mean_absolute_error(targets, preds)
    val_r2 = r2_score(targets, preds, multioutput='uniform_average')
    steer_mae = mean_absolute_error(targets[:,0], preds[:,0])
    throttle_mae = mean_absolute_error(targets[:,1], preds[:,1])
    
    epoch_time = time.time() - epoch_start
    
    # Log
    history['epoch'].append(epoch)
    history['train_loss'].append(avg_train_loss)
    history['val_mae'].append(val_mae)
    history['val_r2'].append(val_r2)
    history['epoch_time'].append(epoch_time)
    
    print(f"\nEpoch {epoch:02d} | Time: {epoch_time:.1f}s")
    print(f"   Train Loss : {avg_train_loss:.5f}")
    print(f"   Val MAE    : {val_mae:.4f}  (Steer: {steer_mae:.4f} | Throttle: {throttle_mae:.4f})")
    print(f"   Val R²     : {val_r2:.4f}")
    
    scheduler.step(val_mae)
    
    # Save latest
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_mae': val_mae,
    }, os.path.join(SAVE_DIR, 'latest_model.pth'))
    
    # Save best
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_model.pth'))
        print(f"   New best model! Val MAE = {best_val_mae:.4f}")

# ==================== FINAL RESULTS ===
total_time = time.time() - total_start_time
print("\n" + "="*65)
print("TRAINING COMPLETED!")
print("="*65)
print(f"Total time          : {str(timedelta(seconds=int(total_time)))}")
print(f"Best Val MAE        : {best_val_mae:.4f}")
print(f"Final Val MAE       : {val_mae:.4f}")
print(f"Final Val R²        : {val_r2:.4f}")
print(f"Final Steer MAE     : {steer_mae:.4f}")
print(f"Final Throttle MAE     : {throttle_mae:.4f}")

# Save history & plot
pd.DataFrame(history).to_csv('training_history_normalized.csv', index=False)
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.plot(history['epoch'], history['train_loss'], 'o-'); plt.title('Train Loss'); plt.grid()
plt.subplot(1, 3, 2); plt.plot(history['epoch'], history['val_mae'], 'o-', color='orange'); plt.title('Val MAE'); plt.grid()
plt.subplot(1, 3, 3); plt.plot(history['epoch'], history['val_r2'], 'o-', color='green'); plt.title('Val R²'); plt.grid()
plt.tight_layout()
plt.savefig('training_curves_normalized.png', dpi=200)
plt.show()

# ==================== FINALIZE LOGGING ====================
sys.stdout = original_stdout
log_file.close()
final_log_path = os.path.join(SAVE_DIR, 'training_log.txt')
os.replace(temp_log_path, final_log_path)  # Atomic move

print(f"\nAll files saved in './{SAVE_DIR}/':")
print("   • best_model.pth")
print("   • latest_model.pth")
print("   • model_architecture.txt")
print("   • training_log.txt")
print("   • training_history_normalized.csv")
print("   • training_curves_normalized.png")

print(f"\nTraining log saved → {final_log_path}")
print("Ready for deployment!")
