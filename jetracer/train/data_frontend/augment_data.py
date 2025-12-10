# Augmentation script that performs augmentations on the images in the dataset using the CPU.
import pandas as pd
import numpy as np
import cv2
import random
import os

INPUT_CSV = "combined_dataset.csv"
OUTPUT_CSV = "combined_augmented_dataset.csv"
TEST_OUTPUT_DIR = "test_combination_pngs"
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

IMG_W = 160
IMG_H = 120
PIXELS = IMG_W * IMG_H


# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------

def flat_to_image(row):
    rgb = row.values.astype(np.uint8)
    return rgb.reshape((IMG_H, IMG_W, 3))


def image_to_flat(img):
    return img.reshape((-1,))


def augment_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-10, 10)) % 180
    hsv[..., 1] *= random.uniform(0.7, 1.3)
    hsv[..., 2] *= random.uniform(0.7, 1.3)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augment_blur_or_sharpen(img):
    if random.random() < 0.5:
        return cv2.GaussianBlur(img, (5, 5), 1.2)
    else:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)


def augment_noise(img):
    noisy = img.copy()
    num_pixels = int(PIXELS * random.uniform(0.05, 0.15))
    coords = np.random.randint(0, PIXELS, size=num_pixels)
    for c in coords:
        y = c // IMG_W
        x = c % IMG_W
        noise = np.random.normal(0, 25, size=3)
        noisy[y, x] = np.clip(noisy[y, x].astype(float) + noise, 0, 255)
    return noisy


def augment_flip(img, steer_norm):
    flipped = cv2.flip(img, 1)
    return flipped, -steer_norm


def augment_random_shadow(img):
    x1, y1 = random.randint(0, IMG_W), 0
    x2, y2 = random.randint(0, IMG_W), IMG_H
    mask = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    cv2.line(mask, (x1, y1), (x2, y2), 1.0, thickness=IMG_W)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    shadow_intensity = random.uniform(0.4, 0.8)
    shaded = img.astype(np.float32)
    shaded[:, :, :] *= (1 - shadow_intensity * mask[:, :, None])
    return np.clip(shaded, 0, 255).astype(np.uint8)


def augment_color_temperature(img):
    shift = random.randint(-30, 30)
    b, g, r = cv2.split(img.astype(np.int16))
    r = np.clip(r + shift, 0, 255)
    b = np.clip(b - shift, 0, 255)
    return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])


def depth_noise(depth):
    return depth + random.uniform(-5.0, 5.0)


# -----------------------------------------------------------
# NEW AUGMENTATIONS FOR COMBINATION MODE
# -----------------------------------------------------------

def augment_motion_blur(img):
    k = random.choice([3, 5, 7, 9])
    kernel = np.zeros((k, k))
    if random.random() < 0.5:
        kernel[int((k-1)/2), :] = np.ones(k)
    else:
        kernel[:, int((k-1)/2)] = np.ones(k)
    kernel /= k
    return cv2.filter2D(img, -1, kernel)


def augment_steering_jitter(steer_norm):
    jitter = random.uniform(-0.03, 0.03)
    return steer_norm + jitter


def augment_random_occlusion(img):
    occ_area = int(PIXELS * 0.06)
    max_w = int(IMG_W * 0.4)
    max_h = int(IMG_H * 0.4)
    w = random.randint(5, max_w)
    h = max(5, occ_area // max(w, 1))
    h = min(h, max_h)
    x = random.randint(0, IMG_W - w)
    y = random.randint(0, IMG_H - h)
    occluded = img.copy()
    color = random.randint(0, 50)
    occluded[y:y+h, x:x+w] = (color, color, color)
    return occluded


# -----------------------------------------------------------
# FULL COMBINATION AUGMENTATION
# -----------------------------------------------------------

def full_combination(img, steer_norm, depth_val):
    img = augment_color(img)
    img = augment_blur_or_sharpen(img)
    img = augment_noise(img)
    img = augment_random_shadow(img)
    img = augment_color_temperature(img)
    img = augment_motion_blur(img)
    img = augment_random_occlusion(img)
    img, steer_norm = augment_flip(img, steer_norm)
    steer_norm = augment_steering_jitter(steer_norm)
    depth_val = depth_noise(depth_val)
    return img, steer_norm, depth_val


# -----------------------------------------------------------
# CLEAN DEPTH FIELD
# -----------------------------------------------------------

def fix_depth_value(d):
    """Convert depth_front to centimeters consistently."""
    if pd.isna(d):
        return d

    # Already correct (50–2000 cm usual range)
    if d > 50:
        return d

    # Meter-range values (1–10m → 100–1000 cm)
    if 1.0 < d < 10.0:
        return d * 100.0

    # cm/1000 values from RealSense (0.05–2.0 → 50–2000 cm)
    if 0.05 < d < 2.0:
        return d * 1000.0

    return d


# -----------------------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------------------

df = pd.read_csv(INPUT_CSV)

# Fix depth before augmentation
df["depth_front"] = df["depth_front"].apply(fix_depth_value)

print("=== DEPTH CLEANUP COMPLETE ===")
print(df["depth_front"].describe())

# -----------------------------------------------------------
# CORRELATION REPORT
# -----------------------------------------------------------

cols_to_corr = ["depth_front", "steer_us", "throttle_us", "steer_norm", "throttle_norm"]
print("\n=== CORRELATION WITH DEPTH ===")
print(df[cols_to_corr].corr())


# -----------------------------------------------------------
# AUGMENTATION LOOP
# -----------------------------------------------------------

rows = []
combo_png_saved = 0

for idx, row in df.iterrows():
    base_row = row.copy()
    steer_norm = row["steer_norm"]
    depth_val = row["depth_front"]

    img = flat_to_image(row.iloc[6:])

    # Keep original
    rows.append(base_row)

    # 25% COLOR
    if random.random() < 0.25:
        cimg = augment_color(img)
        new = base_row.copy()
        new.iloc[6:] = image_to_flat(cimg)
        rows.append(new)

    # 25% BLUR / SHARPEN
    if random.random() < 0.25:
        bimg = augment_blur_or_sharpen(img)
        new = base_row.copy()
        new.iloc[6:] = image_to_flat(bimg)
        rows.append(new)

    # 25% NOISE
    if random.random() < 0.25:
        nimg = augment_noise(img)
        new = base_row.copy()
        new.iloc[6:] = image_to_flat(nimg)
        rows.append(new)

    # 50% FLIP
    if random.random() < 0.50:
        fimg, s = augment_flip(img, steer_norm)
        new = base_row.copy()
        new["steer_norm"] = s
        new.iloc[6:] = image_to_flat(fimg)
        rows.append(new)

    # FULL COMBINATION × 4
    for _ in range(4):
        cimg, s2, d2 = full_combination(img.copy(), steer_norm, depth_val)
        new = base_row.copy()
        new["steer_norm"] = s2
        new["depth_front"] = d2
        new.iloc[6:] = image_to_flat(cimg)
        rows.append(new)

        if combo_png_saved < 3:
            png_path = os.path.join(TEST_OUTPUT_DIR, f"combo_sample_{combo_png_saved+1}.png")
            cv2.imwrite(png_path, cv2.cvtColor(cimg, cv2.COLOR_RGB2BGR))
            combo_png_saved += 1


# SAVE OUTPUT
out_df = pd.DataFrame(rows, columns=df.columns)
out_df.to_csv(OUTPUT_CSV, index=False)

print("\nAugmentation complete!")
print(f"Original rows: {len(df)}")
print(f"Augmented rows: {len(out_df)}")
print("Saved 3 combination PNG samples to:", TEST_OUTPUT_DIR)
