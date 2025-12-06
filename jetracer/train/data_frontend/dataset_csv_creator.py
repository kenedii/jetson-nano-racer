import os
import csv
from PIL import Image

# Configuration
root_dir = 'data'  # The folder containing all 'run_...' subfolders
output_csv = 'combined_dataset.csv'  # The output file name

# Function to extract image filename from path (ignoring prefix)
def get_image_filename(rgb_path):
    return os.path.basename(rgb_path)

# Find all subdirectories in root_dir that start with 'run_'
subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('run_')]

# Collect all rows from all CSVs to determine image size first (assume all images same size)
num_pixels = None
sample_image_path = None

# Find a sample image to determine size
for run_dir in subdirs:
    csv_path = os.path.join(root_dir, run_dir, 'dataset.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 7:
                    rgb_path = row[6]
                    image_filename = get_image_filename(rgb_path)
                    image_path = os.path.join(root_dir, run_dir, image_filename)
                    if os.path.exists(image_path):
                        sample_image_path = image_path
                        break
        if sample_image_path:
            break

if not sample_image_path:
    raise FileNotFoundError("No valid image found to determine size.")

# Load sample image to get dimensions
img = Image.open(sample_image_path).convert('RGB')
width, height = img.size
num_pixels = width * height

# Generate header
header = ['timestamp', 'steer_us', 'throttle_us', 'steer_norm', 'throttle_norm', 'depth_front']
for i in range(1, num_pixels + 1):
    header.extend([f'R{i}', f'G{i}', f'B{i}'])

# Now collect all data rows
all_rows = []

for run_dir in subdirs:
    csv_path = os.path.join(root_dir, run_dir, 'dataset.csv')
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 7:
                    timestamp, steer_us, throttle_us, steer_norm, throttle_norm, depth_front, rgb_path = row[:7]
                    image_filename = get_image_filename(rgb_path)
                    image_path = os.path.join(root_dir, run_dir, image_filename)
                    if os.path.exists(image_path):
                        img = Image.open(image_path).convert('RGB')
                        if img.size != (width, height):
                            print(f"Warning: Image {image_path} has different size {img.size}, skipping.")
                            continue
                        pixels = list(img.getdata())  # List of (r, g, b) tuples
                        flat_pixels = [str(val) for pixel in pixels for val in pixel]  # Flatten to ['r1', 'g1', 'b1', ...]
                        data_row = [timestamp, steer_us, throttle_us, steer_norm, throttle_norm, depth_front] + flat_pixels
                        all_rows.append(data_row)

# Write to output CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(all_rows)

print(f"Combined dataset saved to {output_csv}")
