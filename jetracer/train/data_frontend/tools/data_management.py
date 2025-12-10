"""
RC Car Dataset Viewer & Cleaner + Combined CSV Creator
- View & clean runs
- Delete frames
- Bulk remove zero-throttle
- Create full pixel-flattened combined_dataset.csv from all runs
"""

import os
import json
import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
import csv

st.set_page_config(page_title="RC Car Dataset Viewer & Cleaner", layout="wide")

# -------------------------
# Session State
# -------------------------
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "current_folder" not in st.session_state:
    st.session_state.current_folder = None

THUMB_WINDOW = 10

# -------------------------
# Smart Path Resolver (fixes runs_rgb_depth/...)
# -------------------------
def resolve_image_path(root_folder: str, raw_path: str) -> str:
    if not raw_path or str(raw_path).strip() in ["", "nan", "None"]:
        return ""

    raw = str(raw_path).strip().replace("\\", "/")
    prefixes = ["runs_rgb_depth/", "runs_rgb_depth/run_", "rgb_depth/", "rgb/", "../rgb/"]
    cleaned = raw
    for p in prefixes:
        if cleaned.startswith(p):
            cleaned = cleaned[len(p):]

    bases = [
        root_folder,
        str(Path(root_folder).parent),
        str(Path(root_folder).parent.parent),
        str(Path(root_folder).parent / "rgb"),
        str(Path(root_folder).parent / "runs_rgb_depth"),
    ]

    candidates = []
    for base in bases:
        if os.path.isdir(base):
            p = os.path.normpath(os.path.join(base, cleaned))
            candidates.extend([p, p.replace(".png", ".jpg"), p.replace(".jpg", ".png")])

    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0] if candidates else os.path.join(root_folder, raw)

# -------------------------
# Combined CSV Creator (your script embedded)
# -------------------------
def create_combined_csv(root_dir: str):
    output_csv = os.path.join(root_dir, "combined_dataset.csv")
    
    if os.path.exists(output_csv):
        st.warning(f"combined_dataset.csv already exists. Overwriting...")

    with st.spinner("Scanning runs and finding sample image..."):
        subdirs = [d for d in os.listdir(root_dir) 
                  if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('run_')]
        if not subdirs:
            st.error("No run_* folders found.")
            return

        sample_image_path = None
        for run_dir in subdirs:
            run_path = os.path.join(root_dir, run_dir)
            csv_path = os.path.join(run_path, "dataset.csv")
            if not os.path.exists(csv_path):
                continue
            df_temp = pd.read_csv(csv_path)
            img_col = next((c for c in df_temp.columns if any(x in c.lower() for x in ["rgb", "image", "path"])), None)
            if img_col is None:
                continue
            for val in df_temp[img_col]:
                if pd.notna(val):
                    full_path = resolve_image_path(run_path, val)
                    if os.path.exists(full_path):
                        sample_image_path = full_path
                        break
            if sample_image_path:
                break

        if not sample_image_path:
            st.error("Could not find any valid image in any run.")
            return

        img = Image.open(sample_image_path).convert('RGB')
        width, height = img.size
        num_pixels = width * height

    # Build header
    header = ['timestamp', 'steer_us', 'throttle_us', 'steer_norm', 'throttle_norm', 'depth_front']
    for i in range(1, num_pixels + 1):
        header.extend([f'R{i}', f'G{i}', f'B{i}'])

    all_rows = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    total_runs = len(subdirs)
    run_count = 0

    for run_dir in subdirs:
        run_count += 1
        status_text.text(f"Processing {run_dir} ({run_count}/{total_runs})...")
        run_path = os.path.join(root_dir, run_dir)
        csv_path = os.path.join(run_path, "dataset.csv")
        
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        img_col = next((c for c in df.columns if any(x in c.lower() for x in ["rgb", "image", "path"])), None)
        if img_col is None:
            continue

        for idx, row in df.iterrows():
            rgb_path = row[img_col] if img_col in row and pd.notna(row[img_col]) else None
            if not rgb_path:
                continue

            full_img_path = resolve_image_path(run_path, rgb_path)
            if not os.path.exists(full_img_path):
                continue

            try:
                img = Image.open(full_img_path).convert('RGB')
                if img.size != (width, height):
                    continue
                pixels = [str(val) for pixel in img.getdata() for val in pixel]

                data_row = [
                    str(row.get("timestamp", "")),
                    str(row.get("steer_us", "")),
                    str(row.get("throttle_us", "")),
                    str(row.get("steer_norm", 0)),
                    str(row.get("throttle_norm", 0)),
                    str(row.get("depth_front", ""))
                ] + pixels

                all_rows.append(data_row)
            except Exception as e:
                st.warning(f"Error reading image {full_img_path}: {e}")

        progress_bar.progress(run_count / total_runs)

    status_text.text("Writing combined_dataset.csv...")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

    progress_bar.empty()
    status_text.empty()
    st.success(f"Combined dataset created!\nSaved to: `{output_csv}`\nTotal frames: {len(all_rows)}")

# -------------------------
# (Rest of helpers unchanged: CSV load, old format, delete, etc.)
# -------------------------
def is_csv_folder(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "dataset.csv"))

def load_csv_dataset(path: str):
    csv = os.path.join(path, "dataset.csv")
    if not os.path.exists(csv):
        return None, None
    try:
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        return df, csv
    except Exception as e:
        st.error(f"CSV error: {e}")
        return None, None

def find_image_column(df: pd.DataFrame):
    for col in ["rgb_path", "rgbpath", "image_path", "image"]:
        if col in df.columns:
            return col
    for col in df.columns:
        if any(k in col.lower() for k in ["rgb", "image", "path"]):
            return col
    return None

def delete_csv_frame(df: pd.DataFrame, csv_path: str, idx: int, folder: str):
    row = df.iloc[idx]
    img_col = find_image_column(df)
    if img_col and pd.notna(row[img_col]):
        p = resolve_image_path(folder, row[img_col])
        if os.path.exists(p):
            try:
                os.remove(p)
            except:
                pass
    df = df.drop(idx).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    return df

def load_old_frame(folder: str, idx: int):
    img_p = os.path.join(folder, f"{idx}.jpg")
    json_p = os.path.join(folder, f"{idx}.json")
    if not (os.path.exists(img_p) and os.path.exists(json_p)):
        return None, None
    try:
        img = Image.open(img_p)
        with open(json_p) as f:
            meta = json.load(f)
        return img, meta
    except:
        return None, None

def delete_old_frame(folder: str, idx: int):
    for ext in [".jpg", ".json"]:
        p = os.path.join(folder, f"{idx}{ext}")
        if os.path.exists(p):
            try:
                os.remove(p)
            except:
                pass

def old_frame_count(folder: str):
    return len([f for f in os.listdir(folder) if f.endswith(".jpg")])

# -------------------------
# Main UI
# -------------------------
st.title("RC Car Dataset Viewer & Cleaner")

root_folder = st.text_input(
    "Root folder:",
    placeholder="e.g. X:\\coding_projects\\school\\jetracer1\\data"
)

if not root_folder or not os.path.isdir(root_folder):
    st.info("Enter a valid folder path.")
    st.stop()

# NEW: Create Combined CSV Button
st.markdown("### Create Training Dataset")
if st.button("Create combined_dataset.csv (pixel-flattened)", type="primary", use_container_width=True):
    create_combined_csv(root_folder)

st.markdown("---")

folders = sorted([d for d in os.listdir(root_folder)
                  if os.path.isdir(os.path.join(root_folder, d))])
if not folders:
    st.warning("No folders found.")
    st.stop()

selected = st.selectbox("Select run/session:", folders, key="folder_sel")
folder_path = os.path.join(root_folder, selected)

if st.session_state.current_folder != selected:
    st.session_state.idx = 0
    st.session_state.current_folder = selected

is_csv = is_csv_folder(folder_path)

# [Rest of viewer code — unchanged from last working version]
# (CSV branch + old format branch with fixed thumbnails)

if is_csv:
    df, csv_path = load_csv_dataset(folder_path)
    if df is None:
        st.stop()

    total = len(df)
    img_col = find_image_column(df)

    st.markdown("### Bulk Cleanup")
    if "throttle_norm" in df.columns:
        if st.button("Remove all throttle_norm == 0 frames"):
            with st.spinner("Deleting..."):
                before = len(df)
                for i in reversed(df[df["throttle_norm"] == 0].index.tolist()):
                    df = delete_csv_frame(df, csv_path, i, folder_path)
                st.success(f"Removed {before - len(df)} frames → {len(df)} left")
                st.rerun()

    c1, c2, c3 = st.columns([1, 2, 1])
    if c1.button("Previous", use_container_width=True):
        st.session_state.idx = max(0, st.session_state.idx - 1)
        st.rerun()
    current = c2.number_input("Frame", 0, total-1, st.session_state.idx, step=1, key="csv_idx")
    if current != st.session_state.idx:
        st.session_state.idx = current
        st.rerun()
    if c3.button("Next", use_container_width=True):
        st.session_state.idx = min(total-1, st.session_state.idx + 1)
        st.rerun()

    idx = st.session_state.idx
    row = df.iloc[idx]
    img_path = resolve_image_path(folder_path, row[img_col]) if img_col else None

    col_img, col_info = st.columns([4, 2])
    if img_path and os.path.exists(img_path):
        col_img.image(Image.open(img_path), caption=f"Frame {idx+1}/{total}", use_column_width=True)
    else:
        col_img.error("Image not found")
        if img_col:
            col_img.code(f"CSV path: {row[img_col]}")

    with col_info:
        st.subheader("Controls")
        t = float(row.get("throttle_norm", 0))
        s = float(row.get("steer_norm", 0))
        st.write(f"**Throttle:** {t:.3f}")
        st.progress(min(max(t, 0), 1))
        dir_text = "LEFT" if s > 0.05 else "RIGHT" if s < -0.05 else "STRAIGHT"
        st.write(f"**Steering:** {dir_text} ({s:+.3f})")

    if st.button("Delete This Frame", type="primary", use_container_width=True):
        df = delete_csv_frame(df, csv_path, idx, folder_path)
        st.success("Deleted")
        if idx >= len(df):
            st.session_state.idx = max(0, len(df)-1)
        st.rerun()

    st.markdown("### Thumbnail Strip (click to jump)")
    half = THUMB_WINDOW // 2
    start = max(0, idx - half)
    end = min(total, start + THUMB_WINDOW)
    if end - start < THUMB_WINDOW:
        start = max(0, end - THUMB_WINDOW)

    thumb_cols = st.columns(THUMB_WINDOW)
    for i in range(THUMB_WINDOW):
        pos = start + i
        if pos >= total:
            thumb_cols[i].write("")
            continue
        path = resolve_image_path(folder_path, df.iloc[pos][img_col]) if img_col else None
        if path and os.path.exists(path):
            thumb_cols[i].image(Image.open(path).resize((130, 90)), use_column_width=True)
        else:
            thumb_cols[i].write("missing")
        if thumb_cols[i].button(str(pos), key=f"thumb_{pos}"):
            st.session_state.idx = pos
            st.rerun()

else:
    total = old_frame_count(folder_path)
    if total == 0:
        st.error("No frames.")
        st.stop()

    c1, c2, c3 = st.columns([1, 2, 1])
    if c1.button("Previous", use_container_width=True):
        st.session_state.idx = max(0, st.session_state.idx - 1)
        st.rerun()
    current = c2.number_input("Frame", 0, total-1, st.session_state.idx, step=1, key="old_idx")
    if current != st.session_state.idx:
        st.session_state.idx = current
        st.rerun()
    if c3.button("Next", use_container_width=True):
        st.session_state.idx = min(total-1, st.session_state.idx + 1)
        st.rerun()

    idx = st.session_state.idx
    img, meta = load_old_frame(folder_path, idx)
    if not img:
        st.error("Frame missing")
        st.stop()

    col_img, col_info = st.columns([4, 2])
    col_img.image(img, caption=f"Frame {idx}", use_column_width=True)

    with col_info:
        st.subheader("Metadata")
        steer = meta.get("steering", 0)
        throttle = meta.get("throttle", 0)
        dir_text = "LEFT" if steer < -0.05 else "RIGHT" if steer > 0.05 else "STRAIGHT"
        st.write(f"**Steering:** {dir_text} ({steer:+.3f})")
        st.write(f"**Throttle:** {throttle:.3f}")
        with st.expander("Full JSON"):
            st.json(meta)

    if st.button("Delete This Frame", type="primary", use_container_width=True):
        delete_old_frame(folder_path, idx)
        st.success("Deleted")
        st.rerun()

    st.markdown("### Thumbnail Strip")
    half = THUMB_WINDOW // 2
    start = max(0, idx - half)
    end = min(total, start + THUMB_WINDOW)
    if end - start < THUMB_WINDOW:
        start = max(0, end - THUMB_WINDOW)

    cols = st.columns(THUMB_WINDOW)
    for i in range(THUMB_WINDOW):
        pos = start + i
        if pos >= total:
            cols[i].write("")
            continue
        timg, _ = load_old_frame(folder_path, pos)
        if timg:
            cols[i].image(timg.resize((130, 90)), use_column_width=True)
        if cols[i].button(str(pos), key=f"oldthumb_{pos}"):
            st.session_state.idx = pos
            st.rerun()

st.caption(f"Root: {root_folder} • Current: {selected} • {'CSV Run' if is_csv else 'Old Session'} • Frames: {total}")
