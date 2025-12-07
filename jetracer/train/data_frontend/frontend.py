"""
RC Car Dataset Viewer & Cleaner
- Fixed: thumbnail clicks now work 100% reliably
- Fixed: navigation (single click)
- Fixed: broken CSV paths with runs_rgb_depth/...
- Supports both CSV runs and old jpg+json sessions
"""

import os
import json
import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path

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
# Smart Path Resolver (fixes runs_rgb_depth/... fake paths)
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
# CSV Helpers
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

# -------------------------
# Old Format Helpers
# -------------------------
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
# Main App
# -------------------------
st.title("RC Car Dataset Viewer & Cleaner")

root_folder = st.text_input(
    "Root folder:",
    placeholder="e.g. X:\\coding_projects\\school\\jetracer1\\data"
)

if not root_folder or not os.path.isdir(root_folder):
    st.info("Enter a valid path to continue.")
    st.stop()

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

# ========================
# CSV Format
# ========================
if is_csv:
    df, csv_path = load_csv_dataset(folder_path)
    if df is None:
        st.stop()

    total = len(df)
    img_col = find_image_column(df)

    # Bulk cleanup
    st.markdown("### Bulk Cleanup")
    if "throttle_norm" in df.columns:
        if st.button("Remove all throttle_norm == 0 frames"):
            with st.spinner("Deleting..."):
                before = len(df)
                for i in reversed(df[df["throttle_norm"] == 0].index.tolist()):
                    df = delete_csv_frame(df, csv_path, i, folder_path)
                st.success(f"Removed {before - len(df)} frames → {len(df)} left")
                st.rerun()

    # Navigation
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

    # Image + Info
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

    # FIXED Thumbnail Strip — columns created once
    st.markdown("### Thumbnail Strip (click to jump)")
    half = THUMB_WINDOW // 2
    start = max(0, idx - half)
    end = min(total, start + THUMB_WINDOW)
    if end - start < THUMB_WINDOW:
        start = max(0, end - THUMB_WINDOW)

    thumb_cols = st.columns(THUMB_WINDOW)  # Create all columns first

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

# ========================
# Old jpg+json Format
# ========================
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
