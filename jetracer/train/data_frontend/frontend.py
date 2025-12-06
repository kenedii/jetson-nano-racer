#frontend.py WITH SAFE CSV LOADING + THUMBNAILS VIEW

import os
import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title="RC Car Dataset Viewer", layout="wide")

# -------------------------
# Utility functions
# -------------------------

def normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def load_run_folders(root_dir: str):
    if not os.path.isdir(root_dir):
        return []
    return sorted([
        d for d in os.listdir(root_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(root_dir, d))
    ])

def load_dataset(run_path: str):
    csv_path = os.path.join(run_path, "dataset.csv")
    if not os.path.exists(csv_path):
        return None, None

    try:
        df = pd.read_csv(csv_path)
        df = normalize_df_columns(df)
        return df, csv_path

    except pd.errors.EmptyDataError:
        st.warning(f"⚠️ dataset.csv in {run_path} is empty or corrupted. Skipping this run.")
        return None, None

    except Exception as e:
        st.error(f"Failed to load {csv_path}: {e}")
        return None, None

def resolve_image_path(root_folder: str, raw_path: str) -> str:
    if raw_path is None:
        return ""
    p = str(raw_path).strip()
    if p == "":
        return ""

    if os.path.isabs(p):
        return os.path.normpath(p)

    p_fwd = p.replace("\\", "/")
    root_base = os.path.basename(root_folder).replace("\\", "/")

    if p_fwd.startswith(root_base + "/"):
        parent = os.path.dirname(root_folder)
        candidate = os.path.normpath(os.path.join(parent, *p_fwd.split("/")))
        if os.path.exists(candidate): return candidate
        alt = os.path.normpath(os.path.join(root_folder, *p_fwd.split("/")))
        if os.path.exists(alt): return alt
        return candidate

    candidate = os.path.normpath(os.path.join(root_folder, *p_fwd.split("/")))
    if os.path.exists(candidate): return candidate

    cwd_candidate = os.path.normpath(os.path.join(os.getcwd(), *p_fwd.split("/")))
    if os.path.exists(cwd_candidate): return cwd_candidate

    return candidate

def delete_frame_and_image(df: pd.DataFrame, csv_path: str, index_to_delete: int, root_folder: str):
    if index_to_delete < 0 or index_to_delete >= len(df):
        return df

    row = df.iloc[index_to_delete]

    for possible in ["rgb_path", "rgbpath", "image_path", "image"]:
        if possible in df.columns:
            rgb_key = possible
            break
    else:
        rgb_key = None

    img_full = None
    if rgb_key:
        raw = row[rgb_key]
        img_full = resolve_image_path(root_folder, raw)

    try:
        if img_full and os.path.exists(img_full):
            os.remove(img_full)
    except Exception as e:
        st.warning(f"Failed to delete {img_full}: {e}")

    df = df.drop(index_to_delete).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    return df

# -------------------------
# Steering visualization
# -------------------------

def steering_visualization(steer_norm: float) -> str:
    try:
        s = float(steer_norm)
    except:
        return "CENTER"
    if s > 0.05: return "RIGHT"
    if s < -0.05: return "LEFT"
    return "CENTER"

# -------------------------
# Frame counting
# -------------------------

def count_total_frames(root_folder: str) -> int:
    total = 0
    for run in load_run_folders(root_folder):
        df, _ = load_dataset(os.path.join(root_folder, run))
        if isinstance(df, pd.DataFrame):
            total += len(df)
    return total

# -------------------------
# Bulk removal
# -------------------------

def remove_zero_accel_frames(root_folder: str):
    summary = []
    for run in load_run_folders(root_folder):
        run_path = os.path.join(root_folder, run)
        df, csv_path = load_dataset(run_path)
        if df is None:
            continue

        if "throttle_norm" not in df.columns:
            summary.append((run, len(df), len(df), 0))
            continue

        before = len(df)
        delete_indices = df[df["throttle_norm"] == 0].index.tolist()

        removed = 0
        for idx in reversed(delete_indices):
            df = delete_frame_and_image(df, csv_path, idx, root_folder)
            removed += 1

        after = len(df)
        summary.append((run, before, after, removed))

    return summary

# -------------------------
# UI
# -------------------------
st.title("📊 RC Car Dataset Viewer & Cleaner")

root_folder = st.text_input("Root folder containing run_* folders:")

if root_folder and os.path.isdir(root_folder):
    st.markdown(f"**Total Frames Across ALL Runs:** {count_total_frames(root_folder)}")

st.markdown("---")

# Bulk cleanup
if root_folder and os.path.isdir(root_folder):
    st.markdown("### 🧹 Bulk Cleanup")
    if st.button("🔥 Remove ALL throttle_norm == 0 frames"):
        summary = remove_zero_accel_frames(root_folder)
        st.success("Bulk cleanup complete.")
        for run, before, after, removed in summary:
            st.write(f"{run}: {before} → {after} (removed {removed})")
        st.write(f"Updated total: **{count_total_frames(root_folder)}**")

st.markdown("---")

# Main viewer
if root_folder:
    if not os.path.isdir(root_folder):
        st.error("Invalid root folder.")
        st.stop()

    runs = load_run_folders(root_folder)
    if not runs:
        st.warning("No run_* folders found.")
        st.stop()

    selected_run = st.selectbox("Select a run:", runs)

    if "last_run" not in st.session_state:
        st.session_state.last_run = None

    if st.session_state.last_run != selected_run:
        st.session_state.idx = 0
        st.session_state.last_run = selected_run

    run_path = os.path.join(root_folder, selected_run)
    df, csv_path = load_dataset(run_path)

    if df is None:
        st.error("This run has no valid dataset.csv.")
        st.stop()

    df = normalize_df_columns(df)

    if "idx" not in st.session_state:
        st.session_state.idx = 0

    idx = st.session_state.idx

    # Navigation
    col_prev, col_mid, col_next = st.columns([1, 1, 1])

    if col_prev.button("⬅️ Previous"):
        st.session_state.idx = max(0, idx - 1)

    if col_next.button("Next ➡️"):
        st.session_state.idx = min(len(df) - 1, idx + 1)

    idx = st.session_state.idx
    row = df.iloc[idx]

    # Find image path
    image_col = None
    for c in ["rgb_path", "rgbpath", "image_path", "image"]:
        if c in df.columns:
            image_col = c
            break

    raw_img_path = row[image_col] if image_col else ""
    img_full = resolve_image_path(root_folder, raw_img_path)

    img_col, info_col = st.columns([4, 2])

    if os.path.exists(img_full):
        img = Image.open(img_full)
        img_col.image(img, caption=f"Frame {idx+1}/{len(df)}", use_column_width=True)
    else:
        img_col.error(f"Image not found: {img_full}")

    # Info
    info_col.subheader("Frame Info")

    throttle = float(row.get("throttle_norm", 0.0))
    steer = float(row.get("steer_norm", 0.0))

    info_col.write("**Throttle:**")
    info_col.progress(min(max(throttle, 0.0), 1.0))

    left_strength = max(0.0, steer)
    right_strength = max(0.0, -steer)

    info_col.write("**Steering**")
    info_col.progress(left_strength, text="Left Turn")
    info_col.progress(right_strength, text="Right Turn")

    label = steering_visualization(steer)
    if label == "LEFT":
        info_col.write("**Steering:** ⬅️ LEFT")
    elif label == "RIGHT":
        info_col.write("**Steering:** ➡️ RIGHT")
    else:
        info_col.write("**Steering:** ⏺ CENTER")

    # Delete frame
    st.markdown("---")
    if st.button("🗑 Delete this frame"):
        df = delete_frame_and_image(df, csv_path, idx, root_folder)
        st.success(f"Deleted frame {idx}")
        if len(df) == 0:
            st.session_state.idx = 0
        else:
            st.session_state.idx = min(idx, len(df) - 1)
        st.experimental_rerun()

    st.markdown("---")
    st.write(f"Run: {selected_run} — Frames: {len(df)}")

    # -------------------------
    # THUMBNAIL STRIP (10 images)
    # -------------------------

    st.subheader("Thumbnail Strip (click to jump)")

    thumbs_per_row = 10
    start = max(0, idx - thumbs_per_row//2)
    end = min(len(df), start + thumbs_per_row)

    thumb_cols = st.columns(end - start)

    for i, c in enumerate(range(start, end)):
        row_thumb = df.iloc[c]
        raw_p = row_thumb[image_col] if image_col else ""
        img_p = resolve_image_path(root_folder, raw_p)

        if os.path.exists(img_p):
            im = Image.open(img_p)
            thumb_cols[i].image(im, use_column_width=True)

        if thumb_cols[i].button(f"Go {c}"):
            st.session_state.idx = c
            st.experimental_rerun()

    # Preview nearby rows
    with st.expander("DataFrame Preview:"):
        st.write(df.iloc[max(0, idx-2): idx+3])

# ---------------------------
# CONFIGURATION
# ---------------------------
DATA_ROOT = "sessions"  # Root folder containing all sessions


# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------

def load_frame(session, index):
    """Load a single frame: image + metadata JSON."""
    session_path = os.path.join(DATA_ROOT, session)

    img_path = os.path.join(session_path, f"{index}.jpg")
    meta_path = os.path.join(session_path, f"{index}.json")

    if not os.path.exists(img_path) or not os.path.exists(meta_path):
        return None, None

    img = Image.open(img_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return img, meta


def get_num_frames(session):
    session_path = os.path.join(DATA_ROOT, session)
    frames = [f for f in os.listdir(session_path) if f.endswith(".jpg")]
    return len(frames)


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="RC Car Data Viewer", layout="wide")
st.title("🚗 RC Car Dataset Viewer")

# Session selection
sessions = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
session = st.sidebar.selectbox("Select Session", sessions)

if not session:
    st.stop()

num_frames = get_num_frames(session)
st.sidebar.write(f"Total frames: {num_frames}")

# Frame index slider
frame_index = st.sidebar.number_input(
    "Frame Index", min_value=0, max_value=max(0, num_frames - 1), value=0, step=1
)

# Load frame
img, meta = load_frame(session, frame_index)

if img is None:
    st.error("Frame not found.")
    st.stop()

# ---------------------------
# FIX: Steering label flipped
# If steering < 0 → RIGHT; steering > 0 → LEFT
# ---------------------------
steer_value = meta.get("steering", 0)
throttle_value = meta.get("throttle", 0)

if steer_value < 0:
    steer_label = "➡️ RIGHT"
elif steer_value > 0:
    steer_label = "⬅️ LEFT"
else:
    steer_label = "⬆️ STRAIGHT"

# Display main frame + metadata
col1, col2 = st.columns([2, 1])

with col1:
    st.image(img, caption=f"Frame {frame_index}", use_column_width=True)

with col2:
    st.subheader("Metadata")
    st.write(f"**Steering:** {steer_label}")
    st.write(f"**Steering Raw:** {steer_value:.3f}")
    st.write(f"**Throttle:** {throttle_value:.3f}")
    st.write(meta)

# -------------------------------------------
# NEW FEATURE: Thumbnail viewer (10 images)
# -------------------------------------------
st.subheader("Thumbnail Preview")

THUMB_WINDOW = 10
start_index = max(0, frame_index - (THUMB_WINDOW // 2))
end_index = min(num_frames, start_index + THUMB_WINDOW)

thumb_cols = st.columns(10)

for i, idx in enumerate(range(start_index, end_index)):
    thumb_img, _ = load_frame(session, idx)
    if thumb_img:
        thumb_cols[i].image(thumb_img.resize((120, 80)))
        if thumb_cols[i].button(f"Go {idx}"):
            st.experimental_set_query_params(frame=idx)
            st.rerun()
