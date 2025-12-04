"""
frontend.py
Streamlit app to browse RC car run folders, view frames, visualize controls,
delete bad frames, and bulk-remove frames where throttle_norm == 0.

Run:
    streamlit run frontend.py
"""

import os
import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title="RC Car Dataset Viewer", layout="wide")


# -------------------------
# Utility functions
# -------------------------

def normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names to avoid KeyErrors."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def load_run_folders(root_dir: str):
    """Return sorted list of subfolders starting with 'run_' in root_dir."""
    if not os.path.isdir(root_dir):
        return []
    return sorted([d for d in os.listdir(root_dir) if d.startswith("run_") and os.path.isdir(os.path.join(root_dir, d))])


def load_dataset(run_path: str):
    """Load dataset.csv from run_path."""
    csv_path = os.path.join(run_path, "dataset.csv")
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    df = normalize_df_columns(df)
    return df, csv_path


def resolve_image_path(root_folder: str, raw_path: str) -> str:
    """Normalize CSV image paths, safely resolving Windows/Linux inversions and avoiding double-root."""
    if raw_path is None:
        return ""

    p = str(raw_path).strip()
    if p == "":
        return ""

    # If path is absolute already, normalize and return
    if os.path.isabs(p):
        return os.path.normpath(p)

    # Normalize slashes for checking
    p_forward = p.replace("\\", "/")

    # Root dir base name (example: 'runs_rgb_depth')
    root_base = os.path.basename(root_folder).replace("\\", "/")

    # If CSV path already begins with the root base (avoid duplicate join)
    if p_forward.startswith(root_base + "/"):
        # Join with parent of root_folder
        parent = os.path.dirname(root_folder)
        candidate = os.path.normpath(os.path.join(parent, *p_forward.split("/")))
        if os.path.exists(candidate):
            return candidate
        # fallback to joining with root_folder
        alt = os.path.normpath(os.path.join(root_folder, *p_forward.split("/")))
        if os.path.exists(alt):
            return alt
        return candidate

    # Typical path: relative to selected root_folder
    candidate = os.path.normpath(os.path.join(root_folder, *p_forward.split("/")))
    if os.path.exists(candidate):
        return candidate

    # Try as relative to current working directory
    cwd_candidate = os.path.normpath(os.path.join(os.getcwd(), *p_forward.split("/")))
    if os.path.exists(cwd_candidate):
        return cwd_candidate

    # Final fallback: return candidate (even if doesn't exist) so caller can report missing
    return candidate


def delete_frame_and_image(df: pd.DataFrame, csv_path: str, index_to_delete: int, root_folder: str):
    """Delete image file and CSV row. Returns updated df."""
    if index_to_delete < 0 or index_to_delete >= len(df):
        return df

    row = df.iloc[index_to_delete]

    # support several column name variants
    possible_keys = ["rgb_path", "rgbpath", "image_path", "image"]
    rgb_key = next((k for k in possible_keys if k in df.columns), None)

    # If no image column found, just remove row and save CSV
    if rgb_key is None:
        df = df.drop(index_to_delete).reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        return df

    raw_img = row[rgb_key]
    img_full = resolve_image_path(root_folder, raw_img)

    try:
        if img_full and os.path.exists(img_full):
            os.remove(img_full)
    except Exception as e:
        st.warning(f"Failed to delete image file {img_full}: {e}")

    df = df.drop(index_to_delete).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    return df


def steering_visualization(steer_norm: float) -> str:
    """
    Interpret steer_norm so that:
      - positive steer_norm => LEFT
      - negative steer_norm => RIGHT
    """
    try:
        s = float(steer_norm)
    except Exception:
        return "CENTER"
    if s > 0.05:
        return "LEFT"
    elif s < -0.05:
        return "RIGHT"
    return "CENTER"


# -------------------------
# Bulk Removal: throttle_norm == 0
# -------------------------

def remove_zero_accel_frames(root_folder: str):
    """
    Delete all frames with throttle_norm == 0 in ALL run_* folders.
    Returns a summary list of tuples: (run_name, before_count, after_count, removed_count)
    """
    runs = load_run_folders(root_folder)
    summary = []

    for run in runs:
        run_path = os.path.join(root_folder, run)
        df, csv_path = load_dataset(run_path)
        if df is None:
            continue

        if "throttle_norm" not in df.columns:
            summary.append((run, len(df), len(df), 0))
            continue

        before = len(df)
        # indices to delete (list so we can reverse delete)
        to_delete = df[df["throttle_norm"] == 0].index.tolist()

        removed = 0
        # iterate reversed so indices remain valid as we drop rows
        for idx in reversed(to_delete):
            # delete both file and row
            df = delete_frame_and_image(df, csv_path, idx, root_folder)
            removed += 1

        after = len(df)
        summary.append((run, before, after, removed))

    return summary


# -------------------------
# Helper: Count total frames across all runs
# -------------------------
def count_total_frames(root_folder: str) -> int:
    total = 0
    runs = load_run_folders(root_folder)
    for run in runs:
        run_path = os.path.join(root_folder, run)
        df, _ = load_dataset(run_path)
        if df is None:
            continue
        total += len(df)
    return total


# -------------------------
# Streamlit UI
# -------------------------
st.title("📊 RC Car Dataset Viewer & Cleaner")

root_folder = st.text_input("Root folder containing run_YYYYMMDD_HHMMSS folders (full path):")

# Show total frames across all runs (if folder valid)
if root_folder and os.path.isdir(root_folder):
    total_frames = count_total_frames(root_folder)
    st.markdown(f"**Total Frames Across All Runs:** {total_frames}")

st.markdown("---")

# Bulk Remove Zero Accel
if root_folder and os.path.isdir(root_folder):
    st.markdown("### 🧹 Bulk Cleanup Tool")
    if st.button("🔥 Remove ALL frames with throttle_norm == 0 (across ALL runs)"):
        summary = remove_zero_accel_frames(root_folder)
        st.success("Bulk cleanup complete.")
        for run, before, after, removed in summary:
            st.write(f"**{run}** — {before} → {after} (removed {removed})")
        # refresh total
        total_frames = count_total_frames(root_folder)
        st.write(f"Updated total frames: **{total_frames}**")

st.markdown("---")

if root_folder:
    if not os.path.isdir(root_folder):
        st.error("Provided root folder does not exist.")
        st.stop()

    run_folders = load_run_folders(root_folder)

    if not run_folders:
        st.warning("No folders starting with 'run_' found.")
        st.stop()

    selected_run = st.selectbox("Select a run folder:", run_folders)

    if selected_run:
        run_path = os.path.join(root_folder, selected_run)
        df, csv_path = load_dataset(run_path)
        if df is None:
            st.error(f"No dataset.csv found inside {run_path}")
            st.stop()

        df = normalize_df_columns(df)

        if "idx" not in st.session_state:
            st.session_state.idx = 0

        # Navigation arrows
        col_prev, col_center, col_next = st.columns([1, 1, 1])

        if col_prev.button("⬅️ Previous"):
            st.session_state.idx = max(0, st.session_state.idx - 1)

        if col_next.button("Next ➡️"):
            st.session_state.idx = min(len(df) - 1, st.session_state.idx + 1)

        # Arrow-key JS
        st.markdown(
            """
            <script>
            document.addEventListener('keydown', function(e) {
                if (e.key === 'ArrowLeft') {
                    window.location.href += (window.location.href.indexOf('?') === -1 ? '?' : '&') + 'prev=true';
                }
                if (e.key === 'ArrowRight') {
                    window.location.href += (window.location.href.indexOf('?') === -1 ? '?' : '&') + 'next=true';
                }
            });
            </script>
            """,
            unsafe_allow_html=True,
        )

        params = st.experimental_get_query_params()
        if "prev" in params:
            st.session_state.idx = max(0, st.session_state.idx - 1)
            st.experimental_set_query_params()
        if "next" in params:
            st.session_state.idx = min(len(df) - 1, st.session_state.idx + 1)
            st.experimental_set_query_params()

        idx = min(max(0, st.session_state.idx), len(df) - 1)
        st.session_state.idx = idx

        row = df.iloc[idx]

        # Determine image column
        image_col = None
        for c in ["rgb_path", "rgbpath", "image_path", "image"]:
            if c in df.columns:
                image_col = c
                break
        if image_col is None:
            for c in df.columns:
                if "rgb" in c.lower() or "img" in c.lower():
                    image_col = c
                    break

        raw_img_path = row[image_col] if image_col else ""
        img_full_path = resolve_image_path(root_folder, raw_img_path)

        # Wider image column to restore larger image display
        img_col, info_col = st.columns([4, 2])

        if not img_full_path or not os.path.exists(img_full_path):
            img_col.error(f"Image not found: {img_full_path}")
        else:
            try:
                img = Image.open(img_full_path)
                img_col.image(img, caption=f"Frame {idx+1}/{len(df)}", use_column_width=True)
            except Exception as e:
                img_col.error(f"Failed to open image {img_full_path}: {e}")

        # Frame info
        info_col.subheader("Frame Information")
        if "timestamp" in df.columns:
            info_col.write(f"**Timestamp:** {row['timestamp']}")
        if "depth_front" in df.columns:
            info_col.write(f"**Depth (front):** {row['depth_front']}")

        throttle_norm = float(row["throttle_norm"]) if "throttle_norm" in df.columns else 0.0
        steer_norm = float(row["steer_norm"]) if "steer_norm" in df.columns else 0.0

        info_col.write("**Throttle (normalized):**")
        try:
            info_col.progress(min(max(float(throttle_norm), 0.0), 1.0))
        except:
            info_col.progress(0.0)

        # Fixed left/right visualization (interpreting positive = LEFT, negative = RIGHT)
        info_col.markdown("**Steering**")
        # Now positive steer_norm -> left bar, negative -> right bar
        left_strength = max(0.0, float(steer_norm))   # positive => left
        right_strength = max(0.0, -float(steer_norm))  # negative => right

        # Show left then right so user reads left->right naturally
        info_col.progress(left_strength, text="Left Turn")
        info_col.progress(right_strength, text="Right Turn")

        steer_label = steering_visualization(steer_norm)
        if steer_label == "LEFT":
            info_col.write("**Steering:** ⬅️ LEFT")
        elif steer_label == "RIGHT":
            info_col.write("**Steering:** ➡️ RIGHT")
        else:
            info_col.write("**Steering:** ⏺ CENTER")

        # Delete button
        st.markdown("---")
        st.write("Delete this frame permanently.")
        if st.button("🗑️ Delete this frame"):
            df = delete_frame_and_image(df, csv_path, idx, root_folder)
            st.success(f"Deleted frame {idx}.")
            new_len = len(df)
            if new_len == 0:
                st.session_state.idx = 0
            else:
                st.session_state.idx = min(idx, new_len - 1)
            # update top total by rerunning
            st.experimental_rerun()

        st.markdown("---")
        st.write(f"**Run:** {selected_run} — **Frames:** {len(df)}")

        with st.expander("Preview dataframe rows:"):
            st.write(df.iloc[max(0, idx - 2): idx + 3])
