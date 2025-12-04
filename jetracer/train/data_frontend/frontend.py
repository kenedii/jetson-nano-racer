"""
viewer.py
Streamlit app to browse RC car run folders, view frames, visualize controls,
and delete bad frames (removes image file + CSV row).

This version fixes Windows path issues and prevents doubling the run folder
when CSV paths already include the run folder name.

Run:
    streamlit run viewer.py
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
    """Strip whitespace from column names to avoid KeyErrors from trailing spaces."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def load_run_folders(root_dir: str):
    """Return a sorted list of subfolders starting with 'run_' in root_dir."""
    if not os.path.isdir(root_dir):
        return []
    return sorted([d for d in os.listdir(root_dir) if d.startswith("run_")])


def load_dataset(run_path: str):
    """Load dataset.csv from run_path and normalize column names."""
    csv_path = os.path.join(run_path, "dataset.csv")
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    df = normalize_df_columns(df)
    return df, csv_path


def resolve_image_path(root_folder: str, raw_path: str) -> str:
    """
    Robust image path resolver.

    Handles cases where:
    - csv 'rgb_path' is absolute -> return normalized absolute
    - csv 'rgb_path' is relative and already contains the run-folder root (avoid double-joining)
    - csv 'rgb_path' is a plain relative path -> join with root_folder
    - tries sensible fallbacks (cwd + relative)
    """
    if raw_path is None:
        return ""

    p = str(raw_path).strip()

    # Quick sanitize of slashes: use forward slashes for uniform checks
    p_forward = p.replace("\\", "/")

    # If the csv contains an absolute path already
    if os.path.isabs(p):
        return os.path.normpath(p)

    # Base folder name for the selected root (e.g., 'runs_rgb_depth')
    root_base = os.path.basename(root_folder).replace("\\", "/")

    # If the relative path already begins with the root_base (e.g., "runs_rgb_depth/...")
    if p_forward.startswith(root_base + "/"):
        # Join with parent directory of root_folder (one level up) to avoid duplication
        parent = os.path.dirname(root_folder)
        candidate = os.path.join(parent, *p_forward.split("/"))
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate):
            return candidate
        # If not found, also try joining root_folder (in case of unexpected layout)
        alt = os.path.normpath(os.path.join(root_folder, *p_forward.split("/")))
        if os.path.exists(alt):
            return alt
        # fallback to normalized candidate even if it doesn't exist
        return candidate

    # Otherwise, typical case: join root_folder + relative path parts
    candidate = os.path.normpath(os.path.join(root_folder, *p_forward.split("/")))
    if os.path.exists(candidate):
        return candidate

    # Fallbacks: try cwd + relative
    cwd_candidate = os.path.normpath(os.path.join(os.getcwd(), *p_forward.split("/")))
    if os.path.exists(cwd_candidate):
        return cwd_candidate

    # Final fallback: return the most likely candidate (root_folder + relative) normalized
    return candidate


def delete_frame_and_image(df: pd.DataFrame, csv_path: str, index_to_delete: int, root_folder: str):
    """
    Delete the CSV row at index_to_delete and the corresponding image file.
    Returns updated dataframe.
    """
    # Defensive: ensure index is valid
    if index_to_delete < 0 or index_to_delete >= len(df):
        return df

    row = df.iloc[index_to_delete]
    # Support column name variations by trying several common names
    possible_keys = ["rgb_path", "rgbpath", "image_path", "image"]
    rgb_key = None
    for k in possible_keys:
        if k in df.columns:
            rgb_key = k
            break
    if rgb_key is None:
        # fallback to exact column if present
        if "rgb_path" in df.columns:
            rgb_key = "rgb_path"
        else:
            # nothing we can do: remove row but can't delete file
            df = df.drop(index_to_delete).reset_index(drop=True)
            df.to_csv(csv_path, index=False)
            return df

    raw_img_path = row[rgb_key]
    img_full = resolve_image_path(root_folder, raw_img_path)

    # Delete image file if it exists
    try:
        if img_full and os.path.exists(img_full):
            os.remove(img_full)
    except Exception as e:
        # If deletion fails, continue to remove row and save CSV
        st.warning(f"Failed to delete image file {img_full}: {e}")

    # Remove row from dataframe and save CSV
    df = df.drop(index_to_delete).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    return df


def steering_visualization(steer_norm: float) -> str:
    """Return a simple label for steering value."""
    try:
        s = float(steer_norm)
    except Exception:
        return "CENTER"
    if s < -0.05:
        return "LEFT"
    elif s > 0.05:
        return "RIGHT"
    return "CENTER"


# -------------------------
# Streamlit UI
# -------------------------

st.title("📊 RC Car Dataset Viewer & Cleaner")

root_folder = st.text_input("Root folder containing run_YYYYMMDD_HHMMSS folders (full path):")

if root_folder:
    if not os.path.isdir(root_folder):
        st.error("Provided root folder path does not exist or is not a directory.")
        st.stop()

    run_folders = load_run_folders(root_folder)

    if not run_folders:
        st.warning("No folders starting with 'run_' found inside the provided root folder.")
        st.stop()

    selected_run = st.selectbox("Select a run folder:", run_folders)

    if selected_run:
        run_path = os.path.join(root_folder, selected_run)
        df, csv_path = load_dataset(run_path)
        if df is None:
            st.error(f"No dataset.csv found inside {run_path}")
            st.stop()

        # Ensure column names are trimmed (again, defensive)
        df = normalize_df_columns(df)

        # Session state index
        if "idx" not in st.session_state:
            st.session_state.idx = 0

        # Top navigation
        col_prev, col_center, col_next = st.columns([1, 1, 1])

        if col_prev.button("⬅️ Previous"):
            st.session_state.idx = max(0, st.session_state.idx - 1)

        if col_next.button("Next ➡️"):
            st.session_state.idx = min(len(df) - 1, st.session_state.idx + 1)

        # Arrow key support (simple query-param trick)
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
            # remove param by rerunning without it
            st.experimental_set_query_params()
        if "next" in params:
            st.session_state.idx = min(len(df) - 1, st.session_state.idx + 1)
            st.experimental_set_query_params()

        # Clamp index
        idx = min(max(0, st.session_state.idx), max(0, len(df) - 1))
        st.session_state.idx = idx

        # Get row safely
        try:
            row = df.iloc[idx]
        except Exception:
            st.error("Index out of range for the dataset.")
            st.stop()

        # Build and resolve image path
        # Support several possible column names for image path
        image_col_candidates = ["rgb_path", "rgbpath", "image_path", "image"]
        image_col = next((c for c in image_col_candidates if c in df.columns), None)
        if image_col is None:
            # If none found, try to guess by looking for a column containing 'rgb' or 'img'
            image_col = None
            for c in df.columns:
                if "rgb" in c.lower() or "img" in c.lower() or "image" in c.lower():
                    image_col = c
                    break

        raw_img_path = row[image_col] if image_col else ""
        img_full_path = resolve_image_path(root_folder, raw_img_path)

        # Layout: image + info
        img_col, info_col = st.columns([3, 2])

        if not img_full_path or not os.path.exists(img_full_path):
            img_col.error(f"Image not found: {img_full_path}")
            # still show text info below
        else:
            try:
                img = Image.open(img_full_path)
                img_col.image(img, caption=f"Frame {idx+1}/{len(df)}", use_column_width=True)
            except Exception as e:
                img_col.error(f"Failed to open image {img_full_path}: {e}")

        # Info panel
        info_col.subheader("Frame Information")
        # Show timestamp if present
        if "timestamp" in df.columns:
            info_col.write(f"**Timestamp:** {row['timestamp']}")
        # Depth front
        if "depth_front" in df.columns:
            info_col.write(f"**Depth (front):** {row['depth_front']}")

        # Throttle and steering normalized values visualization
        throttle_norm = float(row["throttle_norm"]) if "throttle_norm" in df.columns else 0.0
        steer_norm = float(row["steer_norm"]) if "steer_norm" in df.columns else 0.0

        info_col.write("**Throttle (normalized):**")
        # ensure progress between 0 and 1; if throttle_norm can be negative, clamp
        try:
            t_val = min(max(float(throttle_norm), 0.0), 1.0)
        except Exception:
            t_val = 0.0
        info_col.progress(t_val)

        steer_label = steering_visualization(steer_norm)
        if steer_label == "LEFT":
            info_col.write("**Steering:** ⬅️ LEFT")
        elif steer_label == "RIGHT":
            info_col.write("**Steering:** ➡️ RIGHT")
        else:
            info_col.write("**Steering:** ⏺ CENTER")

        # Delete controls
        st.markdown("---")
        st.write("Use the button below to permanently delete this frame (image file + CSV row).")
        if st.button("🗑️ Delete this frame permanently"):
            # perform deletion (image + csv row)
            df = delete_frame_and_image(df, csv_path, idx, root_folder)
            st.success(f"Deleted frame {idx}. CSV saved: {csv_path}")
            # adjust index to stay within bounds
            new_len = len(df)
            if new_len == 0:
                st.session_state.idx = 0
            else:
                st.session_state.idx = min(idx, new_len - 1)
            # rerun to show updated frame
            st.experimental_rerun()

        # Optional: show a small summary for the run
        st.markdown("---")
        st.write(f"**Run:** {selected_run}  —  **Frames:** {len(df)}")
        # Show a small tail of the dataframe columns that might be useful
        with st.expander("Preview dataframe columns and current row"):
            st.write(df.iloc[max(0, idx - 2): idx + 3])

