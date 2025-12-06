# realsense_full.py
import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Global objects - created once
pipeline = None
align = None
# Store only RGB frame and the center depth value (float) to minimize bandwidth/CPU
latest_frames = {"rgb": None, "depth_center": 0.0}
import threading
frame_lock = threading.Lock()
stop_event = threading.Event()

def camera_worker():
    """Background fetcher: copies RGB, reads center depth only (float)."""
    global latest_frames
    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            aligned = align.process(frames)
            
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if color_frame and depth_frame:
                rgb = np.asanyarray(color_frame.get_data()).copy()

                # Read only center depth to avoid copying the whole depth map (~600KB/frame)
                w = depth_frame.get_width()
                h = depth_frame.get_height()
                depth_center = float(depth_frame.get_distance(w // 2, h // 2))

                with frame_lock:
                    latest_frames["rgb"] = rgb
                    latest_frames["depth_center"] = depth_center
        except Exception as e:
            print(f"[RealSense Thread Error] {e}")

def start_pipeline():
    global pipeline, align
    if pipeline is None:
        pipeline = rs.pipeline()
        config = rs.config()

        # Enable ONLY RGB and Depth streams (Disable IR to save USB bandwidth/CPU)
        # Reduced resolution + 15 FPS to lower CPU load on Jetson Nano
        config.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 15)
        config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)
        # config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 15) # DISABLED

        pipeline.start(config)

        # FIX: Align depth and IR to the color image by passing rs.stream.color
        align = rs.align(rs.stream.color)
        print("[RealSense] Pipeline started - RGB + IR + Depth ready")
        
        # Start background thread
        t = threading.Thread(target=camera_worker, daemon=True)
        t.start()

def stop_pipeline():
    global pipeline, align
    stop_event.set()
    # Give the thread a moment to exit the loop
    time.sleep(0.5)
    if pipeline:
        try:
            pipeline.stop()
        except Exception as e:
            print(f"[RealSense] Error stopping pipeline: {e}")
        pipeline = None
        align = None
    print("[RealSense] Pipeline stopped.")

def get_aligned_frames():
    """Returns (rgb, depth_center_float) with single lock acquisition"""
    start_pipeline()
    with frame_lock:
        if latest_frames["rgb"] is None:
            return None, None
        return latest_frames["rgb"], latest_frames["depth_center"]

# --------------------- RGB ---------------------
def get_rgb_image():
    start_pipeline()
    with frame_lock:
        if latest_frames["rgb"] is None:
            return None
        return latest_frames["rgb"].copy()


def save_rgb_image(filename):
    img = get_rgb_image()
    if img is not None:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr)
        print("[SAVED] RGB -> " + filename)
        return True
    return False


# --------------------- IR (dots) ---------------------
def get_ir_image():
    start_pipeline()
    with frame_lock:
        if latest_frames["ir"] is None:
            return None
        return latest_frames["ir"].copy()


def save_ir_image(filename):
    img = get_ir_image()
    if img is not None:
        cv2.imwrite(filename, img)
        print("[SAVED] IR dots -> " + filename)
        return True
    return False


# --------------------- DEPTH ---------------------
def get_depth_image():
    # Depth map is no longer stored to reduce CPU. Return None to indicate unavailable.
    return None


def save_depth_image(filename, colored=True):
    depth = get_depth_image()
    if depth is None:
        print("[Depth] Full depth map not stored (optimized mode).")
        return False
    # If enabled in the future, the code below can run.
    return False


# --------------------- Distance helper ---------------------
def get_center_distance():
    _, depth_center = get_aligned_frames()
    if depth_center is None:
        return 0
    if depth_center == 0:
        return 0
    return depth_center


# --------------------- Quick test ---------------------
if __name__ == "__main__":
    print("--- Starting Camera Test ---")
    save_rgb_image("test_rgb.jpg")
    save_ir_image("test_ir.jpg")
    save_depth_image("test_depth.png", colored=True)
    
    dist = get_center_distance()
    if dist > 0:
        print("\nDistance in front of camera: **%.3f meters**" % dist)
    else:
        print("\nNo valid depth at center (Is the object too far, too close, or reflective?)")
    
    # Stop the pipeline explicitly
    if pipeline:
        pipeline.stop()
        print("[RealSense] Pipeline stopped.")
