# realsense_full.py
import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Global objects - created once
pipeline = None
align = None
latest_frames = {"rgb": None, "depth": None, "ir": None}
import threading
frame_lock = threading.Lock()
stop_event = threading.Event()

def camera_worker():
    global latest_frames
    while not stop_event.is_set():
        try:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            aligned = align.process(frames)
            
            # Get frames
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            # IR frame (index 1)
            ir_frame = aligned.get_infrared_frame(1)

            if color_frame and depth_frame:
                # Convert to numpy arrays
                rgb = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())
                
                ir = None
                if ir_frame:
                    ir_data = np.asanyarray(ir_frame.get_data())
                    ir = cv2.cvtColor(ir_data, cv2.COLOR_GRAY2BGR)

                # Update global state safely
                with frame_lock:
                    latest_frames["rgb"] = rgb
                    latest_frames["depth"] = depth
                    latest_frames["ir"] = ir
        except Exception as e:
            print(f"[RealSense Thread Error] {e}")

def start_pipeline():
    global pipeline, align
    if pipeline is None:
        pipeline = rs.pipeline()
        config = rs.config()

        # Enable the three streams we need
        # Reduced to 15 FPS to lower CPU load on Jetson Nano
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 15)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)

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
    """Returns (rgb, depth) tuple efficiently with single lock acquisition"""
    start_pipeline()
    with frame_lock:
        if latest_frames["rgb"] is None or latest_frames["depth"] is None:
            return None, None
        return latest_frames["rgb"].copy(), latest_frames["depth"].copy()

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
    start_pipeline()
    with frame_lock:
        if latest_frames["depth"] is None:
            return None
        return latest_frames["depth"].copy()


def save_depth_image(filename, colored=True):
    depth = get_depth_image()
    if depth is None:
        return False

    # Save raw 16-bit depth (perfect for later use)
    raw_name = filename.replace(".png", "_raw.png")
    cv2.imwrite(raw_name, depth)
    print("[SAVED] Raw depth -> " + raw_name)

    if colored:
        # Scale depth for visualization (e.g., scale 5 meters (5000mm) to 255)
        vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255/5000), cv2.COLORMAP_JET)
        color_name = filename.replace(".png", "_color.jpg")
        cv2.imwrite(color_name, vis)
        print("[SAVED] Depth color -> " + color_name)

    return True


# --------------------- Distance helper ---------------------
def get_center_distance():
    depth = get_depth_image()
    if depth is None:
        return 0
    h, w = depth.shape
    dist_mm = depth[h//2, w//2]
    
    # Depth value of 0 means no valid depth detected (e.g., outside range or reflective)
    if dist_mm == 0:
        return 0
    return dist_mm / 1000.0  # return meters


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
