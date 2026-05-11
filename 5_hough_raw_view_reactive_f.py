"""
ultimate_qcar_pilot.py
The definitive autonomous driving script for QCar.
--------------------------------------------------
FEATURES:
1. Dynamic Lane Memory: Remembers lane width to handle broken lines.
2. Polynomial Curve Fitting: Steers correctly on sharp turns (2nd order poly).
3. Live HSV Tuning: Press 'T' to calibrate camera in real-time.
4. Robust Confidence: Fuses line density, vertical spread, and history.
5. Smart HUD: Visualizes Memory, Target, Curves, and Inputs.

CONTROLS:
- Arrow Keys: Manual Override (Throttle/Steering)
- 'T': Toggle HSV Tuning Mode (Sliders)
- 'R': Toggle ROI View
- 'M': Toggle Mask View (Black/White Edges)
- 'C': Toggle Center/Target View
- 'Q': Quit
"""

import os
import signal
import numpy as np
from threading import Thread
import time
import cv2
from pal.products.qcar import QCar, QCarCameras, IS_PHYSICAL_QCAR
import keyboard

# ================= Configuration =================
CONTROLLER_UPDATE_RATE = 100
CSI_WIDTH, CSI_HEIGHT = 820, 410
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480

# --- Lane Logic Defaults ---
INITIAL_LANE_WIDTH = 600      # Pixels (at bottom of screen)
RIGHT_LANE_OFFSET_PX = 280    # Target offset from right line (300)

# --- Initial Color (Yellow) ---
h_min, s_min, v_min = 15, 50, 50
h_max, s_max, v_max = 45, 255, 255
LOWER_YELLOW = np.array([h_min, s_min, v_min])
UPPER_YELLOW = np.array([h_max, s_max, v_max])

# --- Physics ---
MAX_THROTTLE = 0.3
MAX_STEERING = 0.6
THROTTLE_STEP = 0.02
STEERING_STEP = 0.05

# --- Globals ---
KILL_THREAD = False
throttle = 0.0
raw_steering = 0.0
lane_width_memory = INITIAL_LANE_WIDTH

# --- View Toggles ---
show_roi = False
show_mask = False  # 'M' toggles this
show_center = True
tuning_mode = False

def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)

# ================= Keyboard Handlers =================
def update_thr(d): global throttle; throttle = np.clip(throttle + d, -MAX_THROTTLE, MAX_THROTTLE)
def update_str(d): global raw_steering; raw_steering = np.clip(raw_steering + d, -MAX_STEERING, MAX_STEERING)
def toggle_roi(): global show_roi; show_roi = not show_roi
def toggle_mask(): global show_mask; show_mask = not show_mask
def toggle_center(): global show_center; show_center = not show_center
def quit_app(): global KILL_THREAD; KILL_THREAD = True

def toggle_tuning():
    global tuning_mode
    tuning_mode = not tuning_mode
    if tuning_mode: print("\n[TUNING MODE ON] Adjust sliders.")
    else: print(f"\n[SAVED] L:[{h_min},{s_min},{v_min}] H:[{h_max},{s_max},{v_max}]")

def nothing(x): pass

# ================= Advanced Lane Algorithm =================
'''def process_lane_logic(img, current_memory):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    
    # 1. Edge Detection & ROI
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=yellow_mask)
    blur = cv2.GaussianBlur(masked_gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    roi_top = int(h * 0.55)
    vertices = np.array([[(0, h), (w, h), (w//2 + 50, roi_top), (w//2 - 50, roi_top)]], np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 2. Line Extraction
    # lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 40, minLineLength=40, maxLineGap=100)
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 20, minLineLength=20, maxLineGap=150)
    
    lat_err = 0.0
    conf = 0.0
    curve_pts = None
    target_pt = None
    new_memory = current_memory
    
    all_x, all_y = [], []
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                all_x.extend([x1, x2])
                all_y.extend([y1, y2])

    # 3. Decision Logic
    if len(all_x) > 10:
        try:
            # Curve Fitting
            poly = np.polyfit(all_y, all_x, 2)
            plot_y = np.linspace(roi_top, h, 20)
            plot_x = np.polyval(poly, plot_y).astype(int)
            curve_pts = np.column_stack((plot_x, plot_y)).astype(np.int32)
            
            bottom_x = int(np.polyval(poly, h))
            target_x = bottom_x + RIGHT_LANE_OFFSET_PX
            target_pt = (target_x, h)
            
            center_x = w // 2
            error_px = target_x - center_x
            # lat_err = np.clip(error_px / (w / 2.0), -1.0, 1.0)
            xm_per_pix = 1.25 / RIGHT_LANE_OFFSET_PX
            lat_err = error_px * xm_per_pix
            
            spread = (max(all_y) - min(all_y)) / (h - roi_top)
            conf = min(1.0, 0.5 + spread * 0.5) 
            
        except:
            conf = 0.0
    
    return lat_err, conf, curve_pts, target_pt, new_memory, yellow_mask, masked_edges'''
    
def process_lane_logic(img, current_memory):
    # No globals for memory/history anymore. 
    # Purely reactive frame-by-frame logic.

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

    # 1. FIX: THICKEN THE LINE (Dilation)
    # Stops blinking by filling small holes.
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=1)

    # 2. Edge Detection & ROI
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=yellow_mask)
    edges = cv2.Canny(masked_gray, 30, 100)
    
    roi_top = int(h * 0.55)
    vertices = np.array([[(0, h), (w, h), (w//2 + 50, roi_top), (w//2 - 50, roi_top)]], np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # 3. Line Extraction
    # maxLineGap=50 stops it from jumping to the right side
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 10, minLineLength=15, maxLineGap=50)
    
    lat_err = 0.0
    conf = 0.0
    curve_pts = None
    target_pt = None
    
    all_x, all_y = [], []
    
    # --- FILTERING LOGIC ---
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 == 0: continue 
                slope = (y2 - y1) / (x2 - x1)
                
                # A. REJECT HORIZONTAL LINES (Curbs)
                if abs(slope) < 0.2:
                    continue

                # B. RIGHT SIDE SHIELD
                # Ignore anything on the right side (width * 0.55) INCR TO LESSEN EFFECT
                if x1 > (w * 0.65) or x2 > (w * 0.65):
                    continue

                all_x.extend([x1, x2])
                all_y.extend([y1, y2])

    # --- DECISION LOGIC (No History) ---
    if len(all_x) > 20:
        try:
            y_spread = max(all_y) - min(all_y)
            
            # Anti-Shake: Straight line (deg 1) for short lines, Curve (deg 2) for long lines
            degree = 1 if y_spread < 80 else 2
            current_poly = np.polyfit(all_y, all_x, degree)
            
            # Visuals
            plot_y = np.linspace(roi_top, h, 20)
            plot_x = np.polyval(current_poly, plot_y).astype(int)
            curve_pts = np.column_stack((plot_x, plot_y)).astype(np.int32)
            
            # Target
            bottom_x = int(np.polyval(current_poly, h))
            target_x = bottom_x + RIGHT_LANE_OFFSET_PX
            target_pt = (target_x, h)

            # Lateral Error (Meters)
            center_x = w // 2
            error_px = target_x - center_x
            xm_per_pix = 1.25 / RIGHT_LANE_OFFSET_PX 
            lat_err = error_px * xm_per_pix 
            
            # Confidence
            spread = (max(all_y) - min(all_y)) / (h - roi_top)
            conf = min(1.0, 0.5 + spread * 0.5)

        except:
            # If polyfit fails, just return 0
            lat_err = 0.0
            conf = 0.0
            curve_pts = None
            target_pt = None

    else:
        # No lines found -> Return 0
        lat_err = 0.0
        conf = 0.0
        curve_pts = None
        target_pt = None

    # We return 'current_memory' back just to satisfy the function signature, 
    # but we didn't change it or use it.
    return lat_err, conf, curve_pts, target_pt, current_memory, yellow_mask, masked_edges

# ================= Main Loop =================
def controlLoop():
    global KILL_THREAD, throttle, raw_steering, lane_width_memory, tuning_mode
    global h_min, s_min, v_min, h_max, s_max, v_max, LOWER_YELLOW, UPPER_YELLOW
    
    win_name = "Ultimate QCar Pilot"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    tuner_open = False
    qcar = QCar(readMode=1, frequency=CONTROLLER_UPDATE_RATE)
    cameras = QCarCameras(frameWidth=CSI_WIDTH, frameHeight=CSI_HEIGHT, frameRate=30, enableFront=True)
    
    with qcar, cameras:
        while not KILL_THREAD:
            qcar.read()
            if cameras.csiFront.read():
                frame = cameras.csiFront.imageData.copy()
                
                # --- TUNING UI ---
                if tuning_mode:
                    if not tuner_open:
                        cv2.namedWindow("HSV Tuner")
                        cv2.createTrackbar("H Min", "HSV Tuner", h_min, 179, nothing)
                        cv2.createTrackbar("S Min", "HSV Tuner", s_min, 255, nothing)
                        cv2.createTrackbar("V Min", "HSV Tuner", v_min, 255, nothing)
                        cv2.createTrackbar("H Max", "HSV Tuner", h_max, 179, nothing)
                        cv2.createTrackbar("S Max", "HSV Tuner", s_max, 255, nothing)
                        cv2.createTrackbar("V Max", "HSV Tuner", v_max, 255, nothing)
                        tuner_open = True
                    
                    h_min = cv2.getTrackbarPos("H Min", "HSV Tuner")
                    s_min = cv2.getTrackbarPos("S Min", "HSV Tuner")
                    v_min = cv2.getTrackbarPos("V Min", "HSV Tuner")
                    h_max = cv2.getTrackbarPos("H Max", "HSV Tuner")
                    s_max = cv2.getTrackbarPos("S Max", "HSV Tuner")
                    v_max = cv2.getTrackbarPos("V Max", "HSV Tuner")
                    LOWER_YELLOW = np.array([h_min, s_min, v_min])
                    UPPER_YELLOW = np.array([h_max, s_max, v_max])
                elif tuner_open:
                    cv2.destroyWindow("HSV Tuner")
                    tuner_open = False

                # --- PROCESS ---
                lat_err, conf, curve, target, mem, mask_c, edges = process_lane_logic(frame, lane_width_memory)
                lane_width_memory = mem
                
                # --- VIEW COMPOSITION ---
                # 'M' Key: If True, show B&W Edges. If False, show RGB.
                if show_mask:
                    disp = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) # Convert B&W to 3-channel
                else:
                    disp = frame.copy()
                
                disp = cv2.resize(disp, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                sx, sy = DISPLAY_WIDTH / CSI_WIDTH, DISPLAY_HEIGHT / CSI_HEIGHT
                
                # Overlays
                if curve is not None and len(curve) > 0:
                    scaled_curve = np.array([[int(p[0]*sx), int(p[1]*sy)] for p in curve])
                    cv2.polylines(disp, [scaled_curve], False, (0, 255, 255), 3)
                
                if show_roi:
                    roi_y = int(CSI_HEIGHT * 0.55 * sy)
                    cv2.line(disp, (0, roi_y), (DISPLAY_WIDTH, roi_y), (255, 0, 0), 1)
                
                if show_center and target:
                    tx, ty = int(target[0]*sx), int(target[1]*sy)
                    cv2.circle(disp, (tx, ty), 8, (0, 255, 0), -1)
                    cv2.line(disp, (DISPLAY_WIDTH//2, DISPLAY_HEIGHT), (tx, ty), (0, 255, 0), 2)

                # --- HUD TEXT (RESTORED) ---
                cv2.putText(disp, f"LAT ERR: {lat_err:.3f} m", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(disp, f"CONF: {conf:.2f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                mode_str = "TUNING" if tuning_mode else "DRIVE"
                cv2.putText(disp, f"MODE: {mode_str}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
                # Added THR and STR back:
                cv2.putText(disp, f"THR: {throttle:+.2f} | STR: {raw_steering:+.2f}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.imshow(win_name, disp)
                cv2.waitKey(1)
            
            qcar.write(throttle, raw_steering)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    keyboard.add_hotkey('up', lambda: update_thr(THROTTLE_STEP))
    keyboard.add_hotkey('down', lambda: update_thr(-THROTTLE_STEP))
    keyboard.add_hotkey('left', lambda: update_str(-STEERING_STEP))
    keyboard.add_hotkey('right', lambda: update_str(STEERING_STEP))
    keyboard.add_hotkey('q', quit_app)
    
    keyboard.add_hotkey('t', toggle_tuning)
    keyboard.add_hotkey('r', toggle_roi)
    keyboard.add_hotkey('m', toggle_mask)
    keyboard.add_hotkey('c', toggle_center)

    print("--- ULTIMATE QCAR PILOT STARTED ---")
    print("KEYS: [Arrows] Drive | [T] Tune HSV | [R/M/C] View Toggles | [Q] Quit")
    
    controlThread = Thread(target=controlLoop)
    controlThread.start()
    
    try:
        while controlThread.is_alive() and not KILL_THREAD: time.sleep(1)
    finally:
        KILL_THREAD = True