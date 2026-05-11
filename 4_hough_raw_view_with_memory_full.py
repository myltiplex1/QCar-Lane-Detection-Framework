"""
ultimate_qcar_pilot_final_v2.py
The definitive autonomous driving script for QCar.
--------------------------------------------------
FEATURES:
1. STABILITY: Temporal Smoothing (0.9) to stop Green Dot jitter.
2. MEMORY: Drives 'blind' for ~1s if line disappears (shadows/glare).
3. ORIGINAL CONFIDENCE: Uses vertical pixel spread for score.
4. TUNING: Press 'T' to adjust HSV in real-time.

CONTROLS:
- Arrow Keys: Manual Override
- 'T': Toggle HSV Tuning
- 'R/M/C': Toggle ROI / Mask / Center Views
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
RIGHT_LANE_OFFSET_PX = 285   # Target offset from right line (300)

# --- Stabilization & Memory Settings ---
SMOOTHING_FACTOR = 0.9       # 0.9 = Very smooth (less jitter)
prev_poly = None              # History of curve
prev_target_x = None          # History of green dot position

last_valid_lat_err = 0.0      # Last good steering value
last_valid_curve = None       # Last good visual curve
last_valid_conf = 0.0         # Last good confidence score
lost_lane_count = 0           # How many frames have we lost the line?
MAX_LOST_FRAMES = 15          # Drive "blind" for ~1 sec before stopping

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
show_mask = False 
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

# ================= Advanced Lane Algorithm (Stable + Memory + Old Conf) =================
'''def process_lane_logic(img, current_memory):
    # Bring in all necessary globals for memory and smoothing
    global prev_poly, prev_target_x, SMOOTHING_FACTOR
    global last_valid_lat_err, last_valid_curve, last_valid_conf, lost_lane_count, MAX_LOST_FRAMES

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

    # --- DECISION LOGIC ---
    
    # CASE 1: Line Detected (Normal Operation)
    if len(all_x) > 10:
        lost_lane_count = 0  # We found the line, reset counter
        
        try:
            # Fit new polynomial
            current_poly = np.polyfit(all_y, all_x, 2)

            # --- SMOOTHING ---
            if prev_poly is not None:
                # Weighted Average: (New * Alpha) + (Old * Beta)
                alpha = 1.0 - SMOOTHING_FACTOR 
                averaged_poly = (alpha * current_poly) + (SMOOTHING_FACTOR * prev_poly)
            else:
                averaged_poly = current_poly
            
            # Save history
            prev_poly = averaged_poly
            
            # Generate Visuals
            plot_y = np.linspace(roi_top, h, 20)
            plot_x = np.polyval(averaged_poly, plot_y).astype(int)
            curve_pts = np.column_stack((plot_x, plot_y)).astype(np.int32)
            
            # Calculate Target
            bottom_x = int(np.polyval(averaged_poly, h))
            raw_target_x = bottom_x + RIGHT_LANE_OFFSET_PX
            
            # Smooth Target Position
            if prev_target_x is not None:
                smooth_target_x = int(((1.0 - SMOOTHING_FACTOR) * raw_target_x) + (SMOOTHING_FACTOR * prev_target_x))
            else:
                smooth_target_x = raw_target_x
            
            prev_target_x = smooth_target_x
            target_pt = (smooth_target_x, h)

            # Calculate Lateral Error
            center_x = w // 2
            error_px = smooth_target_x - center_x
            # lat_err = np.clip(error_px / (w / 2.0), -1.0, 1.0)
            xm_per_pix = 1.25 / RIGHT_LANE_OFFSET_PX
            lat_err = error_px * xm_per_pix
            
            # --- ORIGINAL CONFIDENCE CALCULATION ---
            # Based on how "tall" the line is (vertical spread)
            spread = (max(all_y) - min(all_y)) / (h - roi_top)
            conf = min(1.0, 0.5 + spread * 0.5)
            
            # Save Valid Data for Persistence
            last_valid_lat_err = lat_err
            last_valid_curve = curve_pts
            last_valid_conf = conf

        except:
            # Calculation failed (rare), treat as lost frame
            lost_lane_count += 1
            lat_err = last_valid_lat_err
            curve_pts = last_valid_curve
            target_pt = (prev_target_x, h) if prev_target_x else None
            conf = last_valid_conf * 0.8 # Degrade confidence slightly

    # CASE 2: No Line Detected (Dropout Protection)
    else:
        lost_lane_count += 1
        
        if lost_lane_count < MAX_LOST_FRAMES:
            # "I'm blind, but I remember where the road was!"
            lat_err = last_valid_lat_err
            curve_pts = last_valid_curve
            
            if prev_target_x is not None:
                target_pt = (prev_target_x, h)
            else:
                target_pt = None
            
            # Use last valid confidence, but lower it to indicate "Memory Mode"
            conf = last_valid_conf * 0.8 
        else:
            # Lost for too long (over 1 sec), reset to safety
            lat_err = 0.0
            curve_pts = None
            target_pt = None
            conf = 0.0

    return lat_err, conf, curve_pts, target_pt, new_memory, yellow_mask, masked_edges'''
    
    
'''def process_lane_logic(img, current_memory):
    global prev_poly, prev_target_x, SMOOTHING_FACTOR
    global last_valid_lat_err, last_valid_curve, last_valid_conf, lost_lane_count, MAX_LOST_FRAMES

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

    # ==========================================================
    # FIX: THICKEN THE LINE (DILATION)
    # This fills small holes in the yellow line so it doesn't flicker.
    # ==========================================================
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=2) # "Fat" line is easier to see

    # 1. Edge Detection & ROI
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=yellow_mask)
    
    # Reduced Canny thresholds to see faint lines
    edges = cv2.Canny(masked_gray, 30, 100) 
    
    roi_top = int(h * 0.55)
    vertices = np.array([[(0, h), (w, h), (w//2 + 50, roi_top), (w//2 - 50, roi_top)]], np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # ==========================================================
    # FIX: RELAX HOUGH PARAMETERS
    # Threshold 20 -> 10 (Accept weaker lines)
    # MaxGap 150 -> 200 (Connect lines across bigger gaps)
    # ==========================================================
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 10, minLineLength=15, maxLineGap=200)
    
    lat_err = 0.0
    conf = 0.0
    curve_pts = None
    target_pt = None
    new_memory = current_memory
    
    all_x, all_y = [], []
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 == 0: continue 
                slope = (y2 - y1) / (x2 - x1)
                
                # Keep slope filter to ignore intersection curbs
                if abs(slope) < 0.2:
                    continue

                all_x.extend([x1, x2])
                all_y.extend([y1, y2])

    # --- DECISION LOGIC ---
    if len(all_x) > 20:
        lost_lane_count = 0 
        
        try:
            y_spread = max(all_y) - min(all_y)
            
            # Anti-Shake: Use Degree 1 (Straight) if line is short (<100px)
            degree = 1 if y_spread < 100 else 2
            current_poly = np.polyfit(all_y, all_x, degree)
            
            if degree == 1:
                current_poly = [0, current_poly[0], current_poly[1]] 

            # SMOOTHING
            if prev_poly is not None:
                alpha = 1.0 - SMOOTHING_FACTOR 
                averaged_poly = (alpha * current_poly) + (SMOOTHING_FACTOR * prev_poly)
            else:
                averaged_poly = current_poly
            
            prev_poly = averaged_poly
            
            # Visuals
            plot_y = np.linspace(roi_top, h, 20)
            plot_x = np.polyval(averaged_poly, plot_y).astype(int)
            curve_pts = np.column_stack((plot_x, plot_y)).astype(np.int32)
            
            # Target
            bottom_x = int(np.polyval(averaged_poly, h))
            raw_target_x = bottom_x + RIGHT_LANE_OFFSET_PX
            
            if prev_target_x is not None:
                smooth_target_x = int(((1.0 - SMOOTHING_FACTOR) * raw_target_x) + (SMOOTHING_FACTOR * prev_target_x))
            else:
                smooth_target_x = raw_target_x
            
            prev_target_x = smooth_target_x
            target_pt = (smooth_target_x, h)

            # --- LATERAL ERROR (METERS) ---
            center_x = w // 2
            error_px = smooth_target_x - center_x
            xm_per_pix = 1.25 / RIGHT_LANE_OFFSET_PX 
            lat_err = error_px * xm_per_pix 
            
            # Confidence
            spread = (max(all_y) - min(all_y)) / (h - roi_top)
            conf = min(1.0, 0.5 + spread * 0.5)
            
            last_valid_lat_err = lat_err
            last_valid_curve = curve_pts
            last_valid_conf = conf

        except:
            lost_lane_count += 1
            lat_err = last_valid_lat_err
            curve_pts = last_valid_curve
            target_pt = (prev_target_x, h) if prev_target_x else None
            conf = last_valid_conf * 0.8 

    else:
        lost_lane_count += 1
        if lost_lane_count < MAX_LOST_FRAMES:
            lat_err = last_valid_lat_err
            curve_pts = last_valid_curve
            target_pt = (prev_target_x, h) if prev_target_x else None
            conf = last_valid_conf * 0.8 
        else:
            lat_err = 0.0
            curve_pts = None
            target_pt = None
            conf = 0.0

    return lat_err, conf, curve_pts, target_pt, new_memory, yellow_mask, masked_edges'''
    
    
    
def process_lane_logic(img, current_memory):
    global prev_poly, prev_target_x, SMOOTHING_FACTOR
    global last_valid_lat_err, last_valid_curve, last_valid_conf, lost_lane_count, MAX_LOST_FRAMES

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

    # Dilation helps fill small holes, but we won't overdo it
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=1) 

    # 1. Edge Detection & ROI
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=yellow_mask)
    edges = cv2.Canny(masked_gray, 30, 100)
    
    roi_top = int(h * 0.55)
    vertices = np.array([[(0, h), (w, h), (w//2 + 50, roi_top), (w//2 - 50, roi_top)]], np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Reduced maxLineGap from 200 -> 50. 
    # This prevents it from connecting the Left Line to the Right Curb.
    lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180, 10, minLineLength=15, maxLineGap=50)
    
    lat_err = 0.0
    conf = 0.0
    curve_pts = None
    target_pt = None
    new_memory = current_memory
    
    all_x, all_y = [], []
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 - x1 == 0: continue 
                slope = (y2 - y1) / (x2 - x1)
                
                # Filter 1: Ignore Intersection Curbs (Horizontal lines)
                if abs(slope) < 0.2: continue

                # Filter 2: IGNORE RIGHT SIDE NOISE
                # If the yellow blob is on the Right half of the screen (> w/2), 
                # it is NOT the center line. It is a curb or noise. Kill it.INCR TO LESSEN EFFECT
                if x1 > (w * 0.65) or x2 > (w * 0.65):
                    continue

                all_x.extend([x1, x2])
                all_y.extend([y1, y2])

    # --- DECISION LOGIC ---
    if len(all_x) > 20:
        lost_lane_count = 0 
        
        try:
            y_spread = max(all_y) - min(all_y)
            
            # Anti-Shake: Straight line for short detections
            degree = 1 if y_spread < 80 else 2
            current_poly = np.polyfit(all_y, all_x, degree)
            
            if degree == 1:
                current_poly = [0, current_poly[0], current_poly[1]] 

            # SMOOTHING
            if prev_poly is not None:
                alpha = 1.0 - SMOOTHING_FACTOR 
                averaged_poly = (alpha * current_poly) + (SMOOTHING_FACTOR * prev_poly)
            else:
                averaged_poly = current_poly
            
            prev_poly = averaged_poly
            
            # Visuals
            plot_y = np.linspace(roi_top, h, 20)
            plot_x = np.polyval(averaged_poly, plot_y).astype(int)
            curve_pts = np.column_stack((plot_x, plot_y)).astype(np.int32)
            
            # Target
            bottom_x = int(np.polyval(averaged_poly, h))
            raw_target_x = bottom_x + RIGHT_LANE_OFFSET_PX
            
            if prev_target_x is not None:
                smooth_target_x = int(((1.0 - SMOOTHING_FACTOR) * raw_target_x) + (SMOOTHING_FACTOR * prev_target_x))
            else:
                smooth_target_x = raw_target_x
            
            prev_target_x = smooth_target_x
            target_pt = (smooth_target_x, h)

            # --- LATERAL ERROR (METERS) ---
            center_x = w // 2
            error_px = smooth_target_x - center_x
            xm_per_pix = 1.25 / RIGHT_LANE_OFFSET_PX 
            lat_err = error_px * xm_per_pix 
            
            conf = min(1.0, 0.5 + (y_spread / (h - roi_top)) * 0.5)
            
            last_valid_lat_err = lat_err
            last_valid_curve = curve_pts
            last_valid_conf = conf

        except:
            # Calculation failed
            lost_lane_count += 1
            lat_err = last_valid_lat_err
            curve_pts = last_valid_curve
            target_pt = (prev_target_x, h) if prev_target_x else None
            conf = last_valid_conf * 0.8 

    else:
        # No lines found after filtering
        lost_lane_count += 1
        if lost_lane_count < MAX_LOST_FRAMES:
            lat_err = last_valid_lat_err
            curve_pts = last_valid_curve
            target_pt = (prev_target_x, h) if prev_target_x else None
            conf = last_valid_conf * 0.8 
        else:
            lat_err = 0.0
            curve_pts = None
            target_pt = None
            conf = 0.0

    return lat_err, conf, curve_pts, target_pt, new_memory, yellow_mask, masked_edges

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
                if show_mask:
                    disp = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) 
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

                # --- HUD TEXT ---
                state_text = "DRIVE"
                if lost_lane_count > 0: state_text = "MEMORY (LOST)"
                if lost_lane_count >= MAX_LOST_FRAMES: state_text = "NO LINE"
                if tuning_mode: state_text = "TUNING"

                cv2.putText(disp, f"LAT ERR: {lat_err:.3f} m", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(disp, f"CONF: {conf:.2f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(disp, f"STATUS: {state_text}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(disp, f"THR: {throttle:+.2f} | STR: {raw_steering:+.2f}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

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

    print("--- ULTIMATE QCAR PILOT (FINAL v2) STARTED ---")
    print("KEYS: [Arrows] Drive | [T] Tune HSV | [R/M/C] View Toggles | [Q] Quit")
    
    controlThread = Thread(target=controlLoop)
    controlThread.start()
    
    try:
        while controlThread.is_alive() and not KILL_THREAD: time.sleep(1)
    finally:
        KILL_THREAD = True