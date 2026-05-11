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

# --- Perspective Transform (BEV) Points ---
'''SRC_PTS = np.float32([
    [100, 410],       # Bottom-Left
    [720, 410],       # Bottom-Right
    [520, 265],       # Top-Right
    [320, 265]        # Top-Left
])'''

'''DST_PTS = np.float32([
    [150, 480],       # Bottom-Left
    [490, 480],       # Bottom-Right
    [490, 0],         # Top-Right
    [150, 0]          # Top-Left
])'''

SRC_PTS = np.float32([
    [109, 410],       # Bottom-Left
    [689, 410],       # Bottom-Right
    [491, 265],       # Top-Right
    [319, 265]        # Top-Left
])
DST_PTS = np.float32([[200, 480], [600, 480], [600, 0], [200, 0]])



M_WARP = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
M_INV = cv2.getPerspectiveTransform(DST_PTS, SRC_PTS)

# --- Lane Logic Defaults ---
# This represents the distance from the Yellow Line to the Center of the Right Lane
RIGHT_LANE_OFFSET_PX = 196

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
show_mask = False
show_bev = False  
show_src = False 
tuning_mode = False

# ================= Utility Functions =================
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)

def update_thr(d): global throttle; throttle = np.clip(throttle + d, -MAX_THROTTLE, MAX_THROTTLE)
def update_str(d): global raw_steering; raw_steering = np.clip(raw_steering + d, -MAX_STEERING, MAX_STEERING)
def toggle_mask(): global show_mask; show_mask = not show_mask
def toggle_bev(): global show_bev; show_bev = not show_bev
def toggle_src(): global show_src; show_src = not show_src
def quit_app(): global KILL_THREAD; KILL_THREAD = True
def toggle_tuning(): global tuning_mode; tuning_mode = not tuning_mode
def nothing(x): pass

# ================= Sliding Window Logic =================
# Searches full width, Peak-only Conf
def get_sliding_windows(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    base_x = np.argmax(histogram)
    
    nwindows = 10
    window_height = int(binary_warped.shape[0] // nwindows)
    margin = 80
    minpix = 30
    
    nonzero = binary_warped.nonzero()
    nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])
    
    current_x = base_x
    lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_x_low = current_x - margin
        win_x_high = current_x + margin
        
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                     (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        lane_inds.append(good_inds)
        if len(good_inds) > minpix:
            current_x = int(np.mean(nonzerox[good_inds]))

    if len(lane_inds) > 0:
        lane_inds = np.concatenate(lane_inds)
        return nonzerox[lane_inds], nonzeroy[lane_inds]
    return np.array([]), np.array([])

def process_lane_logic(img):
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, M_WARP, (w, h))
    hsv_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_warped, LOWER_YELLOW, UPPER_YELLOW)
    
    lat_err, conf, curve_pts, target_pt = 0.0, 0.0, None, None
    lane_x, lane_y = get_sliding_windows(mask)
    
    # if len(lane_x) > 0: print(f"Pixels Found: {len(lane_x)}")
    
    if len(lane_x) > 50:
        try:
            poly = np.polyfit(lane_y, lane_x, 2)
            plot_y = np.linspace(0, h-1, h)
            plot_x = poly[0]*plot_y**2 + poly[1]*plot_y + poly[2]
            curve_pts = np.column_stack((plot_x, plot_y)).astype(np.int32)
            
            # --- CALCULATING LATERAL ERROR IN METERS ---
            # 1. From QLabs setup: Lane Width = 2.5m.
            # 2. Distance from Yellow Line to Lane Center = 1.25m.
            # 3. Conversion: 1.25m / 245px = 0.0051 m/px.
            xm_per_pix = 1.25 / RIGHT_LANE_OFFSET_PX
            
            # bottom_x is the position of the Yellow Line
            bottom_x = int(np.polyval(poly, h))
            
            # target_x is the position of the Lane Center
            target_x = bottom_x + RIGHT_LANE_OFFSET_PX
            
            center_x = w // 2
            error_px = target_x - center_x
            
            # Resulting Error in Meters
            lat_err = error_px * xm_per_pix
            
            target_pt = (int(target_x), h - 10)
            conf = np.clip(len(lane_x) / 3000.0, 0.0, 1.0)
        except: pass
    
    return lat_err, conf, curve_pts, target_pt, mask, warped
 
# ================= Sliding Window Logic (UPDATED) ================= 
# Features: Intersection "Blinders" (ROI) + Curve-Safe Confidence Score   
'''def get_sliding_windows(binary_warped):
    h, w = binary_warped.shape
    
    # --- CONFIG ---
    nwindows = 10
    margin = 80       # Search width
    minpix = 30       # Minimum pixels to recenter window
    
    # ========================================================
    # 1. INTERSECTION FIX: ROI (Ignore Edges)
    # ========================================================
    # Only search the bottom half of the image
    search_h = h // 2
    
    # Ignore the far left and far right (20% of width)
    # This prevents locking onto curbs/sidewalks at intersections
    roi_side_margin = int(w * 0.20) # Ignores 20% of the screen on both sides.
    search_start = roi_side_margin
    search_end = w - roi_side_margin
    
    # Sum pixels vertically ONLY within safe central ROI
    histogram_roi = np.sum(binary_warped[search_h:, search_start:search_end], axis=0)
    
    if np.max(histogram_roi) == 0:
        return np.array([]), np.array([]), 0.0

    # Find peak inside ROI, then add offset back for global coordinate
    local_peak = np.argmax(histogram_roi)
    base_x = local_peak + search_start
    
    # ========================================================
    # 2. CURVE FIX: Confidence (Sum vs Max)
    # ========================================================
    # Sum pixels in a small window (+/- 10px) around the peak
    # This ensures slanted/curved lines still give high confidence
    conf_window = 20 
    
    h_start = max(0, local_peak - conf_window // 2)
    h_end   = min(len(histogram_roi), local_peak + conf_window // 2)
    
    total_pixels_in_peak = np.sum(histogram_roi[h_start:h_end]) / 255.0
    conf = np.clip(total_pixels_in_peak / search_h, 0.0, 1.0)

    # ========================================================
    # 3. SLIDING WINDOW LOGIC
    # ========================================================
    window_height = int(h // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    current_x = base_x
    lane_inds = []

    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_x_low = current_x - margin
        win_x_high = current_x + margin
        
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                     (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        
        lane_inds.append(good_inds)
        
        if len(good_inds) > minpix:
            current_x = int(np.mean(nonzerox[good_inds]))

    lane_inds = np.concatenate(lane_inds) if len(lane_inds) > 0 else []
    
    if len(lane_inds) == 0:
        return np.array([]), np.array([]), 0.0
        
    lx = nonzerox[lane_inds]
    ly = nonzeroy[lane_inds]
    
    return lx, ly, conf
    
def process_lane_logic(img):
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, M_WARP, (w, h))
    hsv_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_warped, LOWER_YELLOW, UPPER_YELLOW)
    
    # Initialize defaults
    lat_err, final_conf, curve_pts, target_pt = 0.0, 0.0, None, None
    
    # --- CHANGED: Unpack 3 values now ---
    lane_x, lane_y, start_conf = get_sliding_windows(mask)
    
    if len(lane_x) > 50:
        try:
            poly = np.polyfit(lane_y, lane_x, 2)
            plot_y = np.linspace(0, h-1, h)
            plot_x = poly[0]*plot_y**2 + poly[1]*plot_y + poly[2]
            curve_pts = np.column_stack((plot_x, plot_y)).astype(np.int32)
            
            # Meter Conversion
            xm_per_pix = 1.25 / RIGHT_LANE_OFFSET_PX
            
            bottom_x = int(np.polyval(poly, h))
            target_x = bottom_x + RIGHT_LANE_OFFSET_PX
            
            center_x = w // 2
            error_px = target_x - center_x
            lat_err = error_px * xm_per_pix
            
            target_pt = (int(target_x), h - 10)

            # --- HYBRID CONFIDENCE ---
            # We combine the "Start Consistency" (start_conf) 
            # with the "Total Pixels Found" (len(lane_x))
            # If the start is solid, we trust it.
            pix_conf = np.clip(len(lane_x) / 3000.0, 0.0, 1.0)
            
            # You can average them, or just use one. 
            # Averaging is safest:
            final_conf = (start_conf + pix_conf) / 2.0
            
        except: 
            pass
    
    return lat_err, final_conf, curve_pts, target_pt, mask, warped'''
    


# ================= Main Loop =================
def controlLoop():
    global KILL_THREAD, throttle, raw_steering, tuning_mode, show_bev, show_src
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
                        cv2.createTrackbar("H Min", "HSV Tuner", h_min, 179, nothing); cv2.createTrackbar("S Min", "HSV Tuner", s_min, 255, nothing)
                        cv2.createTrackbar("V Min", "HSV Tuner", v_min, 255, nothing); cv2.createTrackbar("H Max", "HSV Tuner", h_max, 179, nothing)
                        cv2.createTrackbar("S Max", "HSV Tuner", s_max, 255, nothing); cv2.createTrackbar("V Max", "HSV Tuner", v_max, 255, nothing)
                        tuner_open = True
                    h_min, s_min, v_min = cv2.getTrackbarPos("H Min", "HSV Tuner"), cv2.getTrackbarPos("S Min", "HSV Tuner"), cv2.getTrackbarPos("V Min", "HSV Tuner")
                    h_max, s_max, v_max = cv2.getTrackbarPos("H Max", "HSV Tuner"), cv2.getTrackbarPos("S Max", "HSV Tuner"), cv2.getTrackbarPos("V Max", "HSV Tuner")
                    LOWER_YELLOW, UPPER_YELLOW = np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max])
                elif tuner_open:
                    cv2.destroyWindow("HSV Tuner"); tuner_open = False

                lat_err, conf, curve, target, mask_img, bev_img = process_lane_logic(frame)
                
                # --- VIEW COMPOSITION ---
                if show_bev:
                    disp = bev_img.copy()
                    if show_mask: disp = cv2.bitwise_and(disp, disp, mask=mask_img)
                    if curve is not None:
                        cv2.polylines(disp, [curve], False, (0, 255, 255), 5)
                        if target: cv2.circle(disp, target, 10, (0, 255, 0), -1)
                else:
                    disp = frame.copy()
                    # Show SRC Points (The Trapezoid)
                    if show_src:
                        pts = SRC_PTS.astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(disp, [pts], True, (0, 255, 0), 3)
                    
                    # Project curve back to Camera View
                    if curve is not None:
                        proj_curve = cv2.perspectiveTransform(curve.reshape(-1, 1, 2).astype(np.float32), M_INV)
                        cv2.polylines(disp, [proj_curve.astype(np.int32)], False, (0, 255, 255), 3)

                disp_resized = cv2.resize(disp, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                # Display Error in Meters
                cv2.putText(disp_resized, f"LAT ERR: {lat_err:.3f} m | CONF: {conf:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(disp_resized, f"VIEW: {'BEV' if show_bev else 'CAM'} | SRC BOX: {'ON' if show_src else 'OFF'}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(disp_resized, f"THR: {throttle:+.2f} | STR: {raw_steering:+.2f}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.imshow(win_name, disp_resized)
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
    keyboard.add_hotkey('m', toggle_mask)
    keyboard.add_hotkey('p', toggle_bev)
    keyboard.add_hotkey('s', toggle_src)

    print("--- BEV QCAR PILOT STARTED ---")
    print("KEYS: [P] BEV View | [S] Toggle SRC Box | [T] Tune | [Q] Quit")
    
    controlThread = Thread(target=controlLoop); controlThread.start()
    try:
        while controlThread.is_alive() and not KILL_THREAD: time.sleep(1)
    finally: KILL_THREAD = True