import os
import signal
import numpy as np
from threading import Thread
import time
import cv2
from pal.products.qcar import QCar, QCarCameras
import keyboard

# ================= Configuration =================
CONTROLLER_UPDATE_RATE = 100
CSI_WIDTH, CSI_HEIGHT = 820, 410
DISPLAY_WIDTH, DISPLAY_HEIGHT = 820, 410
WARP_W, WARP_H = 800, 480
REAL_LANE_WIDTH = 2.5 # Actual width of the road in meters

# --- File Paths ---
CALIB_FILE = "data_calibration.npy"
WIDTH_FILE = "data_width.npy"
HSV_FILE   = "data_hsv.npy"

# --- Warp Constants ---
DST_PTS = np.float32([[200, 480], [600, 480], [600, 0], [200, 0]])
DEFAULT_SRC = np.float32([[100, 410], [720, 410], [520, 265], [320, 265]])

# ================= State Management =================
class State:
    def __init__(self):
        self.kill = False
        self.mode = "DRIVE"
        self.view_bev = False 
        self.show_roi = True    
        self.show_mask = False  
        self.show_overlay = True 
        
        # Internal flags for window management
        self.hsv_init = False
        
        # Physics / Control
        self.throttle = 0.0
        self.steering = 0.0
        self.calib_pt_idx = 3
        
        # Data
        self.src_pts = np.copy(DEFAULT_SRC)
        self.lane_width = 570
        self.hsv_lower = np.array([15, 50, 50])
        self.hsv_upper = np.array([45, 255, 255])

    def load_data(self):
        if os.path.exists(CALIB_FILE):
            try: self.src_pts = np.load(CALIB_FILE)
            except: pass
        if os.path.exists(WIDTH_FILE):
            try: self.lane_width = int(np.load(WIDTH_FILE)[0])
            except: pass
        if os.path.exists(HSV_FILE):
            try:
                hsv = np.load(HSV_FILE)
                self.hsv_lower = hsv[0]
                self.hsv_upper = hsv[1]
            except: pass
        print(f"Loaded Data | Width: {self.lane_width}px | HSV Loaded")
        current_m_per_pix = REAL_LANE_WIDTH / self.lane_width
        print(f">> M_PER_PIX: {current_m_per_pix:.6f} (based on width {self.lane_width})")

state = State()
state.load_data()

# ================= Helper Functions =================
def nothing(x): pass

def sig_handler(*args): state.kill = True
signal.signal(signal.SIGINT, sig_handler)

def save_hsv():
    data = np.array([state.hsv_lower, state.hsv_upper])
    np.save(HSV_FILE, data)
    print(">> SAVED HSV DATA.")

def save_calib():
    np.save(CALIB_FILE, state.src_pts)
    print(">> SAVED WARP POINTS.")

def save_width():
    np.save(WIDTH_FILE, np.array([state.lane_width]))
    print(f">> SAVED LANE WIDTH ({state.lane_width}px).")

# ================= Toggles & Key Handling =================
def toggle_view_bev(): state.view_bev = not state.view_bev
def toggle_roi(): state.show_roi = not state.show_roi
def toggle_mask(): state.show_mask = not state.show_mask
def toggle_overlay(): state.show_overlay = not state.show_overlay

def toggle_calib_mode():
    if state.mode == "CALIB":
        state.mode = "DRIVE"
        save_calib()
        print("--- CALIBRATION CLOSED ---")
    else:
        state.mode = "CALIB"
        state.throttle = 0; state.steering = 0
        print("--- CALIBRATION MODE ---")

def toggle_width_mode():
    if state.mode == "WIDTH":
        state.mode = "DRIVE"
        save_width()
        print("--- WIDTH TUNING CLOSED ---")
    else:
        state.mode = "WIDTH"
        state.throttle = 0; state.steering = 0
        print("--- WIDTH TUNING MODE ---")

def toggle_hsv_tuning():
    # JUST CHANGE STATE HERE - DO NOT TOUCH OPENCV WINDOWS
    if state.mode == "TUNING":
        state.mode = "DRIVE"
        save_hsv()
        print("--- HSV TUNING CLOSED ---")
    else:
        state.mode = "TUNING"
        state.throttle = 0; state.steering = 0
        print("--- HSV TUNING MODE ---")

def cycle_point(): state.calib_pt_idx = (state.calib_pt_idx + 1) % 4

def handle_arrow(key):
    if state.mode == "CALIB":
        step = 2
        if key == 'up': state.src_pts[state.calib_pt_idx][1] -= step
        elif key == 'down': state.src_pts[state.calib_pt_idx][1] += step
        elif key == 'left': state.src_pts[state.calib_pt_idx][0] -= step
        elif key == 'right': state.src_pts[state.calib_pt_idx][0] += step
    
    elif state.mode == "WIDTH":
        step = 1
        if key == 'left': state.lane_width -= step
        elif key == 'right': state.lane_width += step
        print(f"Width: {state.lane_width}px")
    
    elif state.mode == "DRIVE":
        # --- SLOW SPEED STEP ---
        t_step = 0.02
        s_step = 0.1
        
        if key == 'up': state.throttle = np.clip(state.throttle + t_step, -0.3, 0.3)
        elif key == 'down': state.throttle = np.clip(state.throttle - t_step, -0.3, 0.3)
        elif key == 'left': state.steering = np.clip(state.steering - s_step, -0.6, 0.6)
        elif key == 'right': state.steering = np.clip(state.steering + s_step, -0.6, 0.6)

# ================= Vision Pipeline =================
def get_sliding_windows(binary_warped):
    # Search bottom half only
    search_h = binary_warped.shape[0] // 2
    histogram = np.sum(binary_warped[search_h:, :], axis=0)
    
    if np.max(histogram) == 0: 
        return np.array([]), np.array([]), 0.0

    base_x = np.argmax(histogram)
    
    # --- CONFIDENCE CALC ---
    pixel_count = np.max(histogram) / 255.0
    conf = np.clip(pixel_count / search_h, 0.0, 1.0)
    
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    margin = 80
    lane_inds = ((nonzerox > (base_x - margin)) & (nonzerox < (base_x + margin)))
    
    return nonzerox[lane_inds], nonzeroy[lane_inds], conf
 
# ================= Sliding Window Logic (UPDATED) =================
# Features: Intersection "Blinders" (ROI) + Curve-Safe Confidence Score 
'''def get_sliding_windows(binary_warped):
    h, w = binary_warped.shape
    
    # --- CONFIG ---
    nwindows = 10
    margin = 80       # Width of the search box for sliding windows
    minpix = 30       # Min pixels found to recenter the window
    
    # ========================================================
    # 1. SEARCH AREA (Intersection Safety Fix)
    # ========================================================
    # Only search the bottom half of the image
    search_h = h // 2
    
    # Ignore the far left and far right edges (e.g., 20% of width on each side)
    # This prevents the car from locking onto a curb or crossing line at an intersection
    roi_side_margin = int(w * 0.20) # Ignores 20% of the screen on both sides.
    search_start = roi_side_margin
    search_end = w - roi_side_margin
    
    # ========================================================
    # 2. HISTOGRAM & PEAK FINDING
    # ========================================================
    # Sum pixels vertically, BUT ONLY within our safe central ROI
    histogram_roi = np.sum(binary_warped[search_h:, search_start:search_end], axis=0)
    
    if np.max(histogram_roi) == 0:
        return np.array([]), np.array([]), 0.0

    # Find the peak index inside the ROI, then add the offset back 
    # so 'base_x' is correct for the full image
    local_peak = np.argmax(histogram_roi)
    base_x = local_peak + search_start
    
    # ========================================================
    # 3. CONFIDENCE CALCULATION (Curve/Slant Fix)
    # ========================================================
    # Instead of checking just one column (which fails on slanted lines),
    # we sum the pixels in a small window (+/- 10px) around the peak.
    conf_window = 20 
    
    h_start = max(0, local_peak - conf_window // 2)
    h_end   = min(len(histogram_roi), local_peak + conf_window // 2)
    
    # "How many white pixels are in this lane chunk?"
    total_pixels_in_peak = np.sum(histogram_roi[h_start:h_end]) / 255.0
    
    # Normalize: Compare pixels found vs. Height of search area
    # 1.0 = Solid Vertical Line | 0.5 = Dashed Line
    conf = np.clip(total_pixels_in_peak / search_h, 0.0, 1.0)

    # ========================================================
    # 4. SLIDING WINDOW LOGIC (Standard)
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
    
    return lx, ly, conf'''

def process_vision(img, M_warp):
    warped = cv2.warpPerspective(img, M_warp, (WARP_W, WARP_H))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, state.hsv_lower, state.hsv_upper)
    
    lx, ly, conf = get_sliding_windows(mask)
    
    curve_data = None
    target_pt = None
    lat_err = 0.0
    
    m_per_pix = REAL_LANE_WIDTH / max(1, state.lane_width) # Calculate meters per pixel dynamically
    
    if len(lx) > 50:
        try:
            poly = np.polyfit(ly, lx, 2)
            plot_y = np.linspace(0, WARP_H-1, WARP_H)
            plot_x = poly[0]*plot_y**2 + poly[1]*plot_y + poly[2]
            curve_data = np.column_stack((plot_x, plot_y)).astype(np.int32)
            
            yellow_bottom = np.polyval(poly, WARP_H)
            half_width = state.lane_width / 2.0
            target_x = yellow_bottom + half_width
            
            car_center = WARP_W // 2
            err_px = target_x - car_center
            lat_err = err_px * m_per_pix
            # lat_err = err_px * 0.005
            
            target_pt = np.array([int(target_x), WARP_H - 50])
        except: pass

    return warped, mask, lat_err, conf, curve_data, target_pt

# ================= Main Loop =================
def controlLoop():
    cv2.namedWindow("QCar View", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("QCar View", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    
    qcar = QCar(readMode=1, frequency=CONTROLLER_UPDATE_RATE)
    cameras = QCarCameras(frameWidth=CSI_WIDTH, frameHeight=CSI_HEIGHT, frameRate=30, enableFront=True)
    
    # Background for Tuner Window
    tuner_bg = np.zeros((300, 400, 3), dtype=np.uint8)
    
    with qcar, cameras:
        while not state.kill:
            qcar.read()
            if cameras.csiFront.read():
                frame = cameras.csiFront.imageData.copy()
                
                # ============ WINDOW MANAGEMENT (THREAD SAFE) ============
                # If in Tuning Mode but window not created yet:
                if state.mode == "TUNING" and not state.hsv_init:
                    cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("HSV Tuner", 400, 350)
                    cv2.createTrackbar("H Min", "HSV Tuner", state.hsv_lower[0], 179, nothing)
                    cv2.createTrackbar("S Min", "HSV Tuner", state.hsv_lower[1], 255, nothing)
                    cv2.createTrackbar("V Min", "HSV Tuner", state.hsv_lower[2], 255, nothing)
                    cv2.createTrackbar("H Max", "HSV Tuner", state.hsv_upper[0], 179, nothing)
                    cv2.createTrackbar("S Max", "HSV Tuner", state.hsv_upper[1], 255, nothing)
                    cv2.createTrackbar("V Max", "HSV Tuner", state.hsv_upper[2], 255, nothing)
                    state.hsv_init = True
                
                # If NOT in Tuning Mode but window still exists:
                elif state.mode != "TUNING" and state.hsv_init:
                    cv2.destroyWindow("HSV Tuner")
                    state.hsv_init = False
                
                # If Window is Open, Update logic
                if state.hsv_init:
                    try:
                        h_min = cv2.getTrackbarPos("H Min", "HSV Tuner")
                        s_min = cv2.getTrackbarPos("S Min", "HSV Tuner")
                        v_min = cv2.getTrackbarPos("V Min", "HSV Tuner")
                        h_max = cv2.getTrackbarPos("H Max", "HSV Tuner")
                        s_max = cv2.getTrackbarPos("S Max", "HSV Tuner")
                        v_max = cv2.getTrackbarPos("V Max", "HSV Tuner")
                        state.hsv_lower = np.array([h_min, s_min, v_min])
                        state.hsv_upper = np.array([h_max, s_max, v_max])
                        
                        # Keep window refreshed
                        cv2.imshow("HSV Tuner", tuner_bg)
                    except:
                        # If user closed window manually
                        state.mode = "DRIVE"
                        state.hsv_init = False

                # ========================================================

                # --- Vision Processing ---
                M = cv2.getPerspectiveTransform(state.src_pts, DST_PTS)
                Minv = cv2.getPerspectiveTransform(DST_PTS, state.src_pts)
                
                warped, mask, err, conf, curve, target = process_vision(frame, M)
                
                # --- Drawing / UI ---
                if state.show_mask:
                    display_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                elif state.view_bev:
                    display_img = warped.copy()
                else:
                    display_img = frame.copy()
                
                # ROI
                if state.show_roi and not state.show_mask:
                    if not state.view_bev:
                        pts = state.src_pts.astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(display_img, [pts], True, (0, 165, 255), 2)
                    else:
                        cv2.rectangle(display_img, (200, 0), (600, 480), (50, 50, 50), 2)

                # Overlay
                if state.show_overlay and not state.show_mask:
                    CLR_YEL = (0, 255, 255)
                    CLR_RED = (0, 0, 255)
                    CLR_GRN = (0, 255, 0)
                    
                    if state.view_bev:
                        cx = WARP_W // 2
                        hw = int(state.lane_width / 2)
                        cv2.line(display_img, (cx - hw, 0), (cx - hw, 480), CLR_YEL, 1)
                        cv2.line(display_img, (cx + hw, 0), (cx + hw, 480), (200, 200, 200), 1)
                        cv2.circle(display_img, (cx, 400), 5, CLR_GRN, -1)

                        if curve is not None:
                            cv2.polylines(display_img, [curve], False, CLR_RED, 3)
                            if target is not None:
                                cv2.circle(display_img, tuple(target), 8, CLR_RED, -1)
                                cv2.line(display_img, (cx, 400), tuple(target), CLR_GRN, 2)
                    else:
                        cx_bev = np.array([[[WARP_W//2, 400]]], dtype=np.float32)
                        cx_raw = cv2.perspectiveTransform(cx_bev, Minv).astype(int)[0][0]
                        cv2.circle(display_img, tuple(cx_raw), 5, CLR_GRN, -1)

                        if curve is not None:
                            curve_float = curve.astype(np.float32).reshape(-1, 1, 2)
                            curve_raw = cv2.perspectiveTransform(curve_float, Minv).astype(int)
                            cv2.polylines(display_img, [curve_raw], False, CLR_RED, 3)
                            if target is not None:
                                tgt_float = target.astype(np.float32).reshape(-1, 1, 2)
                                tgt_raw = cv2.perspectiveTransform(tgt_float, Minv).astype(int)[0][0]
                                cv2.circle(display_img, tuple(tgt_raw), 8, CLR_RED, -1)
                                cv2.line(display_img, tuple(cx_raw), tuple(tgt_raw), CLR_GRN, 2)

                # Calib Dots
                if state.mode == "CALIB" and not state.view_bev and not state.show_mask:
                    pt = tuple(state.src_pts[state.calib_pt_idx].astype(int))
                    cv2.circle(display_img, pt, 8, (0, 0, 255), -1)

                # --- TEXT OVERLAYS ---
                LIGHT_RED = (100, 100, 255)
                LIGHT_GREEN = (100, 255, 100)
                WHITE = (255, 255, 255)
                
                # Top Line: Red (ERR, CONF)
                top_str = f"ERR: {err:+.3f}m  CONF: {conf:.2f}"
                cv2.putText(display_img, top_str, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
                
                # Bottom Line: Green (THR, STR, MODE)
                bot_str = f"THR: {state.throttle:.2f}  STR: {state.steering:.2f}  [{state.mode}]"
                cv2.putText(display_img, bot_str, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

                cv2.imshow("QCar View", display_img)
                cv2.waitKey(1)
            
            qcar.write(state.throttle, state.steering)
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Arrow keys
    keyboard.add_hotkey('up', lambda: handle_arrow('up'))
    keyboard.add_hotkey('down', lambda: handle_arrow('down'))
    keyboard.add_hotkey('left', lambda: handle_arrow('left'))
    keyboard.add_hotkey('right', lambda: handle_arrow('right'))
    
    # Modes
    keyboard.add_hotkey('c', toggle_calib_mode)
    keyboard.add_hotkey('t', toggle_hsv_tuning)
    keyboard.add_hotkey('w', toggle_width_mode)
    
    # Toggles
    keyboard.add_hotkey('p', toggle_view_bev)
    keyboard.add_hotkey('r', toggle_roi)
    keyboard.add_hotkey('m', toggle_mask)
    keyboard.add_hotkey('o', toggle_overlay)
    
    # Utils
    keyboard.add_hotkey('tab', cycle_point)
    keyboard.add_hotkey('q', lambda: setattr(state, 'kill', True))

    print("--- QCAR CONTROLLER STARTED ---")
    t = Thread(target=controlLoop)
    t.start()
    
    while t.is_alive() and not state.kill:
        time.sleep(1)
    state.kill = True