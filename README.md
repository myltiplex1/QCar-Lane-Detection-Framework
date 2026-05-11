# QCar-Lane-Detection-Framework
Advanced lane detection and lane-following algorithms for the Quanser QCar in QLabs simulation environment using OpenCV, Bird's Eye View (BEV) transforms, Sliding Window tracking, and Hough Line detection.

---

# Features

- Bird's Eye View (BEV) lane detection
- Perspective transform calibration
- Sliding window lane tracking
- Hough Line-based lane detection
- Temporal smoothing and lane memory
- Real-time HSV threshold tuning
- Persistent calibration saving/loading
- Lane curvature estimation
- Lateral error estimation in meters
- OpenCV visualization overlays and HUD
- QLabs + QCar integration

---

# Repository Structure

```bash
.
├── script1- 1_calib_bev_sliding_window_em.py
├── script2- 2_fixed_bev_sliding_wind_basic_pull.py
├── script3- 3_fixed_bev_sliding_wind_tweaked_p.py
├── script4- 4_hough_raw_view_with_memory_full.py
├── script5- 5_hough_raw_view_reactive_f.py
├── calibration/
│   ├── data_calibration.npy
│   ├── data_hsv.npy
|   ├── data_width.npy
│   └── view_npy.py
└── README.md
```

---

# Implemented Lane Detection Pipelines

## 1. Calibration-Focused BEV Sliding Window Detector

**Overview**

The most advanced and feature-rich implementation in the repository.

This script applies a perspective transform to generate a Bird's Eye View (BEV) of the road, making lane lines appear parallel for easier tracking. Detection is performed using HSV masking and a sliding window search.

**Key Features**
- Interactive calibration system
- Warp point calibration
- Lane width calibration
- HSV threshold tuning
- Persistent calibration storage using .npy
- Dynamic meters-per-pixel scaling
- Multiple visualization modes
- Real-time overlays and HUD
- Manual steering/throttle controls

**Best Use Case**

Recommended for accurate lane tracking and autonomous driving experiments.

---

## 2. Simple Fixed-Warp BEV Sliding Window

**Overview**

A simplified BEV detector with hard-coded perspective transform parameters.

**Features**
- Fixed warp matrix
- Sliding window lane tracking
- Constant meters-per-pixel scaling
- Basic confidence estimation
- Curve projection onto camera image

**Best Use Case**

Quick testing and stable simulation environments where the camera setup does not change.

---

## 3. Minor-Tweaked Fixed-Warp BEV Sliding Window

**Overview**

A slightly modified version of Script 2 with minor geometric refinements.

**Features**
- Adjusted warp points
- Small offset modifications
- Experimental ROI/blinder logic

**Best Use Case**

Testing small geometric tuning improvements.

---

## 4. Hough-Line Detector with Temporal Memory

**Overview**

A robust lane detector based on Hough Line Transform operating directly on the raw camera image.

**Processing Pipeline**
```
HSV Thresholding
    ↓
Canny Edge Detection
    ↓
Region of Interest
    ↓
HoughLinesP
    ↓
Polynomial Fitting
```

**Features**
- Temporal smoothing
- Lane memory system
- Lost-lane recovery
- Stable lane estimation
- Real-time status HUD
- Handles glare and shadows effectively

**Best Use Case**

Environments with unstable lighting, dashed lanes, or temporary line loss.

---

## 5. Pure Reactive Hough-Line Detector

**Overview**

A lightweight and highly responsive Hough-based detector without smoothing or memory.

**Features**
- Frame-by-frame processing
- Adaptive polynomial fitting
- Fast response time
- Minimal latency

**Best Use Case**

High-speed experiments where responsiveness is preferred over stability.

---

# Comparison Summary

| Script | Method | Robustness | Responsiveness | Calibration | Complexity |
|--------|--------|-----------|-----------------|-------------|-----------|
| Script 1 | BEV + Sliding Window | High | Medium | Full | High |
| Script 2 | Fixed BEV | Medium | Medium | None | Medium |
| Script 3 | Tweaked BEV | Medium | Medium | None | Medium |
| Script 4 | Hough + Memory | Very High | Low-Medium | Minimal | Medium |
| Script 5 | Reactive Hough | Low | Very High | Minimal | Low |

---

# Recommended Pipeline

## Best Overall: Script 1

The calibration-focused BEV detector provides the best balance of:

- Geometric accuracy
- Reliable lateral error estimation
- Calibration flexibility
- Visualization tools
- Long-term usability

It is the recommended implementation for most autonomous driving applications in QLabs.

---

# Technologies Used

- Python
- OpenCV
- NumPy
- Quanser QLabs
- Quanser QCar API

---

# Requirements

Ensure:

- Quanser QLabs is installed and running
- QCar simulation environment is configured correctly
- Camera streaming is enabled

---

# Demo

Add screenshots, GIFs, or simulation videos here.
