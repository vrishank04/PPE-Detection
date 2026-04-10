"""
PPE Compliance Real-Time Detection
===================================
Uses YOLOv8 (best.pt) + OpenCV to detect PPE items via webcam.
Highlights missing PPE with on-screen warnings.

Classes: glove (0), helmet (1), pants (2), vest (3)

Controls:
  - Press 'q' to quit
  - Press 's' to save a screenshot
"""

import cv2
import time
from ultralytics import YOLO

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_PATH = r"best.pt"  # Place your best.pt file in this identical folder to run!
CONFIDENCE_THRESHOLD = 0.40
WEBCAM_INDEX = 0  # Change to 1 if you have multiple cameras

# Class names and their display colors (BGR)
CLASS_COLORS = {
    "glove":  (0, 200, 100),   # green
    "helmet": (255, 150, 0),   # blue-ish
    "pants":  (200, 100, 50),  # teal
    "vest":   (0, 100, 255),   # orange
}

# Required PPE items — if any of these are missing, show a warning
REQUIRED_PPE = {"helmet", "vest", "glove"}

# Warning styling
WARNING_COLOR = (0, 0, 255)       # red
SAFE_COLOR    = (0, 220, 0)       # green
FONT          = cv2.FONT_HERSHEY_SIMPLEX


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
def draw_detection(frame, box, label, conf, color):
    """Draw a bounding box with label on the frame."""
    x1, y1, x2, y2 = map(int, box)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Label background
    text = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    
    # Label text
    cv2.putText(frame, text, (x1 + 3, y1 - 5), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_status_panel(frame, detected_ppe, fps):
    """Draw a PPE compliance status panel at the top of the frame."""
    h, w = frame.shape[:2]
    panel_h = 90
    
    # Semi-transparent dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, panel_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Title
    cv2.putText(frame, "PPE COMPLIANCE MONITOR", (10, 25), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 25), FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    
    # PPE status indicators
    missing_items = REQUIRED_PPE - detected_ppe
    x_offset = 10
    
    for item in sorted(REQUIRED_PPE):
        if item in detected_ppe:
            status_color = SAFE_COLOR
            icon = "[OK]"
        else:
            status_color = WARNING_COLOR
            icon = "[!!]"
        
        text = f"{icon} {item.upper()}"
        cv2.putText(frame, text, (x_offset, 55), FONT, 0.55, status_color, 2, cv2.LINE_AA)
        x_offset += 160
    
    # Overall compliance status
    if missing_items:
        missing_str = ", ".join(m.upper() for m in sorted(missing_items))
        warning_text = f"WARNING: Missing {missing_str}"
        cv2.putText(frame, warning_text, (10, 82), FONT, 0.55, WARNING_COLOR, 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "ALL PPE DETECTED - COMPLIANT", (10, 82), FONT, 0.55, SAFE_COLOR, 2, cv2.LINE_AA)


def draw_warning_flash(frame, missing_items, frame_count):
    """Draw a flashing border when PPE is missing."""
    if not missing_items:
        return
    
    # Flash every 15 frames
    if (frame_count // 15) % 2 == 0:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), WARNING_COLOR, 4)


# ──────────────────────────────────────────────
# Main Detection Loop
# ──────────────────────────────────────────────
def main():
    print("Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)
    class_names = model.names  # {0: 'glove', 1: 'helmet', 2: 'pants', 3: 'vest'}
    print(f"Model loaded. Classes: {class_names}")
    
    print(f"Opening webcam (index={WEBCAM_INDEX})...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check your camera connection.")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Real-time detection started. Press 'q' to quit, 's' to screenshot.")
    
    frame_count = 0
    prev_time = time.time()
    fps = 0.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame. Exiting...")
            break
        
        frame_count += 1
        
        # ── Run YOLOv8 inference ──
        results = model.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )
        
        # ── Parse detections ──
        detected_ppe = set()
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = class_names[cls_id]
                coords = box.xyxy[0].tolist()
                
                # Track detected PPE types
                detected_ppe.add(label)
                
                # Get color for this class
                color = CLASS_COLORS.get(label, (200, 200, 200))
                
                # Draw bounding box
                draw_detection(frame, coords, label, conf, color)
        
        # ── Calculate FPS ──
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time
        
        # ── Draw UI overlays ──
        missing_items = REQUIRED_PPE - detected_ppe
        draw_status_panel(frame, detected_ppe, fps)
        draw_warning_flash(frame, missing_items, frame_count)
        
        # ── Display ──
        cv2.imshow("PPE Compliance Monitor - YOLOv8", frame)
        
        # ── Key handling ──
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            screenshot_path = f"ppe_screenshot_{frame_count}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"Screenshot saved: {screenshot_path}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")


if __name__ == "__main__":
    main()
