import cv2
from ultralytics import YOLO

# 1. Path to your trained model
MODEL_PATH = r"best.pt"

# 2. Path to the image or video you want to test (Use 'r' before the string to ignore backslashes)
TEST_FILE = r"image.png"

print("Loading model...")
model = YOLO(MODEL_PATH)

print(f"Running detection on {TEST_FILE}...")
# 3. Run inference on the file
results = model(TEST_FILE)

# 4. Extract the original raw image
annotated_image = results[0].orig_img.copy()

# 5. Manually draw the bounding boxes and confidence scores using OpenCV for ultimate control
class_names = model.names
colors = {0: (0, 200, 100), 1: (255, 150, 0), 2: (200, 100, 50), 3: (0, 100, 255)}

for box in results[0].boxes:
    # Use native coordinates without scaling
    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])
    label = f"{class_names[cls_id]} {conf:.2f}"
    
    color = colors.get(cls_id, (0, 255, 0))
    
    # Draw ultra-thin boxes
    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
    
    # Draw tiny, crisp text (0.5 scale) to ensure no overlaps
    cv2.putText(annotated_image, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# 6. Display the custom annotated image in a dedicated OpenCV window
cv2.namedWindow("PPE Detection - Image Test", cv2.WINDOW_NORMAL) # Allows you to drag and resize the window!
cv2.imshow("PPE Detection - Image Test", annotated_image)

print("\nPress ANY KEY while clicking on the image window to close it.")
cv2.waitKey(0) # This officially freezes the window open until you press a key!
cv2.destroyAllWindows()

