import cv2
from ultralytics import YOLO

# 1. Path to your trained model (Make sure to place 'best.pt' inside this same folder!)
MODEL_PATH = r"best.pt"

# 2. Path to the image or video you want to test (Use 'r' before the string to ignore backslashes)
TEST_FILE = r"D:\PPE\dataset\images.jpg"

print("Loading model...")
model = YOLO(MODEL_PATH)

print(f"Running detection on {TEST_FILE}...")
# 3. Run inference on the file
results = model(TEST_FILE)

# 4. Extract the original raw image
original_img = results[0].orig_img
height, width = original_img.shape[:2]

# 5. Increase image dimensions by 2x FIRST so we can draw much smaller text relative to the screen size!
larger_image = cv2.resize(original_img, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC)

# 6. Manually draw the bounding boxes and confidence scores using OpenCV for ultimate control
class_names = model.names
colors = {0: (0, 200, 100), 1: (255, 150, 0), 2: (200, 100, 50), 3: (0, 100, 255)}

for box in results[0].boxes:
    # Scale the coordinates up 2x to match the new image size
    x1, y1, x2, y2 = [int(v * 2) for v in box.xyxy[0]]
    conf = float(box.conf[0])
    cls_id = int(box.cls[0])
    label = f"{class_names[cls_id]} {conf:.2f}"
    
    color = colors.get(cls_id, (0, 255, 0))
    
    # Draw ultra-thin boxes
    cv2.rectangle(larger_image, (x1, y1), (x2, y2), color, 2)
    
    # Draw tiny, crisp text to ensure no overlaps but 100% authenticity
    cv2.putText(larger_image, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

# 7. Display the custom annotated image in a dedicated OpenCV window
cv2.imshow("PPE Detection - Image Test (Authentic Custom Overlay)", larger_image)

print("\nPress ANY KEY while clicking on the image window to close it.")
cv2.waitKey(0) # This officially freezes the window open until you press a key!
cv2.destroyAllWindows()

