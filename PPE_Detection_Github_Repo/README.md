# YOLOv8 PPE Compliance Monitor

A robust computer vision pipeline designed to enforce Personal Protective Equipment (PPE) compliance in real-time. This project leverages a highly optimized YOLOv8-Nano model, fine-tuned on custom datasets to detect Hardhats, Vests, and Gloves via live webcam feeds and static media.

## Key Features
- **Real-Time Webcam Inference (`ppe_realtime_detect.py`)**: Streams camera data through YOLOv8, displaying bounding boxes and dynamic on-screen warnings for missing safety equipment.
- **Custom Authentic Static Rendering (`ppe_static_detect.py`)**: Bypasses overlapping YOLO default visuals by using custom OpenCV drawing for micro-scale, authentic confidence readouts on static imagery.
- **Cloud-Accelerated Training Pipelines**: Includes both automated scripts (`remote_t4_training.py`) and Google Colab Jupyter Notebooks strictly tuned for NVIDIA T4 execution on Python 3.12.

## Installation
1. Clone the repository
2. Install the necessary machine learning libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Update `MODEL_PATH` in the detection scripts to point to your `best.pt` weights.

## Usage
To test the real-time detection on your web camera:
```bash
python ppe_realtime_detect.py
```
To test on static images or videos without clutter:
```bash
python ppe_static_detect.py
```
