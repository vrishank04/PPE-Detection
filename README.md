# PPE Detection Project

## Overview

A computer vision-based project designed to detect Personal Protective Equipment (PPE) such as hard hats, safety vests, safety glasses, and boots in real-time using deep learning and object detection models (e.g., YOLOv8).

---

## Features

* **Real-Time Detection:** Processes video streams or webcams for instant compliance monitoring.
* **Multiple Class Detection:** Identifies key safety gear (Hard Hats, Vests, Masks, Gloves, Boots).
* **Alert System:** Triggers warnings or logs violations when mandatory PPE is missing.
* **High Accuracy:** Built on state-of-the-art YOLO architectures optimized for speed and precision.

---

## Tech Stack

* **Language:** Python 3.8+
* **Framework:** PyTorch / Ultralytics YOLO
* **Computer Vision:** OpenCV
* **Data Processing:** NumPy, Pandas

---

## Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/ppe-detection.git
cd ppe-detection

```


2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



---

## Usage

### 1. Training the Model

To train the YOLO model on your custom PPE dataset:

```bash
yolo task=detect mode=train model=yolov8n.pt data=data.yaml epochs=50 imgsz=640

```

### 2. Running Inference

To run detection on a webcam feed:

```bash
python detect.py --source 0

```

To run detection on a video file:

```bash
python detect.py --source path/to/video.mp4

```

---

## Project Structure

```text
ppe-detection/
│
├── dataset/            # Training and validation images/labels
├── models/             # Saved model weights (.pt files)
├── utils/              # Helper scripts for preprocessing and metrics
├── detect.py           # Main script for running inference
├── requirements.txt    # Project dependencies
└── data.yaml           # Dataset configuration file

```

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
