import sys
import os
import shutil
import glob
from roboflow import Roboflow
from ultralytics import YOLO
import torch

def verify_environment():
    print("Verifying Environment...")
    # 1. Enforce Python 3.12
    # We strictly check for 3.12 environment to fulfill user request.
    if sys.version_info.major != 3 or sys.version_info.minor != 12:
        raise RuntimeError(f"Strict requirement: Python 3.12 is required. Current is {sys.version_info.major}.{sys.version_info.minor}.")
    
    # 2. Check CUDA / Device availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! Please ensure this is running on the remote GPU (NVIDIA T4).")
    
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Verified GPU: {gpu_name}")
    if "T4" not in gpu_name:
        print("Warning: Target GPU is not an NVIDIA T4, but proceeding anyway.")

def download_data():
    print("Downloading data from Roboflow...")
    # Using the existing API key derived from workspace context
    rf = Roboflow(api_key="sfGRWRbhP29Y2TV8i6PO") 
    project = rf.workspace("vrishank-umrani-s-workspace").project("ppe-gzzdx-nf8ps")
    dataset = project.version(1).download("yolov8")
    return dataset.location

def train_model(data_path):
    print("Starting YOLOv8 Training...")
    data_yaml = os.path.join(data_path, "data.yaml")
    
    # Load YOLOv8-Nano
    model = YOLO("yolov8n.pt")
    
    # Run training
    # - 25 epochs
    # - imgsz = 640
    # - device = 0 (GPU)
    results = model.train(
        data=data_yaml,
        epochs=25,
        imgsz=640,
        device=0,
        project="runs",
        name="ppe_t4_run",
        exist_ok=True,
        verbose=True
    )
    
    return os.path.join("runs", "ppe_t4_run")

def export_submission(run_dir):
    print("Preparing export artifacts...")
    sub_dir = "Final_Submission"
    os.makedirs(sub_dir, exist_ok=True)
    
    # Gather best.pt
    best_weights = os.path.join(run_dir, "weights", "best.pt")
    if os.path.exists(best_weights):
        shutil.copy(best_weights, os.path.join(sub_dir, "best.pt"))
        print(f"Copied best.pt into {sub_dir}")
    else:
        print("Warning: best.pt not found.")
        
    # Gather key metrics and all generated images
    # YOLO automatically generates results.png and confusion_matrix.png among others
    images = glob.glob(os.path.join(run_dir, "*.png")) + glob.glob(os.path.join(run_dir, "*.jpg"))
    for img_path in images:
        img_name = os.path.basename(img_path)
        shutil.copy(img_path, os.path.join(sub_dir, img_name))
        print(f"Copied {img_name} into {sub_dir}")
        
    # Zip the Final_Submission folder
    shutil.make_archive(sub_dir, 'zip', sub_dir)
    print(f"Successfully packaged everything into {sub_dir}.zip!")

if __name__ == "__main__":
    print("--- GPU Accelerated PPE Detection Pipeline ---")
    verify_environment()
    data_loc = download_data()
    run_dir = train_model(data_loc)
    export_submission(run_dir)
    print("Pipeline Execution Completed Successfully.")
