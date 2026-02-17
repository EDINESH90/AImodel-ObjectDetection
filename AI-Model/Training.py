from ultralytics import YOLO
import time, os
from pathlib import Path

# Record start time
start_time = time.time()

################################## TEST-UP YOLO-V8 MODEL #######################################
# Load a model
model = YOLO("yolov8s.yaml") # build a new model from YAML


# #################################### MODULE: TRAINING #########################################
# Checking Results for HP-PK-Dataset
results = model.train(data="AI-Model/cfg/datasets/Detection-12cls.yaml", 
                      batch=16, epochs=100, imgsz=640, device=0, project="./runs", name="detect/train")
end_time1 = time.time()

# Calculate elapsed time
tr_elapsed_time = end_time1 - start_time
print(f"Elapsed time for TRAINING: {tr_elapsed_time:.2f} seconds")

################################################################################################



