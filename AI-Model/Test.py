from ultralytics import YOLO
import time, os
from pathlib import Path

# Record start time
start_time = time.time()


# ######################## MODULE: TESTING #############################
# Checking Results for HP-PK-Dataset
model = YOLO('runs/detect/train/weights/best.pt')
model = model.val(data="AI-Model/cfg/datasets/HP-PK-DS-12cls.yaml", 
                  split="test", project="./runs", name="detect/val")
end_time2 = time.time()

# Calculate elapsed time
test_elapsed_time = end_time2 - start_time
print(f"Elapsed time for TESTING: {test_elapsed_time:.2f} seconds")

########################################################################