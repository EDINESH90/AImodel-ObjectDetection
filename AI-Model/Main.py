from ultralytics import YOLO
import time, os
from pathlib import Path

# Record start time
start_time = time.time()

############################### MODEL_NAME: YOLO-V8 MODEL #######################################
# Load a model
model = YOLO("yolov8s.yaml") # build a new model from YAML


# #################################### MODULE: TRAINING #########################################
# Checking Results for HP-PK-Dataset
results = model.train(data="AI-Model/cfg/datasets/Detection-12cls.yaml", 
                      batch=16, epochs=100, imgsz=640, device=0, project="./runs", name="detect/train")
end_time1 = time.time()

# Calculate elapsed time
tr_elapsed_time = end_time1 - start_time
# print(f"Elapsed time for TRAINING: {tr_elapsed_time:.2f} seconds")



# #################################### MODULE: TESTING #########################################
# Checking Results for HP-PK-Dataset
model = YOLO('runs/detect/train/weights/best.pt')
model = model.val(data="AI-Model/cfg/datasets/Detection-12cls.yaml", 
                  split="test", project="./runs", name="detect/val")
end_time2 = time.time()

# Calculate elapsed time
elapsed_time2 = end_time2 - start_time
test_elapsed_time = end_time2 - end_time1
# print(f"Elapsed time for TESTING: {test_elapsed_time:.2f} seconds")


# ## ## ------------------------------------------------------------------------------------

# Calculate overall elapsed time
end_time = time.time()
overall_elapsed_time = end_time - start_time
print("--------------------------------------------------")
print("               SUMMARY OF TIME DURATION (Seconds)               ")
print("--------------------------------------------------")
print(f"Elapsed time for TRAINING: {tr_elapsed_time:.2f} sec")
print(f"Elapsed time for TESTING: {test_elapsed_time:.2f} sec")
print("--------------------------------------------------")
print(f"Elapsed time for OVERALL (Tr + Test): {overall_elapsed_time:.2f} sec")
print("--------------------------------------------------")


# ###################### EXPORT ONNX MODEL ###########################
# Export the trained model to ONNX format
# yolo export model=YoloV8s.pt format=onnx imgsz=640 opset=12
# yolo export model=./runs/detect/train/weights/best.pt format=onnx imgsz=640 opset=12



################################################################################################


