from ultralytics import YOLO

# Source can be a file/folder
source = './HP-PK-Dataset/images/test/test300/'

# Load a pretrained YOLOv8n model from Ultralytics model
model = YOLO('./runs/detect/train/weights/best.pt')

# # Run inference on the source
results = model(source, stream=False, save=True, project="./runs", name="detect/predict")  # generator of Results objects


############################ PRINT PREDICTED NUMBER OF LABEL COUNTS ###############################

# Initialize counters for fire and smoke detections
car_count = 0
truck_count = 0
person_count = 0
fire_count = 0
smoke_count = 0
motorbike_count = 0
bicycle_count = 0
bus_count = 0
trafficcone_count = 0
licenseplate_count = 0
cleaningcart_count = 0
shoppingcart_count = 0

# Loop through the result objects
for result in results:
    # Check if boxes were detected
    if hasattr(result, "boxes") and result.boxes is not None:
        # Iterate through all detected class indices
        for cls in result.boxes.cls:
            cls_int = int(cls)
            # Get the corresponding class name using the result's names dictionary
            class_name = result.names[cls_int].lower()  # convert to lowercase for uniformity
            if class_name == "car":
                car_count += 1
            elif class_name == "truck":
                truck_count += 1
            elif class_name == "person":
                person_count += 1
            elif class_name == "fire":
                fire_count += 1
            elif class_name == "smoke":
                smoke_count += 1
            elif class_name == "motorbike":
                motorbike_count += 1
            elif class_name == "bicycle":
                bicycle_count += 1
            elif class_name == "bus":
                bus_count += 1
            elif class_name == "traffic-cone":
                trafficcone_count += 1
            elif class_name == "license-plate":
                licenseplate_count += 1
            elif class_name == "cleaning-cart":
                cleaningcart_count += 1
            elif class_name == "shopping-cart":
                shoppingcart_count += 1

# Print the total count
print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
print("    Total Number of Predicion in Each Label   ")
print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"Car : {car_count}")
print(f"Truck : {truck_count}")
print(f"Person : {person_count}")
print(f"Fire : {fire_count}")
print(f"Smoke : {smoke_count}")
print(f"Motor-bike : {motorbike_count}")
print(f"Bicycle : {bicycle_count}")
print(f"Bus : {bus_count}")
print(f"Traffic-cone : {trafficcone_count}")
print(f"License-plate : {licenseplate_count}")
print(f"Cleaning-cart : {cleaningcart_count}")
print(f"Shopping-cart : {shoppingcart_count}")


################################################################################################