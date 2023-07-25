import ultralytics
import numpy as np
from utils import REAL_WIDTH, START_WIDTH, END_WIDTH, FOCAL_LENGTH_PIXELS

model = ultralytics.YOLO('./../model/yolov8l.pt')

image_path = "./../data/carPOV/c2.jpeg"
detections = model.predict(source=image_path)

objects = ["car", "person", "truck", "traffic light", "traffic lights", "bicycle", "motorcycle", "bus", "train", "truck"]

# print(f'StartWidth : {START_WIDTH}\tendWidth : {END_WIDTH}')

for r in detections:
    if r.boxes is None:
        print(f'No Classes Detected')
        continue
    for box in r.boxes.data:
        label = int(box[-1].item())
        class_name = r.names.get(label)
        if class_name in objects:
            width = box[2] - box[0]
            height = box[3] - box[1]
            center_x = (box[2] + box[0]) / 2
            if START_WIDTH < center_x < END_WIDTH:
                observed_width_pixels = box[2] - box[0]
                distance = (REAL_WIDTH * FOCAL_LENGTH_PIXELS) / observed_width_pixels
                print(f'{class_name}\tFOV Pass\t(distance : {distance:.2f} meters)\tConfidence : {box[4]:.2f}')
            else:
                print(f'{class_name}\tFOV fail\t(OffsetBy : {min(abs(START_WIDTH-center_x), abs(END_WIDTH-center_x)):.2f} pixels)\tConfidence : {box[4]:.2f}')
