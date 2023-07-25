from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official detection model
model = YOLO('yolov8n-seg.pt')  # load an official segmentation model

# Track with the model
results = model.track(source="https://www.youtube.com/watch?v=wqctLW0Hb_0", show=True)
results = model.track(source="https://www.youtube.com/watch?v=wqctLW0Hb_0", show=True, tracker="bytetrack.yaml")
