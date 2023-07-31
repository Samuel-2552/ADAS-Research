from ultralytics import YOLO
 
model = YOLO("yolov8x.pt")  # load a pretrained YOLOv8n model
 




model.train(data="coco128.yaml")  # train the model
model.val()  # evaluate model performance on the validation set
model.predict(source="car4.jpg")  # predict on an image
model.export(format="onnx")  # export the model to ONNX format

