from ultralytics import YOLO

# Load YOLOv10n model from scratch
model = YOLO("best.pt")

# Train the model
model.train(data="skull_data.yaml", epochs=200, imgsz=640)
