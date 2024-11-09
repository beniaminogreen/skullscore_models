from ultralytics import YOLO

# load model
model = YOLO('runs/detect/train5/weights/best.pt') # replace with your custom model pt file

# export model
model.export(format="onnx", dynamic=True)
