from ultralytics import YOLO
import tqdm
import glob

model = YOLO("runs/detect/train2/weights/best.pt")

for img in glob.iglob("test_images/"):
    model(img, save=True)

