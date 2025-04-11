import os
import urllib.request
from ultralytics import YOLO

model_path = "yolo11n.pt"
download_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"

if not os.path.exists(model_path):
    print(f"{model_path} not found. Downloading...")
    urllib.request.urlretrieve(download_url, model_path)
    print("Download complete.")

model = YOLO(model_path)
model.export(format="engine", batch=1, half=True, data="coco.yaml")
