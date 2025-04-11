import os
import urllib.request
from ultralytics import YOLO

MODEL_DIR = os.path.join("model_files", "yolo")
ENGINES_DIR = "engines"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ENGINES_DIR, exist_ok=True)

model_name = "yolo11n"
pt_path = os.path.join(MODEL_DIR, f"{model_name}.pt")
engine_path = os.path.join(ENGINES_DIR, f"{model_name}.engine")
download_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"

if not os.path.exists(pt_path):
    print(f"{pt_path} not found. Downloading...")
    urllib.request.urlretrieve(download_url, pt_path)
    print("Download complete.")

if not os.path.exists(engine_path):
    print(f"{engine_path} not found. Exporting engine...")
    model = YOLO(pt_path)
    model.export(format="engine", batch=1, half=True, data="coco.yaml")

    default_engine_path = f"{model_name}_engine.engine"
    if os.path.exists(default_engine_path):
        os.rename(default_engine_path, engine_path)
        print(f"Engine saved to: {engine_path}")
    else:
        print("Engine export failed or file not found.")
else:
    print(f"Engine already exists at: {engine_path}, skipping export.")
