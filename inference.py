import time
from ultralytics import YOLO

model_path = "hemletYoloV8_100epochs.pt"
video_path = "helmet.mp4"

print("[INFO] Loading model...")
model = YOLO(model_path)

while True:
    print("[INFO] Running inference on video...")
    results = model(video_path, save=True, project="runs", name="result", exist_ok=True)
    print("[INFO] Inference complete. Waiting 60 seconds before next run...\n")
    time.sleep(60)  # 1분마다 반복 실행
