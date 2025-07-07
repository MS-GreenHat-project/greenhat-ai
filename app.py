from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from PIL import Image
import io
import torch

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
CORS(app)

# 모델 로드
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO("hemletYoloV8_100epochs.pt").to(device)
    class_names = model.names
    print(f"✅ 모델 로드 성공. 클래스: {class_names} | 디바이스: {device}")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    model = None

@socketio.on('connect')
def connect():
    print("✅ 클라이언트 연결됨.")

@socketio.on('disconnect')
def disconnect():
    print("❌ 클라이언트 연결 종료.")

@socketio.on('analyze_frame')
def analyze_frame(data_url):
    if not model:
        return

    try:
        header, encoded = data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        results = model(frame, verbose=False, imgsz=416)
        boxes = results[0].boxes

        conf_threshold = 0.5
        mask = boxes.conf > conf_threshold

        emit('analysis_result', {
            "boxes": boxes.xyxyn[mask].tolist(),
            "classes": boxes.cls[mask].tolist(),
            "conf": boxes.conf[mask].tolist()
        })
    except Exception as e:
        print(f"❌ 분석 중 오류: {e}")