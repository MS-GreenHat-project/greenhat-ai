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
# 로컬 개발 시 eventlet이 설치되어 있어야 합니다.
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

# Windows에서 직접 실행하기 위한 코드
if __name__ == '__main__':
    print("🚀 Windows 개발 서버 시작 중... http://127.0.0.1:5000")
    # host='0.0.0.0'으로 설정하여 같은 네트워크의 다른 기기에서도 접속 가능
    socketio.run(app, host='0.0.0.0', port=5000)
