from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from PIL import Image
import io
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
CORS(app)

# --- 모델 로드 ---
try:
    model = YOLO("hemletYoloV8_100epochs.pt")
    class_names = model.names
    print(f"✅ 모델 로드 성공. 클래스: {class_names}")
    HEAD_CLASS_ID = 0
    HELMET_CLASS_ID = 1
except Exception as e:
    print(f"❌ 모델 로드 중 오류 발생: {e}")
    model = None

@socketio.on('connect')
def handle_connect():
    print('✅ 클라이언트가 연결되었습니다.')

@socketio.on('disconnect')
def handle_disconnect():
    print('❌ 클라이언트 연결이 끊겼습니다.')

@socketio.on('analyze_frame')
def handle_analyze_frame(data_url):
    """클라이언트로부터 받은 비디오 프레임을 분석하고 결과를 반환합니다."""
    if not model:
        return

    try:
        # 1. 클라이언트로부터 데이터 수신 확인
        # print(f"[{time.strftime('%H:%M:%S')}] 📸 프레임 수신")

        header, encoded = data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        
        image = Image.open(io.BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 2. YOLO 모델 분석 수행
        results = model(frame, verbose=False)
        
        boxes = results[0].boxes
        
        analysis_results = {
            "boxes": boxes.xyxyn.tolist(),
            "classes": boxes.cls.tolist(),
            "conf": boxes.conf.tolist()
        }

        # 3. 클라이언트로 결과 전송
        emit('analysis_result', analysis_results)
        # print(f"[{time.strftime('%H:%M:%S')}] 🚀 분석 결과 전송 완료")

    except Exception as e:
        print(f"❌ 프레임 처리 중 오류 발생: {e}")


if __name__ == '__main__':
    print("🚀 클라이언트 중심 분석 서버를 시작합니다.")
    socketio.run(app, host='0.0.0.0', port=5000)
