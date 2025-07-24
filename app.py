from flask import Flask, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from PIL import Image
import io
import torch
import logging
import os
import datetime
import time
from azure.storage.filedatalake import DataLakeServiceClient
import requests # --- [추가] Discord 알림을 위한 라이브러리 ---

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="https://datacat.store", async_mode='eventlet', path='/socket.io/')
CORS(app)

# --- Azure DataLake 연결 설정 ---
try:
    AZURE_CONNECTION_STRING = os.environ.get("AZURE_CONNECTION_STRING")
    AZURE_CONTAINER_NAME = os.environ.get("AZURE_CONTAINER_NAME", "training-data")

    if not AZURE_CONNECTION_STRING:
        logger.warning("⚠️ Azure 연결 문자열(환경 변수)이 설정되지 않았습니다.")
        datalake_service_client = None
    else:
        datalake_service_client = DataLakeServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        logger.info(f"✅ Azure DataLake 서비스에 성공적으로 연결되었습니다. 타겟 컨테이너: {AZURE_CONTAINER_NAME}")
except Exception as e:
    logger.error(f"❌ Azure 연결 중 오류 발생: {e}")
    datalake_service_client = None

# --- [추가] Discord Webhook 설정 ---
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1397814713768873984/Vltt1J7wdG4vsTLAZMmvfigcN8aE5Lby6Jxo3Azp8_kmwvcTc1WqfHBu2InjB6rrIhOX"
if not DISCORD_WEBHOOK_URL:
    logger.warning("⚠️ Discord Webhook URL(환경 변수)이 설정되지 않았습니다. 알림이 전송되지 않습니다.")
else:
    logger.info("✅ Discord Webhook이 설정되었습니다.")

# --- 인-메모리 쿨다운 관리 ---
last_capture_times = {}
CAPTURE_COOLDOWN_SECONDS = 10

# --- [추가] Discord 알림 전송 함수 ---
def send_discord_notification(class_name, file_name=None):
    if not DISCORD_WEBHOOK_URL:
        return # Webhook URL이 없으면 함수 종료

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Discord에 보낼 메시지 (임베드 형식으로 가독성 향상)
    embed = {
        "title": "🚨 안전모 미착용 감지!",
        "description": f"**감지 시간:** {timestamp}\n**감지 유형:** {class_name}",
        "color": 15158332,  # 빨간색
        "footer": {"text": "DataCat Safety Monitoring System"}
    }
    
    # 업로드된 파일 정보가 있으면 메시지에 추가
    if file_name:
        embed["description"] += f"\n**저장된 파일:** `{file_name}`"

    # 전송할 데이터
    data = {
        "content": "@here **[경고]** 안전모 미착용이 감지되었습니다!", # 채널의 모든 사람을 호출
        "embeds": [embed]
    }

    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data)
        response.raise_for_status() # HTTP 오류 발생 시 예외 처리
        logger.info(f"✅ Discord 알림 전송 성공.")
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Discord 알림 전송 실패: {e}")


def upload_to_datalake(file_system_name, frame_data, class_name):
    if not datalake_service_client:
        logger.error("❌ 업로드 시도 실패: datalake_service_client가 None입니다. AZURE_CONNECTION_STRING 환경 변수를 확인하세요.")
        return None
    try:
        file_system_client = datalake_service_client.get_file_system_client(file_system=file_system_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = f"raw/{class_name}_{timestamp}.jpg"
        
        file_client = file_system_client.get_file_client(file_name)
        file_client.create_file()
        file_client.append_data(data=frame_data, offset=0, length=len(frame_data))
        file_client.flush_data(len(frame_data))
        
        logger.info(f"✅ 파일 업로드 성공: {file_system_name}/{file_name}")
        return file_name
    except Exception as e:
        logger.error(f"❌ DataLake 업로드 실패: {e}")
        return None

# --- Health Probe 및 모델 로드 ---
@app.route('/health')
def azure_health_probe():
    return 'OK', 200
@app.route('/')
def azure_probe():
    return 'OK', 200
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "best.pt" if os.path.exists("best.pt") else "hemletYoloV8_100epochs.pt"
    model = YOLO(model_path).to(device)
    class_names = model.names   
    logger.info(f"✅ 모델 로드 성공: {model_path}")
    logger.info(f"📊 클래스: {class_names} | 디바이스: {device}")
except Exception as e:
    logger.error(f"❌ 모델 로드 실패: {e}")
    model = None

# --- Socket.IO 이벤트 핸들러 ---
@socketio.on('connect')
def connect():
    logger.info(f"✅ 클라이언트 연결됨. SID: {request.sid}")

@socketio.on('disconnect')
def disconnect():
    if request.sid in last_capture_times:
        del last_capture_times[request.sid]
    logger.info(f"❌ 클라이언트 연결 종료. SID: {request.sid}")

@socketio.on('analyze_frame')
def analyze_frame(data_url):
    if not model:
        return

    session_id = request.sid
    try:
        current_time = time.time()
        last_capture = last_capture_times.get(session_id)

        header, encoded = data_url.split(",", 1)
        image_data = base64.b64decode(encoded)

        image = Image.open(io.BytesIO(image_data))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = model(frame, verbose=False, imgsz=416)
        boxes = results[0].boxes

        conf_threshold = 0.5
        mask = boxes.conf > conf_threshold
        detected_classes = boxes.cls[mask].tolist()

        # head(0.0) 또는 helmet(1.0) 감지 시
        if 0.0 in detected_classes or 1.0 in detected_classes:
            upload_class_name = "head" if 0.0 in detected_classes else "helmet"

            # 쿨다운 조건 확인
            if last_capture is None or (current_time - last_capture) > CAPTURE_COOLDOWN_SECONDS:
                logger.info(f"🚨 [{session_id}] '{upload_class_name}' 감지! 업로드를 시도합니다.")
                
                uploaded_file = upload_to_datalake(
                    file_system_name=AZURE_CONTAINER_NAME,
                    frame_data=image_data,
                    class_name=upload_class_name
                )
                last_capture_times[session_id] = current_time

                # --- [수정] 'head' 감지 시에만 Discord 알림 전송 ---
                if upload_class_name == "head":
                    logger.info("👷‍♂️ 안전모 미착용 확인! Discord 알림을 전송합니다.")
                    send_discord_notification(class_name="안전모 미착용(Head)", file_name=uploaded_file)
                # --------------------------------------------

                if uploaded_file:
                    emit('upload_result', {"status": "success", "file": uploaded_file})
                else:
                    emit('upload_result', {"status": "fail"})
            else:
                logger.info(f"[DEBUG] 쿨다운 때문에 업로드 및 알림 건너뜀. "
                            f"(남은 시간: {CAPTURE_COOLDOWN_SECONDS - (current_time - last_capture):.1f}초)")

        # 분석 결과 emit
        emit('analysis_result', {
            "boxes": boxes.xyxyn[mask].tolist(),
            "classes": detected_classes,
            "conf": boxes.conf[mask].tolist()
        })
    except Exception as e:
        logger.error(f"❌ 분석 중 오류: {e}")