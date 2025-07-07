FROM python:3.9-slim

# 컨테이너 내의 작업 디렉토리 설정
WORKDIR /app

# --- ✅ FIX: OpenCV 실행에 필요한 시스템 라이브러리 설치 ---
# libgl1-mesa-glx는 headless 환경에서 OpenCV가 의존하는 그래픽 라이브러리입니다.
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# 필요한 라이브러리 목록 파일을 먼저 복사
COPY requirements.txt .

# 라이브러리 설치 (Gunicorn 포함)
# --no-cache-dir 옵션으로 불필요한 캐시를 남기지 않아 이미지 크기를 줄입니다.
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 모든 프로젝트 파일을 컨테이너로 복사
# (app.py, hemletYoloV8_100epochs.pt 등)
COPY . .

# 컨테이너가 5000번 포트를 사용함을 명시
EXPOSE 5000

CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--timeout", "120", "--bind", "0.0.0.0:5000", "app:app"]
