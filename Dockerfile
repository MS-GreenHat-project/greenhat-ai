# --- Dockerfile ---
# 이 파일은 앱을 컨테이너 이미지로 만듭니다.

# Python 3.9 버전을 기반으로 합니다.
FROM python:3.9-slim

# 컨테이너 내의 작업 디렉토리 설정
WORKDIR /app

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

# 컨테이너 시작 시 Gunicorn으로 서버 실행
# eventlet 워커는 실시간 통신(Socket.IO)에 필수적이며, 워커 수는 1로 고정해야 합니다.
# --timeout 120: 이미지 분석 시간에 대비해 타임아웃을 넉넉하게 설정합니다.
CMD ["gunicorn", "--worker-class", "eventlet", "-w", "1", "--timeout", "120", "--bind", "0.0.0.0:5000", "app:app"]