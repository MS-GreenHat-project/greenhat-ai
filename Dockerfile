# Base image
FROM python:3.10-slim

# OS 패키지 설치
RUN apt-get update && apt-get install -y \
    git curl gcc ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 종속성 설치
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# 모델, 코드 복사
COPY app.py ./
COPY hemletYoloV8_100epochs.pt ./

# 결과 저장 경로 생성
RUN mkdir -p /app/runs

# 컨테이너 시작 시 app 실행
CMD ["python", "app.py"]
