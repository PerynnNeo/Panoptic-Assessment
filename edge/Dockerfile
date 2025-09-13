FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

ENV VIDEO_SOURCE=samples/input.mp4
ENV SAMPLER_MODE=motion
ENV MOTION_THR=12.0
ENV HEARTBEAT_S=2.0

CMD ["python","/app/app.py"]
