# Panoptic-Assessment
Real-Time Edge-Cloud Person Activity Recognition System

This project implements a **real-time edge–cloud pipeline** for person detection and simple activity recognition.
It is structured into two main services:

* **Edge (`edge/`)**: captures RTSP video, runs lightweight HOG detection + smart sampling, and forwards selected frames.
* **Cloud (`cloud/`)**: ingests frames, tracks objects, classifies activity (*walking* / *stationary*), and generates an annotated output video.

All components are containerized with **Docker Compose**.

---

## Project Structure

```
.
├── edge/                # Edge processor code (detector, sampler, sender)
├── cloud/               # Cloud analyzer code (tracker, activity, annotator, API)
├── streaming/             
├── monitoring/             
├── samples/             # Input video(s) for RTSP streaming
├── results/             # Output metrics and annotated videos
├── docker-compose.yml   # Multi-service config
└── README.md            
```

---

## How to Run

### 1. Start services (edge, cloud, RTSP)

From the repo root:

```powershell
cd "C:your/path"
docker compose build --no-cache edge cloud
docker compose up rtsp cloud edge
```

This launches:

* **rtsp\_server**: streams the sample video at `rtsp://rtsp:8554/stream`
* **edge\_processor**: captures the RTSP feed, detects persons, forwards selected frames
* **cloud\_analyzer**: ingests frames, tracks objects, and logs activities

---

### 2. Publish the sample video

In another terminal, stream the input file into RTSP:

```powershell
docker run --rm --network panoptic-assessment-1_default `
  -v "${PWD}\samples:/samples:ro" `
  jrottenberg/ffmpeg:4.4 -re -stream_loop -1 -i /samples/input.mp4 `
  -c:v libx264 -preset ultrafast -tune zerolatency -c:a aac -ar 44100 -b:a 128k `
  -f rtsp rtsp://rtsp:8554/stream
```

---

### 3. Verify system is running

* **Edge logs** will show:

  ```
  [EDGE] fps=18.5 cpu=3.0% mem=3.2% fwd=120
  ```
* **Cloud logs** will show repeated lines:

  ```
  INFO:     172.20.0.5:xxxxx - "POST /ingest HTTP/1.1" 200 OK
  ```

---

### 4. Outputs

Results are written under `results/`:

* `edge_summary.json` – edge runtime stats (fps, frames forwarded, etc.)
* `edge_metrics.csv` – per-stage timing from edge
* `cloud_metrics.csv` – latency and resource metrics from cloud
* `annotated_activity.avi` – final annotated output video with bounding boxes and labels

Open the AVI file with VLC or any player that supports Xvid/AVI.

---

## Notes

* Edge constrained to **4 CPU cores** and **8GB RAM** (set in `docker-compose.yml`).
* Activities simplified to **walking** and **stationary** for clarity.
* Ghost suppression and stability gating reduce false boxes and clutter.