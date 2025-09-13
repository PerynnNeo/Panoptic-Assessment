import os, time, json
import cv2

from metrics import EdgeMetrics
from video_source import open_source          # now returns a resilient capture wrapper
from detector import HogPersonDetector
from sampler import Sampler
from sender_worker import SenderWorker        # async, bounded queue HTTP sender

RESULTS_DIR = "/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def env(name, default=None, cast=str):
    v = os.getenv(name, default)
    try:
        return cast(v)
    except Exception:
        return default

# Configure via env (compose sets VIDEO_SOURCE to rtsp://rtsp:8554/stream for Task 3)
VIDEO_SOURCE  = env("VIDEO_SOURCE", "samples/input.mp4")
SAMPLER_MODE  = env("SAMPLER_MODE", "motion")
MOTION_THR    = env("MOTION_THR", 12.0, float)
HEARTBEAT_S   = env("HEARTBEAT_S", 2.0, float)
CLOUD_URL     = env("CLOUD_URL", "http://cloud:8000/ingest")

def main():
    print("[EDGE] starting with config:",
          json.dumps({
              "VIDEO_SOURCE": VIDEO_SOURCE,
              "SAMPLER_MODE": SAMPLER_MODE,
              "MOTION_THR": MOTION_THR,
              "HEARTBEAT_S": HEARTBEAT_S,
              "CLOUD_URL": CLOUD_URL
          }, indent=2))

    # resilient capture (auto-reconnects on RTSP hiccups)
    cap = open_source(VIDEO_SOURCE)
    if cap is None:
        raise RuntimeError(f"Cannot open source: {VIDEO_SOURCE}")

    detector = HogPersonDetector()
    sampler  = Sampler(mode=SAMPLER_MODE, motion_thr=MOTION_THR, heartbeat_s=HEARTBEAT_S)
    metrics  = EdgeMetrics(csv_path=f"{RESULTS_DIR}/edge_metrics.csv")

    # async sender with small bounded queue (prevents backpressure from stalling edge)
    sender = SenderWorker(CLOUD_URL, maxsize=5, timeout=5)

    frame_id = 0
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                # For RTSP: ResilientCapture auto-retries internally; loop continues.
                # For file: this typically means EOF; break to finalize.
                # If you're on RTSP and seeing many not-ok reads, let it loop.
                break
            metrics.mark("capture", frame_id, t0)

            t1 = time.time()
            persons = detector.predict(frame)          # [[x1,y1,x2,y2,score], ...]
            metrics.mark("detect", frame_id, t1)

            t2 = time.time()
            forward = sampler.should_forward(frame, persons, t2)
            metrics.mark("sample_decision", frame_id, t2)

            if forward:
                queued = sender.submit(frame, persons, ts_capture=t0)
                if not queued:
                    print("[EDGE->CLOUD] queue full; dropping frame")
                metrics.increment_forwarded()

            metrics.tick_fps()
            metrics.maybe_periodic_print()
            frame_id += 1

    finally:
        # stop sender thread cleanly
        try:
            sender.stop()
        except Exception:
            pass

        # release capture
        try:
            cap.release()
        except Exception:
            pass

        # write summary (include sender stats)
        summary = metrics.finalize()
        summary.update({
            "sender_sent": getattr(sender, "sent", 0),
            "sender_dropped": getattr(sender, "dropped", 0)
        })
        with open(f"{RESULTS_DIR}/edge_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("[EDGE] summary:", summary)

if __name__ == "__main__":
    main()
