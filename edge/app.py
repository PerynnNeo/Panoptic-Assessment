import os, time, json
import cv2

from metrics import EdgeMetrics
from video_source import open_source          # resilient capture wrapper
from detector import HogPersonDetector
from sampler import Sampler
from sender_worker import SenderWorker        # async, bounded queue HTTP sender
from annotator import Annotator

RESULTS_DIR = "/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def env(name, default=None, cast=str):
    v = os.getenv(name, default)
    try:
        return cast(v)
    except Exception:
        return default

# Env config
VIDEO_SOURCE  = env("VIDEO_SOURCE", "samples/input.mp4")
SAMPLER_MODE  = env("SAMPLER_MODE", "motion")
MOTION_THR    = env("MOTION_THR", 12.0, float)
HEARTBEAT_S   = env("HEARTBEAT_S", 2.0, float)
CLOUD_URL     = env("CLOUD_URL", "http://cloud:8000/ingest")
ANNOTATE      = env("ANNOTATE", "0") in ("1", "true", "True")
ANNOTATE_FPS  = env("ANNOTATE_FPS", 15, int)  # match your RTSP fps

def main():
    print("[EDGE] starting with config:",
          json.dumps({
              "VIDEO_SOURCE": VIDEO_SOURCE,
              "SAMPLER_MODE": SAMPLER_MODE,
              "MOTION_THR": MOTION_THR,
              "HEARTBEAT_S": HEARTBEAT_S,
              "CLOUD_URL": CLOUD_URL
          }, indent=2), flush=True)

    # resilient capture (auto-reconnects on RTSP hiccups)
    cap = open_source(VIDEO_SOURCE)
    if cap is None:
        raise RuntimeError(f"Cannot open source: {VIDEO_SOURCE}")

    detector = HogPersonDetector()
    sampler  = Sampler(mode=SAMPLER_MODE, motion_thr=MOTION_THR, heartbeat_s=HEARTBEAT_S)
    metrics  = EdgeMetrics(csv_path=f"{RESULTS_DIR}/edge_metrics.csv")

    annot = Annotator("/results/annotated.mp4", fps=ANNOTATE_FPS) if ANNOTATE else None

    # async sender with bounded queue; optional if CLOUD_URL unset
    sender = None if not CLOUD_URL or CLOUD_URL.strip() == "" else SenderWorker(CLOUD_URL, maxsize=5, timeout=5)

    frame_id = 0
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                # Wait instead of exiting when the RTSP stream isn't publishing yet.
                if VIDEO_SOURCE.lower().startswith("rtsp://"):
                    time.sleep(0.25)
                    continue
                else:
                    break
            metrics.mark("capture", frame_id, t0)

            t1 = time.time()
            persons = detector.predict(frame)          # [[x1,y1,x2,y2,score], ...]
            if ANNOTATE and annot is not None:
                annot.draw_and_write(frame, persons)
            metrics.mark("detect", frame_id, t1)

            t2 = time.time()
            forward = sampler.should_forward(frame, persons, t2)
            metrics.mark("sample_decision", frame_id, t2)

            if forward:
                if sender is not None:
                    queued = sender.submit(frame, persons, ts_capture=t0)
                    if not queued:
                        print("[EDGE->CLOUD] queue full; dropping frame", flush=True)
                metrics.increment_forwarded()

            metrics.tick_fps()
            metrics.maybe_periodic_print()
            frame_id += 1

    finally:
        # stop sender thread cleanly
        try:
            if sender is not None:
                sender.stop()
        except Exception:
            pass

        # release capture
        try:
            cap.release()
        except Exception:
            pass

        if annot is not None:
            try:
                annot.release()
            except Exception:
                pass

        # write summary (include sender stats if present)
        summary = metrics.finalize()
        summary.update({
            "sender_sent": getattr(sender, "sent", 0) if sender is not None else 0,
            "sender_dropped": getattr(sender, "dropped", 0) if sender is not None else 0
        })
        with open(f"{RESULTS_DIR}/edge_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("[EDGE] summary:", summary, flush=True)

if __name__ == "__main__":
    main()
