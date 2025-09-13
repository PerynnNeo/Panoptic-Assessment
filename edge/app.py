import os, time, json
from metrics import EdgeMetrics
from video_source import open_source
from detector import HogPersonDetector
from sampler import Sampler

RESULTS_DIR = "/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def env(name, default=None, cast=str):
    v = os.getenv(name, default)
    try:
        return cast(v)
    except Exception:
        return default

VIDEO_SOURCE = env("VIDEO_SOURCE", "samples/input.mp4")
SAMPLER_MODE = env("SAMPLER_MODE", "motion")
MOTION_THR   = env("MOTION_THR", 12.0, float)
HEARTBEAT_S  = env("HEARTBEAT_S", 2.0, float)

def main():
    print("[EDGE] starting with config:",
          json.dumps({"VIDEO_SOURCE": VIDEO_SOURCE, "SAMPLER_MODE": SAMPLER_MODE,
                      "MOTION_THR": MOTION_THR, "HEARTBEAT_S": HEARTBEAT_S}, indent=2))

    cap = open_source(VIDEO_SOURCE)
    if cap is None:
        raise RuntimeError(f"Cannot open source: {VIDEO_SOURCE}")

    detector = HogPersonDetector()
    sampler = Sampler(mode=SAMPLER_MODE, motion_thr=MOTION_THR, heartbeat_s=HEARTBEAT_S)
    metrics = EdgeMetrics(csv_path=f"{RESULTS_DIR}/edge_metrics.csv")

    frame_id = 0
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok: break
            metrics.mark("capture", frame_id, t0)

            t1 = time.time()
            persons = detector.predict(frame)
            metrics.mark("detect", frame_id, t1)

            t2 = time.time()
            forward = sampler.should_forward(frame, persons, t2)
            metrics.mark("sample_decision", frame_id, t2)
            if forward:
                metrics.increment_forwarded()

            metrics.tick_fps()
            metrics.maybe_periodic_print()

            frame_id += 1
    finally:
        cap.release()
        summary = metrics.finalize()
        with open(f"{RESULTS_DIR}/edge_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("[EDGE] summary:", summary)

if __name__ == "__main__":
    main()
