# cloud/server.py
import os, time, json, csv
from typing import Dict
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np, cv2, psutil

from tracker import Tracker
from activity import classify_activity, forget_track  # walking / stationary only
from annotator import ActivityAnnotator               # writes AVI

# =========================
# Ghost-track control utils
# =========================
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)

# Last time each track had a real detection hit (not just prediction)
LAST_HIT: Dict[str, Dict[int, float]] = {}   # stream_id -> {track_id: last_detection_time}
# Consecutive hit counter to draw only stable tracks
HIT_COUNT: Dict[str, Dict[int, int]] = {}    # stream_id -> {track_id: consecutive_hits}

# =========
# FastAPI
# =========
app = FastAPI(title="Cloud Analysis")
RESULTS_DIR = "/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TRACKERS: Dict[str, Tracker] = {}  # stream_id -> Tracker

# -----------------
# Tunables (via env)
# -----------------
IOU_DRAW_THR          = float(os.getenv("IOU_DRAW_THR",          "0.20"))  # require IoU>=thr with current det
STALE_DRAW_S          = float(os.getenv("STALE_DRAW_S",          "0.8"))   # hide tracks not hit within this window
STALE_FORGET_S        = float(os.getenv("STALE_FORGET_S",        "2.0"))   # forget track state after this long
MIN_HITS_BEFORE_DRAW  = int(os.getenv("MIN_HITS_BEFORE_DRAW",    "3"))     # need N consecutive hits to draw
MAX_DRAW_PER_FRAME    = int(os.getenv("MAX_DRAW_PER_FRAME",      "6"))     # cap boxes per frame in overlay

ACT_ANNOTATE          = os.getenv("ACT_ANNOTATE", "0") in ("1", "true", "True")
ACT_ANNOTATE_FPS      = int(os.getenv("ACT_ANNOTATE_FPS", "15"))
OUT_PATH              = os.getenv("ACT_ANNOTATE_OUT", "/results/annotated_activity.avi")
ANNOT = ActivityAnnotator(OUT_PATH, fps=ACT_ANNOTATE_FPS) if ACT_ANNOTATE else None

def write_csv(path, row, header=None):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists and header:
            w.writerow(header)
        w.writerow(row)

# -------
# Ingest
# -------
@app.post("/ingest")
async def ingest(
    image: UploadFile,
    ts_capture: str = Form(...),
    ts_edge_send: str = Form(...),
    stream_id: str = Form("default"),
    detections: str = Form("[]")
):
    cloud_t0 = time.time()

    # Decode image
    img_bytes = await image.read()
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"error": "decode_failed"}, status_code=400)
    h, w = frame.shape[:2]

    ts_cap = float(ts_capture)

    # Per-stream tracker & state
    trk = TRACKERS.get(stream_id) or Tracker()
    TRACKERS[stream_id] = trk
    LAST_HIT.setdefault(stream_id, {})
    HIT_COUNT.setdefault(stream_id, {})

    # Parse detections and update tracker
    dets = json.loads(detections)
    det_boxes = [(d["x1"], d["y1"], d["x2"], d["y2"]) for d in dets]
    trks = trk.update(det_boxes, ts_cap)

    now = time.time()
    hit_map = LAST_HIT[stream_id]
    cnt_map = HIT_COUNT[stream_id]
    acts = []
    items = []

    # For each track, require an IoU hit with any current detection.
    # Only classify/draw "fresh" AND "stable" tracks.
    for tr in trks:
        box = tuple(map(int, tr.box))
        matched = any(_iou(box, db) >= IOU_DRAW_THR for db in det_boxes)
        if matched:
            hit_map[tr.id] = now
            cnt_map[tr.id] = cnt_map.get(tr.id, 0) + 1
        else:
            if tr.id in cnt_map:
                cnt_map[tr.id] = 0  # break the streak if not matched this frame

        last_hit = hit_map.get(tr.id, 0.0)
        if (now - last_hit) <= STALE_DRAW_S and cnt_map.get(tr.id, 0) >= MIN_HITS_BEFORE_DRAW:
            lbl = classify_activity(tr, now=now, frame_size=(h, w))  # walking / stationary
            acts.append((tr.id, lbl))
            items.append((box, tr.id, lbl))
        # else: skip stale or unstable tracks this frame

    # Forget very stale tracks entirely
    forget_cut = now - STALE_FORGET_S
    for tid, tlast in list(hit_map.items()):
        if tlast < forget_cut:
            hit_map.pop(tid, None)
            cnt_map.pop(tid, None)
            try:
                forget_track(tid)
            except Exception:
                pass

    # Metrics
    cloud_latency_ms = (time.time() - cloud_t0) * 1000.0
    e2e_est_ms = (now - ts_cap) * 1000.0
    write_csv(
        f"{RESULTS_DIR}/cloud_metrics.csv",
        [time.time(), cloud_latency_ms, psutil.cpu_percent(), psutil.virtual_memory().percent],
        header=["ts", "cloud_latency_ms", "cpu_pct", "mem_pct"],
    )

    # Annotation: cap by area to reduce clutter
    if ANNOT is not None and items:
        items.sort(key=lambda it: (it[0][2] - it[0][0]) * (it[0][3] - it[0][1]), reverse=True)
        items = items[:MAX_DRAW_PER_FRAME]
        try:
            ANNOT.draw(frame, items)
        except Exception:
            pass

    return {
        "ok": True,
        "activities": [{"track_id": tid, "label": lbl} for tid, lbl in acts],
        "cloud_latency_ms": cloud_latency_ms,
        "e2e_est_ms": e2e_est_ms,
    }

# -------
# Health
# -------
@app.get("/health")
def health():
    return {"status": "ok"}

# --------------
# Clean shutdown
# --------------
import atexit
@atexit.register
def _close_annot():
    try:
        if ANNOT is not None:
            ANNOT.release()
    except Exception:
        pass
