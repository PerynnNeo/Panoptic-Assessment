import io, os, time, json, csv
from typing import Dict
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np, cv2, psutil
from tracker import Tracker
from activity import classify_activity
from alerts import contextual_alerts

app = FastAPI(title="Cloud Analysis")
RESULTS_DIR="/results"; os.makedirs(RESULTS_DIR, exist_ok=True)
TRACKERS: Dict[str, Tracker] = {}
LOITER_SECONDS = int(os.getenv("LOITER_SECONDS","10"))

def write_csv(path, row, header=None):
    exists=os.path.exists(path)
    with open(path,"a",newline="") as f:
        w=csv.writer(f)
        if not exists and header: w.writerow(header)
        w.writerow(row)

@app.post("/ingest")
async def ingest(
    image: UploadFile,
    ts_capture: str = Form(...),
    ts_edge_send: str = Form(...),
    stream_id: str = Form("default"),
    detections: str = Form("[]")
):
    cloud_t0=time.time()

    img_bytes = await image.read()
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if frame is None: return JSONResponse({"error":"decode_failed"}, status_code=400)

    ts_cap = float(ts_capture)

    trk = TRACKERS.get(stream_id) or Tracker()
    TRACKERS[stream_id] = trk

    dets = json.loads(detections)
    boxes = [(d["x1"], d["y1"], d["x2"], d["y2"]) for d in dets]
    trks = trk.update(boxes, ts_cap)

    acts=[]
    for tr in trks:
        lbl = classify_activity(tr, loiter_seconds=LOITER_SECONDS)
        acts.append((tr.id, lbl))

    alerts = contextual_alerts(acts, loiter_after=22)

    cloud_latency_ms = (time.time()-cloud_t0)*1000.0
    e2e_est_ms = (time.time()-ts_cap)*1000.0

    write_csv(f"{RESULTS_DIR}/cloud_events.csv",
              [time.time(),"default",json.dumps(acts),json.dumps(alerts)],
              header=["ts","stream_id","activities","alerts"])
    write_csv(f"{RESULTS_DIR}/cloud_metrics.csv",
              [time.time(),cloud_latency_ms,psutil.cpu_percent(),psutil.virtual_memory().percent],
              header=["ts","cloud_latency_ms","cpu_pct","mem_pct"])

    return {"ok":True,"activities":[{"track_id":tid,"label":lbl} for tid,lbl in acts],
            "alerts":alerts,"cloud_latency_ms":cloud_latency_ms,"e2e_est_ms":e2e_est_ms}

@app.get("/health")
def health(): return {"status":"ok"}
