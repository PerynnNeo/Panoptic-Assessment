import time, json, requests, cv2

def to_jpeg_bytes(frame, quality=80):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok: raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

def post_frame(cloud_url, frame_bgr, detections, ts_capture, ts_edge_send=None, timeout=5):
    if ts_edge_send is None:
        ts_edge_send = time.time()
    img_bytes = to_jpeg_bytes(frame_bgr)
    dets = [{"x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2),"score":float(s)} for (x1,y1,x2,y2,s) in detections]
    files = {"image": ("frame.jpg", img_bytes, "image/jpeg")}
    data = {
        "ts_capture": str(ts_capture),
        "ts_edge_send": str(ts_edge_send),
        "stream_id": "default",
        "detections": json.dumps(dets),
    }
    r = requests.post(cloud_url, data=data, files=files, timeout=timeout)
    r.raise_for_status()
    return r.json()
