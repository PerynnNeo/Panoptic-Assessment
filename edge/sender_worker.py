import threading, queue, time, json, requests, cv2

def to_jpeg_bytes(frame, quality=80):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok: raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

class SenderWorker:
    def __init__(self, cloud_url: str, maxsize=5, timeout=5):
        self.cloud_url = cloud_url
        self.timeout = timeout
        self.q = queue.Queue(maxsize=maxsize)
        self._stop = False
        self.sent = 0
        self.dropped = 0
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def submit(self, frame_bgr, detections, ts_capture):
        # Backpressure: drop newest when full
        try:
            self.q.put_nowait((frame_bgr, detections, ts_capture))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self):
        s = requests.Session()
        while not self._stop:
            try:
                frame_bgr, dets, ts_cap = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                img = to_jpeg_bytes(frame_bgr, quality=80)
                files = {"image": ("frame.jpg", img, "image/jpeg")}
                data = {
                    "ts_capture": str(ts_cap),
                    "ts_edge_send": str(time.time()),
                    "stream_id": "default",
                    "detections": json.dumps([
                        {"x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2),"score":float(s)}
                        for (x1,y1,x2,y2,s) in dets
                    ]),
                }
                r = s.post(self.cloud_url, data=data, files=files, timeout=self.timeout)
                r.raise_for_status()
                self.sent += 1
            except Exception:
                # swallow and continue; metrics printed by edge app
                pass
            finally:
                self.q.task_done()

    def stop(self):
        self._stop = True
        try:
            self.th.join(timeout=2)
        except Exception:
            pass
