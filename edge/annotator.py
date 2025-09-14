# edge/annotator.py
import cv2, os

class Annotator:
    def __init__(self, out_path="/results/annotated.mp4", fps=15):
        self.out_path = out_path
        self.fps = fps
        self.writer = None
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _ensure_writer(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (w, h))

    def draw_and_write(self, frame_bgr, detections):
        # detections: [(x1,y1,x2,y2,score), ...]
        self._ensure_writer(frame_bgr)
        out = frame_bgr.copy()
        for (x1,y1,x2,y2,score) in detections:
            x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
            cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(out, f"person {score:.2f}", (x1,max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        self.writer.write(out)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
