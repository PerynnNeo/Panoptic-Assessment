# cloud/annotator.py  (AVI writer)
import os, cv2
from typing import List, Tuple

COLORS = {
    "walking":    (60, 220, 60),   # green
    "stationary": (200, 180, 0),   # yellow
    "unknown":    (200, 200, 200)
}

class ActivityAnnotator:
    def __init__(self, out_path="/results/annotated_activity.avi", fps=15,
                 show_legend=True):
        self.out_path = out_path
        self.fps = fps
        self.writer = None
        self.show_legend = show_legend
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _ensure_writer(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # robust AVI
            self.writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (w, h))

    def _draw_legend(self, img):
        if not self.show_legend: return
        items = [("walking", COLORS["walking"]),
                 ("stationary", COLORS["stationary"])]
        x0, y0 = 10, img.shape[0] - 10
        line_h = 18
        for i, (name, color) in enumerate(items[::-1]):
            y = y0 - i * line_h
            cv2.rectangle(img, (x0, y - 12), (x0 + 12, y), color, -1)
            cv2.putText(img, name, (x0 + 18, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240,240,240), 1, cv2.LINE_AA)

    def draw(self,
             frame_bgr,
             items: List[Tuple[Tuple[int,int,int,int], int, str]]):
        """
        items: list of ((x1,y1,x2,y2), track_id, label) where label in {walking, stationary}
        """
        self._ensure_writer(frame_bgr)
        out = frame_bgr.copy()

        for (x1,y1,x2,y2), tid, label in items:
            lbl = label if label in COLORS else "unknown"
            color = COLORS[lbl]
            x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
            cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
            tag = f"{lbl}  #{tid}"
            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y_cap = max(0, y1 - 8)
            cv2.rectangle(out, (x1, y_cap - th - 6), (x1 + tw + 6, y_cap + 2), color, -1)
            cv2.putText(out, tag, (x1 + 3, y_cap),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

        self._draw_legend(out)
        self.writer.write(out)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
