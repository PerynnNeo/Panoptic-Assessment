# edge/detector.py
import os, cv2
import numpy as np

def env(name, default=None, cast=float):
    v = os.getenv(name, default)
    try:
        return cast(v) if cast else v
    except Exception:
        return default

# Tunables (override via edge service env)
DET_MIN_CONF   = env("DET_MIN_CONF",   0.35)   # after sigmoid
DET_MIN_H_FRAC = env("DET_MIN_H_FRAC", 0.12)
DET_MIN_W_FRAC = env("DET_MIN_W_FRAC", 0.06)
AR_MIN         = env("DET_AR_MIN",     1.25)   # h/w
AR_MAX         = env("DET_AR_MAX",     4.00)
BORDER_FRAC    = env("DET_BORDER_FRAC",0.02)
DET_NMS_IOU    = env("DET_NMS_IOU",    0.40)
MAX_DETS       = env("DET_MAX_DETS",   11, int)   # hard cap per frame

def _nms_xyxy(boxes, scores, iou_thr=0.4):
    if len(boxes) == 0:
        return []
    b = np.array(boxes, dtype=np.float32)
    s = np.array(scores, dtype=np.float32)
    x1, y1, x2, y2 = b[:,0], b[:,1], b[:,2], b[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / np.maximum(1e-6, areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep

class HogPersonDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def predict(self, frame_bgr):
        H, W = frame_bgr.shape[:2]

        # HOG detectMultiScale: use positional args (no kwargs!)
        # Signature: img, hitThreshold, winStride, padding, scale, groupThreshold
        rects, weights = self.hog.detectMultiScale(
            frame_bgr,
            0.0,               # hitThreshold
            (8, 8),            # winStride
            (8, 8),            # padding
            1.05,              # scale
            2                  # groupThreshold (stronger grouping)
        )

        if len(rects) == 0:
            return []

        # Map raw SVM scores to ~[0,1] with a sigmoid
        w_arr = np.array(weights, dtype=np.float32).reshape(-1)
        w_norm = 1.0 / (1.0 + np.exp(-w_arr)) if w_arr.size else np.zeros(len(rects), np.float32)

        boxes_xyxy, scores = [], []
        min_h = max(1, int(DET_MIN_H_FRAC * H))
        min_w = max(1, int(DET_MIN_W_FRAC * W))
        border_px = int(BORDER_FRAC * max(H, W))

        for (x, y, w, h), sc in zip(rects, w_norm):
            if sc < DET_MIN_CONF: continue
            if w < min_w or h < min_h: continue
            ar = h / max(1.0, float(w))
            if ar < AR_MIN or ar > AR_MAX: continue
            # border gate
            if x <= border_px or y <= border_px or (x + w) >= (W - border_px) or (y + h) >= (H - border_px):
                continue
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            boxes_xyxy.append([x1, y1, x2, y2])
            scores.append(float(sc))

        if not boxes_xyxy:
            return []

        keep = _nms_xyxy(boxes_xyxy, scores, iou_thr=DET_NMS_IOU)
        keep = _nms_xyxy(boxes_xyxy, scores, iou_thr=DET_NMS_IOU)

        # sort kept indices by score desc and cap to MAX_DETS
        keep_sorted = sorted(keep, key=lambda i: scores[i], reverse=True)[:MAX_DETS]
        
        out = []
        for i in keep_sorted:
            x1, y1, x2, y2 = boxes_xyxy[i]
            out.append([x1, y1, x2, y2, scores[i]])
        return out