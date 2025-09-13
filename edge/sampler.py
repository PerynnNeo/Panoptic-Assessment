import cv2, numpy as np

class Sampler:
    def __init__(self, mode="motion", motion_thr=12.0, heartbeat_s=2.0):
        self.mode = mode
        self.motion_thr = motion_thr
        self.heartbeat_s = heartbeat_s
        self._last_gray = None
        self._last_forward_ts = 0.0

    def _motion_score(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._last_gray is None:
            self._last_gray = gray
            return 0.0
        diff = cv2.absdiff(gray, self._last_gray)
        score = float(np.mean(diff))
        self._last_gray = gray
        return score

    def should_forward(self, frame, detections, now_ts):
        if detections and max([d[-1] for d in detections]) > 0.5:
            self._last_forward_ts = now_ts
            return True

        forward = False
        if self.mode in ("motion", "both"):
            if self._motion_score(frame) > self.motion_thr:
                forward = True
        if not forward and self.mode in ("heartbeat", "both"):
            if (now_ts - self._last_forward_ts) >= self.heartbeat_s:
                forward = True

        if forward:
            self._last_forward_ts = now_ts
        return forward
