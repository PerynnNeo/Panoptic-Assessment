import cv2

class HogPersonDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def predict(self, frame):
        h, w = frame.shape[:2]
        scale = 640.0 / max(h, w) if max(h, w) > 640 else 1.0
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        rects, weights = self.hog.detectMultiScale(frame, winStride=(8,8), padding=(8,8), scale=1.05)
        results = []
        for (x, y, ww, hh), s in zip(rects, weights):
            x1, y1, x2, y2 = int(x/scale), int(y/scale), int((x+ww)/scale), int((y+hh)/scale)
            results.append([x1, y1, x2, y2, float(s)])
        return results
