import time, cv2

class ResilientCapture:
    def __init__(self, spec: str, retry_delay=1.0):
        self.spec = spec
        self.retry_delay = retry_delay
        self.cap = None
        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(self.spec)

    def read(self):
        ok, frame = self.cap.read()
        if ok and frame is not None:
            return True, frame

        # Reconnect on failure
        try:
            self.cap.release()
        except Exception:
            pass
        time.sleep(self.retry_delay)
        self._open()
        ok, frame = self.cap.read()
        return ok, frame

    def release(self):
        try:
            if self.cap: self.cap.release()
        except Exception:
            pass

def open_source(spec: str):
    return ResilientCapture(spec)
