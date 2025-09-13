import time, csv, psutil
from collections import deque

class EdgeMetrics:
    def __init__(self, csv_path):
        self.csv = open(csv_path, "w", newline="")
        self.w = csv.writer(self.csv)
        self.w.writerow(["ts","stage","frame_id","dt_ms","fps","cpu_pct","mem_pct","forwarded"])
        self._first_ts = time.time()
        self._frame_count = 0
        self._forwarded = 0
        self._fps_window = deque(maxlen=60)

    def mark(self, stage, frame_id, start_ts):
        now = time.time()
        dt_ms = (now - start_ts) * 1000.0
        self.w.writerow([f"{now:.3f}", stage, frame_id, f"{dt_ms:.1f}",
                         "", f"{psutil.cpu_percent(interval=None):.1f}",
                         f"{psutil.virtual_memory().percent:.1f}",
                         self._forwarded])

    def tick_fps(self):
        now = time.time()
        self._frame_count += 1
        self._fps_window.append(now)

    def increment_forwarded(self):
        self._forwarded += 1

    def maybe_periodic_print(self, every_s=5):
        now = time.time()
        if not hasattr(self, "_last_print"):
            self._last_print = now
            return
        if (now - self._last_print) >= every_s:
            fps = self.current_fps()
            print(f"[EDGE] fps={fps:.2f} cpu={psutil.cpu_percent():.1f}% mem={psutil.virtual_memory().percent:.1f}% fwd={self._forwarded}")
            self._last_print = now

    def current_fps(self):
        if len(self._fps_window) < 2: return 0.0
        return (len(self._fps_window)-1) / (self._fps_window[-1] - self._fps_window[0])

    def finalize(self):
        self.csv.flush(); self.csv.close()
        runtime_s = time.time() - self._first_ts
        return {"runtime_s": round(runtime_s,2),
                "frames": self._frame_count,
                "forwarded": self._forwarded,
                "avg_fps": round(self.current_fps(),2)}
