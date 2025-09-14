# cloud/activity.py
import time, math
from typing import Optional, Tuple

# Track memory
_STATE = {}  # tid -> dict(last_xy, last_t, fast_ema, slow_ema, label, last_change_t, below_since)

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def classify_activity(
    track,
    now: Optional[float] = None,
    frame_size: Optional[Tuple[int, int]] = None,  # (h, w)
    v_walk_norm: float = 0.040,   # threshold as fraction of frame height per second
    alpha_fast: float = 0.60,     # fast EMA (responsive)
    alpha_slow: float = 0.18,     # slow EMA (stable)
    promote_hold_s: float = 0.35, # time required to switch TO walking
    demote_hold_s: float  = 1.20, # time required to switch TO stationary (slower!)
    hysteresis: float = 0.12,     # deadband around threshold
    demote_below_s: float = 0.80  # how long both EMAs must stay below thr to demote
):
    """
    Two-class activity: 'walking' or 'stationary'.
    Uses dual-rate EMA, asymmetric holds, and requires sustained below-threshold
    evidence before demoting to stationary.
    """
    if now is None:
        now = time.time()

    tid = track.id
    x1, y1, x2, y2 = map(float, track.box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bh = max(1.0, (y2 - y1))  # bbox height fallback normalizer

    st = _STATE.get(tid)
    if st is None:
        st = {
            "last_xy": (cx, cy),
            "last_t": now,
            "fast_ema": 0.0,
            "slow_ema": 0.0,
            "label": "stationary",
            "last_change_t": now,
            "below_since": None  # when both EMAs first fell below demote threshold
        }
        _STATE[tid] = st
        return "stationary"

    dt = max(1e-3, now - st["last_t"])
    inst_speed_px = _dist((cx, cy), st["last_xy"]) / dt

    # Normalize speed by frame height (preferred) or bbox height
    if frame_size is not None:
        fh = max(1.0, float(frame_size[0]))
    else:
        fh = bh
    inst_speed_norm = inst_speed_px / fh

    # Dual EMA update
    fast = alpha_fast * inst_speed_norm + (1.0 - alpha_fast) * st["fast_ema"]
    slow = alpha_slow * inst_speed_norm + (1.0 - alpha_slow) * st["slow_ema"]

    st["last_xy"] = (cx, cy)
    st["last_t"] = now
    st["fast_ema"] = fast
    st["slow_ema"] = slow

    # Hysteresis thresholds
    walk_up = v_walk_norm * (1.0 + hysteresis)
    walk_dn = v_walk_norm * (1.0 - hysteresis)

    cur = st["label"]
    proposed = cur

    # Promotion logic — be quick when motion ramps up
    if cur == "stationary" and fast >= walk_up and fast > slow:
        proposed = "walking"
        if (now - st["last_change_t"]) >= promote_hold_s:
            st["label"] = "walking"
            st["last_change_t"] = now
            st["below_since"] = None
            return st["label"]

    # Demotion logic — be SLOW and require sustained evidence
    # 1) Both EMAs must be below walk_dn
    # 2) Start a timer (below_since) when both go below; only demote if duration >= demote_below_s
    elif cur == "walking":
        both_below = (fast <= walk_dn and slow <= walk_dn)
        if both_below:
            if st["below_since"] is None:
                st["below_since"] = now
            below_dur = now - st["below_since"]
        else:
            st["below_since"] = None
            below_dur = 0.0

        if both_below and below_dur >= demote_below_s:
            proposed = "stationary"
            if (now - st["last_change_t"]) >= demote_hold_s:
                st["label"] = "stationary"
                st["last_change_t"] = now
                st["below_since"] = None
                return st["label"]

    # If neither branch commits a label change, keep the current one
    return st["label"]

def forget_track(track_id: int):
    _STATE.pop(track_id, None)
