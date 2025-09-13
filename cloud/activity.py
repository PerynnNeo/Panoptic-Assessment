def classify_activity(track, loiter_seconds=10):
    spd = track.speed_px_s()
    label = "stationary"
    if spd > 220: label = "running"
    elif spd > 60: label = "walking"

    if len(track.history) >= 3:
        ts0, cx0, cy0, w0, h0 = track.history[-3]
        ts1, cx1, cy1, w1, h1 = track.history[-1]
        dy = cy1 - cy0
        ar_prev = h0 / max(1.0, w0)
        ar_now  = h1 / max(1.0, w1)
        if (ar_prev > 1.2) and (ar_now < 0.8) and (dy > 40):
            label = "fall"

    if label == "stationary":
        t_start = track.history[0][0]; t_now = track.history[-1][0]
        if (t_now - t_start) >= loiter_seconds:
            label = "loitering"

    return label
