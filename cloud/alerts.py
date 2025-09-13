from collections import Counter
from datetime import datetime
def contextual_alerts(activities, loiter_after=22):
    alerts=[]
    hour = datetime.now().hour
    if hour >= loiter_after or hour < 5:
        if any(lbl=="loitering" for _,lbl in activities): alerts.append("After-hours loitering")
    if any(lbl=="fall" for _,lbl in activities): alerts.append("Fall detected")
    if Counter(lbl for _,lbl in activities).get("running",0) >= 3: alerts.append("Crowd running")
    return alerts
