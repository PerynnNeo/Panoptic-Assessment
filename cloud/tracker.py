from collections import deque
import math

def iou(b1, b2):
    x1=max(b1[0],b2[0]); y1=max(b1[1],b2[1]); x2=min(b1[2],b2[2]); y2=min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    a1=max(0,b1[2]-b1[0])*max(0,b1[3]-b1[1])
    a2=max(0,b2[2]-b2[0])*max(0,b2[3]-b2[1])
    return inter/max(1e-6, a1+a2-inter)

class Track:
    def __init__(self, tid, box, ts):
        self.id=tid; self.box=box; self.ts=ts; self.history=deque(maxlen=32)
        self.push(ts, box)
    def push(self, ts, box):
        x1,y1,x2,y2=box; cx=(x1+x2)/2; cy=(y1+y2)/2; w=x2-x1; h=y2-y1
        self.history.append((ts,cx,cy,w,h)); self.box=box; self.ts=ts
    def speed_px_s(self):
        if len(self.history)<2: return 0.0
        (t0,cx0,cy0,_,_), (t1,cx1,cy1,_,_) = self.history[-2], self.history[-1]
        return (( (cx1-cx0)**2 + (cy1-cy0)**2 )**0.5)/max(1e-6,t1-t0)
    def aspect_ratio(self):
        _,_,_,w,h=self.history[-1]; return h/max(1.0,w)

class Tracker:
    def __init__(self, iou_thr=0.3):
        self.iou_thr=iou_thr; self.tracks=[]; self.next_id=1
    def update(self, boxes, ts):
        assigned=set()
        for tr in self.tracks:
            best=None; best_iou=0.0; best_j=-1
            for j,b in enumerate(boxes):
                if j in assigned: continue
                i=iou(tr.box,b)
                if i>best_iou: best, best_iou, best_j=b, i, j
            if best is not None and best_iou>=self.iou_thr:
                tr.push(ts,best); assigned.add(best_j)
        for j,b in enumerate(boxes):
            if j not in assigned:
                tr=Track(self.next_id,b,ts); self.next_id+=1; self.tracks.append(tr)
        return self.tracks
