import cv2

def open_source(spec: str):
    cap = cv2.VideoCapture(spec)
    if not cap.isOpened():
        return None
    return cap
