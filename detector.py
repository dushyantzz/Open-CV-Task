import cv2
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='yolo11n.pt'):
        self.model = YOLO(model_path)
    
    def detect_and_track(self, frame):
        # We track 'person' (class 0) and potentially sacks if a custom model is used.
        # For now, we use standard YOLO to track people and their movements.
        results = self.model.track(frame, persist=True, verbose=False, classes=[0])
        return results
