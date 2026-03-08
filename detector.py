import cv2
from ultralytics import YOLO


class Detector:
    def __init__(self, model_path='yolo11n.pt', conf=0.30, iou=0.50, img_size=640):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.img_size = img_size

    def detect_and_track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            classes=[0],
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            tracker="bytetrack.yaml",
        )
        return results
