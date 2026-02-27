import cv2
import numpy as np

class SackCounter:
    def __init__(self, line_y_ratio=0.6):
        self.line_y_ratio = line_y_ratio
        self.sack_count = 0
        self.tracked_objects = {}  # id -> {'last_y': y, 'crossed': bool}

    def process_tracks(self, results, frame):
        h, w = frame.shape[:2]
        line_y = int(h * self.line_y_ratio)
        
        if not results or results[0].boxes.id is None:
            return self.sack_count

        ids = results[0].boxes.id.cpu().numpy().astype(int)
        bboxes = results[0].boxes.xyxy.cpu().numpy()

        for obj_id, bbox in zip(ids, bboxes):
            # Using center of mass for tracking
            y_point = (bbox[1] + bbox[3]) / 2 
            
            if obj_id not in self.tracked_objects:
                self.tracked_objects[obj_id] = {'last_y': y_point, 'crossed': False}
            else:
                last_y = self.tracked_objects[obj_id]['last_y']
                
                # Directional Crossing: Moving UP (y decreasing) across the line
                # Most loaders walk from bottom to top (truck)
                if not self.tracked_objects[obj_id]['crossed']:
                    if last_y > line_y and y_point <= line_y:
                        self.sack_count += 1
                        self.tracked_objects[obj_id]['crossed'] = True
                
                # Reset if they move back significantly to allow counting next sack
                if self.tracked_objects[obj_id]['crossed'] and y_point > line_y + 50:
                    self.tracked_objects[obj_id]['crossed'] = False

                self.tracked_objects[obj_id]['last_y'] = y_point

        return self.sack_count

    def draw_ui(self, frame, results):
        h, w = frame.shape[:2]
        line_y = int(h * self.line_y_ratio)
        
        # Draw counting line
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 3)
        
        # Stats Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        cv2.putText(frame, f"Sacks Counted: {self.sack_count}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Draw Bounding Boxes
        if results and results[0].boxes.id is not None:
             boxes = results[0].boxes.xyxy.cpu().numpy()
             ids = results[0].boxes.id.cpu().numpy().astype(int)
             for box, obj_id in zip(boxes, ids):
                 x1, y1, x2, y2 = map(int, box)
                 cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                 cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame
