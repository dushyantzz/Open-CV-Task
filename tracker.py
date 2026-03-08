import cv2
import numpy as np


class SackCounter:
    """
    Counts bags being loaded (IN) or unloaded (OUT) by tracking people
    carrying sacks across a vertical counting line.

    Movement is mostly horizontal in warehouse videos (person walks
    left <-> right between truck and staging area).

    Each tracked person is counted once when their center-x crosses
    the line. After counting, the ID enters cooldown so the same
    person isn't double-counted on their return trip within a short
    window.
    """

    def __init__(self, line_x_ratio=0.50, gate_number=1):
        self.line_x_ratio = line_x_ratio
        self.gate_number = gate_number
        self.bags_in = 0
        self.bags_out = 0
        self.tracked_objects = {}
        self.stale_timeout = 90
        self.cooldown_frames = 40

    def process_tracks(self, results, frame):
        h, w = frame.shape[:2]
        line_x = int(w * self.line_x_ratio)

        if not results or results[0].boxes.id is None:
            self._age_tracks()
            return self.bags_in, self.bags_out

        ids = results[0].boxes.id.cpu().numpy().astype(int)
        bboxes = results[0].boxes.xyxy.cpu().numpy()

        active_ids = set()
        for obj_id, bbox in zip(ids, bboxes):
            cx = (bbox[0] + bbox[2]) / 2
            active_ids.add(obj_id)

            if obj_id not in self.tracked_objects:
                side = 'left' if cx < line_x else 'right'
                self.tracked_objects[obj_id] = {
                    'start_side': side,
                    'last_x': cx,
                    'cooldown': 0,
                    'counted': False,
                    'age': 0,
                }
            else:
                track = self.tracked_objects[obj_id]
                track['age'] = 0

                if track['cooldown'] > 0:
                    track['cooldown'] -= 1
                    track['last_x'] = cx
                    continue

                prev_x = track['last_x']
                crossed_right = prev_x < line_x and cx >= line_x
                crossed_left = prev_x > line_x and cx <= line_x

                if crossed_right and not track['counted']:
                    self.bags_in += 1
                    track['counted'] = True
                    track['cooldown'] = self.cooldown_frames
                elif crossed_left and not track['counted']:
                    self.bags_out += 1
                    track['counted'] = True
                    track['cooldown'] = self.cooldown_frames

                if track['counted'] and track['cooldown'] == 0:
                    track['counted'] = False

                track['last_x'] = cx

        self._age_tracks(active_ids)
        return self.bags_in, self.bags_out

    def _age_tracks(self, active_ids=None):
        if active_ids is None:
            active_ids = set()
        to_remove = []
        for oid in self.tracked_objects:
            if oid not in active_ids:
                self.tracked_objects[oid]['age'] += 1
                if self.tracked_objects[oid]['age'] > self.stale_timeout:
                    to_remove.append(oid)
        for oid in to_remove:
            del self.tracked_objects[oid]

    def draw_overlay(self, frame, results):
        """Draw vertical counting line and bounding boxes."""
        h, w = frame.shape[:2]
        line_x = int(w * self.line_x_ratio)

        cv2.line(frame, (line_x, 0), (line_x, h), (0, 255, 255), 2)
        cv2.putText(frame, "IN ->", (line_x + 8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "<- OUT", (line_x - 80, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2, cv2.LINE_AA)

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for box, obj_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                track = self.tracked_objects.get(obj_id, {})
                cd = track.get('cooldown', 0)
                color = (0, 255, 0) if cd > 0 else (255, 180, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"#{obj_id}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame
