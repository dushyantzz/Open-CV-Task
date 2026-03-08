"""
Warehouse Management System – Bag Counting Dashboard
=====================================================
Layout from WareHouse_DashBoard.png:
  - Title: "Warehouse Management System" / "Bag Counting Management System"
  - 3 camera panels (Gate 1/2/3) with Bags In & Bags Out
  - IOT Parameters Monitoring panel (dummy values)

Gate 1 runs live YOLO detection + tracking. Gates 2 & 3 play video only.

Launch:
    python dashboard.py
    python dashboard.py --video1 "Problem Statement Scenario1.mp4"
"""

import sys
import os
import cv2
import time
import random
import threading
import queue
import numpy as np
from pathlib import Path

try:
    from detector import Detector
    from tracker import SackCounter
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False
    print("[WARNING] YOLO/ultralytics not available – dashboard will run without detection.")

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QGridLayout,
    QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

BASE_DIR = Path(__file__).resolve().parent

DARK_BG      = "#1a2733"
PANEL_BG     = "#223344"
BORDER_COLOR = "#3a8a9e"
TITLE_COLOR  = "#f0c040"
TEXT_COLOR    = "#d0d8e0"
ACCENT_TEAL  = "#2a8a9e"
LABEL_ORANGE = "#e8a020"


def default_videos():
    names = [
        "Problem Statement Scenario1.mp4",
        "Problem Statement Scenario2.mp4",
        "Problem Statement Scenario3.mp4",
    ]
    return [str(BASE_DIR / n) if (BASE_DIR / n).exists() else None for n in names]


class FrameGrabber:
    """Background thread that reads video frames, optionally runs YOLO,
    and puts results into a thread-safe queue (no Qt signals from threads)."""

    def __init__(self, gate_idx, video_path, frame_queue, run_detection=False):
        self.gate_idx = gate_idx
        self.video_path = video_path
        self.run_detection = run_detection
        self._queue = frame_queue
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=False)
        self._thread.start()

    def stop(self):
        self._running = False

    def join(self, timeout=3):
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _loop(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[Gate {self.gate_idx + 1}] Cannot open {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        delay = 1.0 / fps

        detector = None
        counter = None
        if self.run_detection and YOLO_AVAILABLE:
            detector = Detector(
                model_path=str(BASE_DIR / "yolo11n.pt"),
                conf=0.30, iou=0.50, img_size=640,
            )
            counter = SackCounter(line_x_ratio=0.50, gate_number=self.gate_idx + 1)

        bags_in, bags_out = 0, 0
        last_results = None
        frame_idx = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if counter:
                    counter = SackCounter(line_x_ratio=0.50, gate_number=self.gate_idx + 1)
                    bags_in, bags_out = 0, 0
                continue

            if detector and counter:
                last_results = detector.detect_and_track(frame)
                bags_in, bags_out = counter.process_tracks(last_results, frame)
                frame = counter.draw_overlay(frame, last_results)

            try:
                self._queue.put_nowait((self.gate_idx, frame, bags_in, bags_out))
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait((self.gate_idx, frame, bags_in, bags_out))

            frame_idx += 1
            time.sleep(delay)

        cap.release()


class GatePanel(QFrame):
    """One camera panel: video + gate label + bags in/out counters."""

    def __init__(self, gate_number, parent=None):
        super().__init__(parent)
        self.gate_number = gate_number
        self.setStyleSheet(f"""
            GatePanel {{
                background: {PANEL_BG};
                border: 2px solid {BORDER_COLOR};
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(280, 180)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet(
            "background: #111; border: 1px solid #444; border-radius: 4px;"
        )
        layout.addWidget(self.video_label)

        info = QWidget()
        info.setStyleSheet("background: transparent;")
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(4, 2, 4, 2)
        info_layout.setSpacing(2)

        self.gate_label = QLabel(f"Gate Number :  {gate_number:02d}")
        self.gate_label.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.gate_label.setStyleSheet(f"color: {TITLE_COLOR};")
        info_layout.addWidget(self.gate_label)

        row = QHBoxLayout()
        self.bags_in_label = QLabel("Bags In : 0")
        self.bags_in_label.setFont(QFont("Segoe UI", 10))
        self.bags_in_label.setStyleSheet(f"color: {TEXT_COLOR};")
        self.bags_out_label = QLabel("Bags Out : 0")
        self.bags_out_label.setFont(QFont("Segoe UI", 10))
        self.bags_out_label.setStyleSheet(f"color: {TEXT_COLOR};")
        row.addWidget(self.bags_in_label)
        row.addWidget(self.bags_out_label)
        info_layout.addLayout(row)

        layout.addWidget(info)

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        lbl_w = self.video_label.width()
        lbl_h = self.video_label.height()
        if lbl_w > 10 and lbl_h > 10:
            scale = min(lbl_w / w, lbl_h / h)
            nw, nh = int(w * scale), int(h * scale)
            rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg))

    def update_counts(self, bags_in, bags_out):
        self.bags_in_label.setText(f"Bags In : {bags_in}")
        self.bags_out_label.setText(f"Bags Out : {bags_out}")


class IOTPanel(QFrame):
    """IOT Parameters Monitoring – dummy sensor values."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            IOTPanel {{
                background: {PANEL_BG};
                border: 2px solid {BORDER_COLOR};
                border-radius: 6px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(6)

        title = QLabel("IOT Parameters Monitoring")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TITLE_COLOR};")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(40)
        grid.setVerticalSpacing(8)

        self.labels = {}
        params = [
            ("Temperature", 0, 0),          ("Smoke and Fire Status", 0, 1),
            ("Humidity", 1, 0),              ("Gate Open/Close Status", 1, 1),
            ("Phosphine Gas Level", 2, 0),
        ]
        for name, row, col in params:
            lbl = QLabel(f"{name} :")
            lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
            lbl.setStyleSheet(f"color: {TEXT_COLOR};")
            val = QLabel("--")
            val.setFont(QFont("Segoe UI", 10))
            val.setStyleSheet(f"color: {LABEL_ORANGE};")
            h_box = QHBoxLayout()
            h_box.addWidget(lbl)
            h_box.addWidget(val)
            h_box.addStretch()
            grid.addLayout(h_box, row, col)
            self.labels[name] = val
        layout.addLayout(grid)

    def refresh_dummy(self):
        self.labels["Temperature"].setText(f"{random.uniform(25, 38):.1f} \u00b0C")
        self.labels["Humidity"].setText(f"{random.uniform(40, 75):.1f} %")
        self.labels["Phosphine Gas Level"].setText(
            f"{random.uniform(0, 0.5):.2f} ppm" if random.random() > 0.1 else "ALERT 1.2 ppm"
        )
        self.labels["Smoke and Fire Status"].setText(
            random.choice(["Normal", "Normal", "Normal", "Smoke Detected"])
        )
        self.labels["Gate Open/Close Status"].setText(
            random.choice(["Open", "Open", "Closed"])
        )


class WarehouseDashboard(QMainWindow):
    def __init__(self, video_paths):
        super().__init__()
        self.setWindowTitle("Warehouse Management System")
        self.setMinimumSize(1100, 720)
        self._apply_palette()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(8)

        title1 = QLabel("Warehouse Management System")
        title1.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title1.setAlignment(Qt.AlignCenter)
        title1.setStyleSheet(f"color: {TITLE_COLOR};")
        root.addWidget(title1)

        subtitle = QLabel("Bag Counting Management System")
        subtitle.setFont(QFont("Segoe UI", 12, QFont.Bold))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(
            f"color: {TITLE_COLOR}; background: {ACCENT_TEAL}; "
            "border-radius: 4px; padding: 4px 16px;"
        )
        root.addWidget(subtitle, alignment=Qt.AlignCenter)

        cam_row = QHBoxLayout()
        cam_row.setSpacing(10)
        self.gate_panels = []
        for i in range(3):
            p = GatePanel(i + 1)
            cam_row.addWidget(p)
            self.gate_panels.append(p)
        root.addLayout(cam_row, stretch=3)

        self.iot_panel = IOTPanel()
        root.addWidget(self.iot_panel, stretch=1)

        self._frame_queue = queue.Queue(maxsize=30)
        self.grabbers = []
        for idx, vpath in enumerate(video_paths):
            if vpath and os.path.isfile(vpath):
                g = FrameGrabber(idx, vpath, self._frame_queue, run_detection=True)
                self.grabbers.append(g)

        for g in self.grabbers:
            g.start()

        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._poll_frames)
        self._poll_timer.start(30)

        self._iot_timer = QTimer()
        self._iot_timer.timeout.connect(self.iot_panel.refresh_dummy)
        self._iot_timer.start(3000)
        self.iot_panel.refresh_dummy()

    def _apply_palette(self):
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(DARK_BG))
        pal.setColor(QPalette.WindowText, QColor(TEXT_COLOR))
        self.setPalette(pal)
        self.setStyleSheet(f"background: {DARK_BG};")

    def _poll_frames(self):
        processed = 0
        while processed < 9:
            try:
                gate_idx, frame, bags_in, bags_out = self._frame_queue.get_nowait()
            except queue.Empty:
                break
            if 0 <= gate_idx < len(self.gate_panels):
                self.gate_panels[gate_idx].update_frame(frame)
                self.gate_panels[gate_idx].update_counts(bags_in, bags_out)
            processed += 1

    def closeEvent(self, event):
        self._poll_timer.stop()
        self._iot_timer.stop()
        for g in self.grabbers:
            g.stop()
        for g in self.grabbers:
            g.join(timeout=2)
        event.accept()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Warehouse Dashboard")
    parser.add_argument("--video1", type=str, default=None)
    parser.add_argument("--video2", type=str, default=None)
    parser.add_argument("--video3", type=str, default=None)
    args = parser.parse_args()

    defaults = default_videos()
    videos = [
        args.video1 or defaults[0],
        args.video2 or defaults[1],
        args.video3 or defaults[2],
    ]

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = WarehouseDashboard(videos)
    win.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
