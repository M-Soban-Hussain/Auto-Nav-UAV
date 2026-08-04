import airsim
import numpy as np
import cv2
import time
import threading
import sys

from ultralytics import YOLO
from PyQt5 import QtWidgets, QtCore, QtGui


# =========================
# CONFIG (HARD-CODED)
# =========================
SCREEN_W = 1366
SCREEN_H = 768

INFER_W = 640
INFER_H = 360

FPS_LIMIT = 10

MODEL_NAME = "yolov8n.pt"

latest_boxes = []
lock = threading.Lock()


# =========================
# AIRSIM SETUP
# =========================
client = airsim.MultirotorClient()
client.confirmConnection()

model = YOLO(MODEL_NAME)


# =========================
# DETECTION LOOP
# =========================
def detection_loop():
    global latest_boxes

    last_time = 0

    while True:

        if time.time() - last_time < 1 / FPS_LIMIT:
            continue

        last_time = time.time()

        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
        ])

        if not responses or responses[0].image_data_uint8 is None:
            continue

        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img = img1d.reshape(responses[0].height, responses[0].width, 3)

        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # resize for YOLO
        frame_small = cv2.resize(frame, (INFER_W, INFER_H))

        results = model(frame_small)[0]

        boxes = []

        if results.boxes is not None:
            for box in results.boxes:

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]

                boxes.append((x1, y1, x2, y2, name, conf))

        with lock:
            latest_boxes = boxes


# =========================
# TRANSPARENT HUD OVERLAY
# =========================
class Overlay(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()

        # transparent always-on-top window
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )

        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # FIXED RESOLUTION
        self.setGeometry(0, 0, SCREEN_W, SCREEN_H)

        # refresh loop
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

    # =========================
    # DRAW HUD ONLY (NO VIDEO)
    # =========================
    def paintEvent(self, event):

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        W = SCREEN_W
        H = SCREEN_H

        # border
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 150, 255), 2))
        painter.drawRect(0, 0, W - 1, H - 1)

        with lock:
            boxes = latest_boxes.copy()

        # scale YOLO (640×360 → 1366×768)
        sx = W / INFER_W
        sy = H / INFER_H

        for (x1, y1, x2, y2, name, conf) in boxes:

            x1 *= sx
            x2 *= sx
            y1 *= sy
            y2 *= sy

            # bounding box
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0), 2))
            painter.drawRect(
                int(x1),
                int(y1),
                int(x2 - x1),
                int(y2 - y1)
            )

            # label
            label = f"{name} {conf:.2f}"

            painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), 2))
            painter.drawText(int(x1), int(y1) - 5, label)


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("Starting AirSim YOLO HUD (clean version)...")

    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

    app = QtWidgets.QApplication(sys.argv)

    overlay = Overlay()
    overlay.show()

    sys.exit(app.exec_())
