import cv2
import logging
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, CODEC, WARMUP_FRAMES

class CameraStream:
    """Manages webcam capture, optimized for USB/IP and Linux V4L2 streaming."""
    def __init__(self, source=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT):
        self.source = int(source) if str(source).isdigit() else source
        self.width = width
        self.height = height
        self.cap = None

    def open(self):
        if isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CODEC))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                for _ in range(WARMUP_FRAMES):
                    self.cap.read()
        else:
            self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open video source '{self.source}'. "
                "In WSL, verify USB/IP attachment with 'usbipd wsl list' and permissions."
            )
        logging.info(f"Camera opened: {self.source} ({self.width}x{self.height})")
        return self

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
