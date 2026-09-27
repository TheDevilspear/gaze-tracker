import cv2
import time
import numpy as np
from scipy import signal
from collections import deque
import logging

class RPPGTracker:
    """
    Real-time remote photoplethysmography (rPPG) tracker based on the POS
    (Plane-Orthogonal-to-Skin) algorithm from rPPG-Toolbox.
    Extracts heart rate (BPM) from skin color variations in the forehead ROI.
    """
    # Landmarks defining the central forehead skin area (avoids hair and eyes)
    FOREHEAD_LANDMARKS = [109, 10, 338, 297, 9, 67]

    def __init__(self, buffer_seconds=4.0, fps_estimate=25.0, min_bpm=45.0, max_bpm=160.0):
        self.buffer_seconds = buffer_seconds
        self.fps_estimate = fps_estimate
        self.min_freq = min_bpm / 60.0  # ~0.75 Hz
        self.max_freq = max_bpm / 60.0  # ~2.67 Hz

        self.rgb_buffer = deque(maxlen=int(buffer_seconds * 35))
        self.time_buffer = deque(maxlen=int(buffer_seconds * 35))
        self.bpm_history = deque(maxlen=5)
        self.last_bpm = None

    def get_forehead_roi(self, frame, landmarks):
        """Extracts the forehead bounding box and mean RGB values."""
        h, w, _ = frame.shape
        points = []
        for idx in self.FOREHEAD_LANDMARKS:
            lm = landmarks[idx]
            points.append((int(lm.x * w), int(lm.y * h)))

        if not points:
            return None, None

        pts = np.array(points, dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(pts)

        # Shrink slightly to avoid hair/eyebrow edges
        pad_w = int(bw * 0.15)
        pad_h = int(bh * 0.15)
        x1 = max(0, x + pad_w)
        y1 = max(0, y + pad_h)
        x2 = min(w, x + bw - pad_w)
        y2 = min(h, y + bh - pad_h)

        if x2 <= x1 or y2 <= y1:
            return None, None

        roi = frame[y1:y2, x1:x2]
        # Frame is BGR in OpenCV -> convert to RGB for POS
        mean_bgr = np.mean(roi, axis=(0, 1))
        mean_rgb = np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float64)

        return (x1, y1, x2 - x1, y2 - y1), mean_rgb

    def _pos_algorithm(self, rgb_array, fs):
        """POS (Plane-Orthogonal-to-Skin) pulse signal extraction."""
        N = len(rgb_array)
        win_len = int(1.6 * fs)
        if N < win_len:
            return None

        H = np.zeros(N)
        proj = np.array([[0, 1, -1], [-2, 1, 1]], dtype=np.float64)

        for n in range(win_len, N):
            m = n - win_len
            sub_rgb = rgb_array[m:n, :]
            mean_c = np.mean(sub_rgb, axis=0)
            if np.any(mean_c <= 0):
                continue

            cn = sub_rgb / mean_c  # shape (win_len, 3)
            s = np.dot(proj, cn.T)  # shape (2, win_len)

            std0 = np.std(s[0, :])
            std1 = np.std(s[1, :])
            if std1 < 1e-6:
                continue

            alpha = std0 / std1
            h = s[0, :] + alpha * s[1, :]
            h = h - np.mean(h)
            H[m:n] += h

        # Bandpass filter around physiological pulse range
        nyq = 0.5 * fs
        low = self.min_freq / nyq
        high = min(self.max_freq / nyq, 0.99)
        if low >= high:
            return None

        b, a = signal.butter(2, [low, high], btype='bandpass')
        try:
            bvp = signal.filtfilt(b, a, H)
            return bvp
        except Exception:
            return None

    def process_frame(self, frame, landmarks, timestamp=None):
        """
        Processes a single frame:
        Updates rolling color buffer, estimates heart rate (BPM) via POS and FFT.
        Returns: (bpm, roi_box)
        """
        roi_box, mean_rgb = self.get_forehead_roi(frame, landmarks)
        if mean_rgb is None:
            return self.last_bpm, None

        now = timestamp if timestamp is not None else time.time()
        self.rgb_buffer.append(mean_rgb)
        self.time_buffer.append(now)

        # Require at least 2.5 seconds of frames to compute pulse
        min_frames = int(2.5 * self.fps_estimate)
        if len(self.rgb_buffer) < min_frames:
            return None, roi_box

        # Estimate actual FPS from timestamps with sanity bounds
        duration = self.time_buffer[-1] - self.time_buffer[0]
        if duration > 0.5:
            actual_fps = (len(self.time_buffer) - 1) / duration
            if actual_fps < 8.0 or actual_fps > 65.0:
                actual_fps = self.fps_estimate
        else:
            actual_fps = self.fps_estimate

        rgb_arr = np.array(self.rgb_buffer)
        bvp = self._pos_algorithm(rgb_arr, actual_fps)
        if bvp is None or len(bvp) < 32:
            return self.last_bpm, roi_box

        # Frequency analysis (FFT) with zero-padding for high resolution (sub-BPM precision)
        fft_len = max(1024, len(bvp) * 4)
        fft_vals = np.abs(np.fft.rfft(bvp, n=fft_len))
        freqs = np.fft.rfftfreq(fft_len, d=1.0 / actual_fps)

        # Mask frequencies in valid heart rate range
        valid_idx = np.where((freqs >= self.min_freq) & (freqs <= self.max_freq))[0]
        if len(valid_idx) == 0:
            return self.last_bpm, roi_box

        peak_idx = valid_idx[np.argmax(fft_vals[valid_idx])]
        instant_bpm = float(freqs[peak_idx] * 60.0)

        # Moving median smoothing to suppress sudden movement artifacts
        self.bpm_history.append(instant_bpm)
        self.last_bpm = float(np.median(self.bpm_history))

        return self.last_bpm, roi_box
