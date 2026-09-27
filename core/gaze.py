import os
import cv2
import numpy as np
import logging
from config import (
    LEFT_EYE_RATIO_INDICES, RIGHT_EYE_RATIO_INDICES,
    LEFT_IRIS_INDICES, RIGHT_IRIS_INDICES,
    CALIBRATION_FILE, SCREEN_WIDTH, SCREEN_HEIGHT
)

def get_eye_ratio(landmarks, eye_indices, iris_indices, w, h):
    """
    Computes horizontal and vertical iris position ratios relative to eye bounding box.
    Returns: (x_ratio, y_ratio, (iris_x, iris_y)) or (None, None, None)
    """
    try:
        left = int(landmarks[eye_indices[0]].x * w)
        right = int(landmarks[eye_indices[1]].x * w)
        top = int(landmarks[eye_indices[2]].y * h)
        bottom = int(landmarks[eye_indices[3]].y * h)

        iris_x = np.mean([landmarks[i].x for i in iris_indices]) * w
        iris_y = np.mean([landmarks[i].y for i in iris_indices]) * h

        if right == left or bottom == top:
            return None, None, None

        x_ratio = (iris_x - left) / float(right - left)
        y_ratio = (iris_y - top) / float(bottom - top)
        return float(x_ratio), float(y_ratio), (int(iris_x), int(iris_y))
    except Exception:
        return None, None, None

def get_combined_eye_ratio(landmarks, w, h):
    """
    Computes average ratio of left and right eyes, aligned to screen coordinates.
    Inverts horizontal axis (1.0 - raw_x) so looking right corresponds to increasing screen X.
    """
    lx, ly, _ = get_eye_ratio(landmarks, LEFT_EYE_RATIO_INDICES, LEFT_IRIS_INDICES, w, h)
    rx, ry, _ = get_eye_ratio(landmarks, RIGHT_EYE_RATIO_INDICES, RIGHT_IRIS_INDICES, w, h)
    if lx is not None and rx is not None:
        raw_x = (lx + rx) / 2.0
        raw_y = (ly + ry) / 2.0
        screen_x_ratio = 1.0 - raw_x
        return screen_x_ratio, raw_y
    return None, None

def get_gaze_features(landmarks, w, h):
    """
    Extracts coupled 4D feature vector combining eye ratios and head pose angles:
    [screen_x_ratio, eye_y_ratio, head_yaw, head_pitch]
    Allows calibration regression to compensate for natural head turns.
    """
    from core.head_pose import get_head_pose_angles
    rx, ry = get_combined_eye_ratio(landmarks, w, h)
    if rx is None or ry is None:
        return None
    yaw, pitch, _ = get_head_pose_angles(landmarks, w, h)
    yaw_val = float(yaw) if yaw is not None else 0.0
    pitch_val = float(pitch) if pitch is not None else 0.0
    return np.array([rx, ry, yaw_val, pitch_val], dtype=np.float64)

class EMASmoother:
    """Exponential Moving Average filter to smooth landmarks and gaze coordinates."""
    def __init__(self, alpha=0.30):
        self.alpha = alpha
        self.state = None

    def update(self, val):
        if val is None:
            return self.state
        val_arr = np.array(val, dtype=np.float64)
        if self.state is None:
            self.state = val_arr
        else:
            self.state = self.alpha * val_arr + (1.0 - self.alpha) * self.state
        return self.state

    def reset(self):
        self.state = None

class GazeCalibrator:
    """
    Manages calibration calculation, persistence, and screen gaze prediction.
    Couples iris ratios and head pose angles [rx, ry, yaw, pitch, 1.0] with feature
    standardization to decouple head rotation from eye gaze.
    """
    def __init__(self, matrix=None, mean=None, scale=None):
        self.matrix = matrix  # Shape (5, 2) for coupled eye+head model
        self.mean = mean
        self.scale = scale

    def fit(self, features_list, screen_coords):
        """Fits standardized regression on coupled eye + head pose features."""
        feats = np.array(features_list, dtype=np.float64)
        coords = np.array(screen_coords, dtype=np.float64)

        # Standardize features (mean=0, std=1) to prevent scale imbalance
        self.mean = np.mean(feats, axis=0)
        self.scale = np.std(feats, axis=0)
        self.scale[self.scale < 1e-6] = 1.0

        norm_X = (feats - self.mean) / self.scale
        X = np.hstack([norm_X, np.ones((len(feats), 1), dtype=np.float64)])
        A, _, _, _ = np.linalg.lstsq(X, coords, rcond=None)
        self.matrix = A
        return self.matrix

    def predict(self, features):
        """Predicts (screen_x, screen_y) given gaze features [rx, ry, yaw, pitch] or [rx, ry]."""
        if self.matrix is None:
            return -1, -1

        feat_arr = np.array(features, dtype=np.float64).flatten()
        if self.mean is not None and self.scale is not None:
            if len(feat_arr) == len(self.mean):
                u = (feat_arr - self.mean) / self.scale
                vec = np.append(u, 1.0)
            elif len(feat_arr) == 2 and len(self.mean) >= 4:
                u = (feat_arr - self.mean[:2]) / self.scale[:2]
                vec = np.array([u[0], u[1], 0.0, 0.0, 1.0], dtype=np.float64)
            else:
                vec = np.append(feat_arr, 1.0)
        else:
            vec = np.append(feat_arr, 1.0)

        pred = np.dot(vec, self.matrix)
        gx = int(np.clip(pred[0], 0, SCREEN_WIDTH))
        gy = int(np.clip(pred[1], 0, SCREEN_HEIGHT))
        return gx, gy

    def evaluate(self, validation_samples):
        """
        validation_samples: list of tuples (features, (true_x, true_y))
        Returns: mean_error_px, relative_error_pct
        """
        if self.matrix is None or not validation_samples:
            return 0.0, 0.0
        errors = []
        for feat, (tx, ty) in validation_samples:
            px, py = self.predict(feat)
            err = np.linalg.norm(np.array([tx, ty]) - np.array([px, py]))
            errors.append(err)
        mean_err = float(np.mean(errors))
        rel_err = (mean_err / SCREEN_WIDTH) * 100.0
        return mean_err, rel_err

    def save(self, filepath=CALIBRATION_FILE):
        if self.matrix is not None:
            data = {
                "matrix": self.matrix,
                "mean": self.mean,
                "scale": self.scale
            }
            np.save(filepath, data, allow_pickle=True)
            logging.info(f"Calibration saved to {filepath}")

    def load(self, filepath=CALIBRATION_FILE):
        if os.path.exists(filepath):
            loaded = np.load(filepath, allow_pickle=True)
            if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.ndim == 0:
                d = loaded.item()
                self.matrix = d.get("matrix")
                self.mean = d.get("mean")
                self.scale = d.get("scale")
            elif isinstance(loaded, np.ndarray):
                self.matrix = loaded
                self.mean = None
                self.scale = None
            logging.info(f"Calibration loaded from {filepath}")
            return True
        return False
