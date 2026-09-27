import cv2
import numpy as np
from math import hypot
from config import (
    LEFT_EYE_EAR_INDICES, RIGHT_EYE_EAR_INDICES,
    LEFT_PUPIL_INDICES, RIGHT_PUPIL_INDICES
)

def get_landmark_coords(landmarks, indices, frame_width=1, frame_height=1):
    """Converts normalized landmarks to pixel or relative coordinates."""
    return [(int(landmarks[idx].x * frame_width), int(landmarks[idx].y * frame_height)) for idx in indices]

def calculate_ear(landmarks):
    """Calculates Eye Aspect Ratio (EAR) for blink detection with zero-division safeguard."""
    def euclidean_distance(p1, p2):
        return hypot(p2[0] - p1[0], p2[1] - p1[1])

    left = get_landmark_coords(landmarks, LEFT_EYE_EAR_INDICES)
    right = get_landmark_coords(landmarks, RIGHT_EYE_EAR_INDICES)

    denom_left = 2.0 * euclidean_distance(left[0], left[3])
    denom_right = 2.0 * euclidean_distance(right[0], right[3])

    left_ear = (euclidean_distance(left[1], left[5]) + euclidean_distance(left[2], left[4])) / denom_left if denom_left > 0 else 0.0
    right_ear = (euclidean_distance(right[1], right[5]) + euclidean_distance(right[2], right[4])) / denom_right if denom_right > 0 else 0.0

    return (left_ear + right_ear) / 2.0

def get_pupil_diameter(landmarks):
    """Calculates average pupil diameter across left and right eyes."""
    left = get_landmark_coords(landmarks, LEFT_PUPIL_INDICES)
    right = get_landmark_coords(landmarks, RIGHT_PUPIL_INDICES)
    if left and right:
        left_d = hypot(left[0][0] - left[1][0], left[0][1] - left[1][1])
        right_d = hypot(right[0][0] - right[1][0], right[0][1] - right[1][1])
        return (left_d + right_d) / 2.0
    return -1.0

def get_head_pose_angles(landmarks, frame_width, frame_height):
    """Estimates head pose (yaw, pitch, roll) via perspective-n-point (PnP)."""
    image_points = np.array([
        get_landmark_coords(landmarks, [33], frame_width, frame_height)[0],
        get_landmark_coords(landmarks, [263], frame_width, frame_height)[0],
        get_landmark_coords(landmarks, [1], frame_width, frame_height)[0],
        get_landmark_coords(landmarks, [61], frame_width, frame_height)[0],
        get_landmark_coords(landmarks, [291], frame_width, frame_height)[0],
        get_landmark_coords(landmarks, [199], frame_width, frame_height)[0]
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),         # Chin
        (-225.0, 170.0, -135.0),      # Left eye left corner
        (225.0, 170.0, -135.0),       # Right eye right corner
        (-150.0, -150.0, -125.0),     # Left Mouth corner
        (150.0, -150.0, -125.0)       # Right mouth corner
    ])

    center = (frame_height / 2.0, frame_width / 2.0)
    focal_length = center[0] / np.tan(30.0 * np.pi / 180.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        rmat, _ = cv2.Rodrigues(rvec)
        proj_matrix = cv2.hconcat((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix, camera_matrix, dist_coeffs)
        yaw = float(euler_angles[1, 0])
        pitch = float(euler_angles[0, 0])
        roll = float(euler_angles[2, 0])
        return yaw, pitch, roll
    return None, None, None

def get_aoi_metrics(gaze_x, gaze_y, aoi_regions):
    """Determines which Area of Interest (AOI) contains the current gaze coordinates."""
    if gaze_x < 0 or gaze_y < 0:
        return None
    for aoi_name, (x1, y1, x2, y2) in aoi_regions.items():
        if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
            return aoi_name
    return None
