import os
import sys
import time
import queue
import argparse
import warnings
import logging
import threading
import numpy as np

# Suppress noisy third-party warnings and logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

import cv2
import mediapipe as mp

from config import (
    CAMERA_INDEX, SAMPLE_AOIS, BLINK_THRESHOLD,
    CALIBRATION_FILE, SCREEN_WIDTH, SCREEN_HEIGHT
)
from core.camera import CameraStream
from core.gaze import get_combined_eye_ratio, get_gaze_features, GazeCalibrator, EMASmoother
from core.head_pose import (
    calculate_ear, get_pupil_diameter,
    get_head_pose_angles, get_aoi_metrics
)
from core.rppg import RPPGTracker
from core.attention import OffScreenTracker
from services.audio import AudioTranscriber
from services.storage import (
    DBWriterThread, export_numerical_csv,
    export_attention_heatmap, export_gaze_path
)

def main():
    parser = argparse.ArgumentParser(description="Unified Multimodal Eye Tracking & Speech Analysis")
    parser.add_argument("--source", default=CAMERA_INDEX, help="Camera index or path to video file")
    parser.add_argument("--mouse", action="store_true", help="Control mouse cursor with gaze")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio transcription")
    parser.add_argument("--no-db", action="store_true", help="Disable MongoDB writing")
    parser.add_argument("--no-rppg", action="store_true", help="Disable rPPG heart rate tracking")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Load Calibration
    calibrator = GazeCalibrator()
    has_calibration = calibrator.load(CALIBRATION_FILE)
    if has_calibration:
        print(f"✅ Loaded screen calibration from '{CALIBRATION_FILE}'.")
    else:
        print("⚠️ No calibration file found. Run 'python calibrate.py' for accurate screen gaze.")

    # Initialize Queues & Threads
    stop_event = threading.Event()
    db_queue = queue.Queue(maxsize=2000)

    db_writer = None
    if not args.no_db:
        db_writer = DBWriterThread(db_queue, stop_event)
        db_writer.start()

    audio_thread = None
    if not args.no_audio:
        audio_thread = AudioTranscriber(db_queue, stop_event)
        audio_thread.start()

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Trackers
    rppg_tracker = RPPGTracker() if not args.no_rppg else None
    gaze_smoother = EMASmoother(alpha=0.35)
    attention_tracker = OffScreenTracker()

    numerical_records = []
    gaze_history = []
    blink_counter = 0
    start_time = time.time()
    last_frame = None

    cv2.namedWindow("Live Gaze & Behavioral Analysis", cv2.WINDOW_AUTOSIZE)

    print("\nStarting live analysis. Press 'q' inside video window to exit.")
    if args.mouse:
        print("🖱️ Mouse control enabled: cursor follows gaze point.")

    with CameraStream(source=args.source) as stream:
        heatmap_accumulator = np.zeros((stream.height, stream.width), dtype=np.float32)

        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                break

            last_frame = frame.copy()
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            current_time = time.time() - start_time
            gaze_screen_x, gaze_screen_y = -1, -1
            ear = 0.0
            blink_status = 0
            pupil_d = -1.0
            yaw, pitch, roll = None, None, None
            current_aoi = None
            current_bpm = None
            face_detected = False

            if results.multi_face_landmarks:
                face_detected = True
                landmarks = results.multi_face_landmarks[0].landmark

                # Compute Screen Gaze with Head-Pose Decoupling
                feat = get_gaze_features(landmarks, w, h)
                if feat is not None:
                    if has_calibration:
                        raw_gx, raw_gy = calibrator.predict(feat)
                    else:
                        raw_gx = int(feat[0] * SCREEN_WIDTH)
                        raw_gy = int(feat[1] * SCREEN_HEIGHT)

                    # Smooth out micro-jitter using EMA
                    smooth_gaze = gaze_smoother.update([raw_gx, raw_gy])
                    gaze_screen_x = int(smooth_gaze[0])
                    gaze_screen_y = int(smooth_gaze[1])

                    # Optional mouse control
                    if args.mouse and gaze_screen_x >= 0 and gaze_screen_y >= 0:
                        try:
                            import pyautogui
                            pyautogui.moveTo(gaze_screen_x, gaze_screen_y, _pause=False)
                        except Exception:
                            pass

                # Pupil diameter, EAR & Blinks
                pupil_d = get_pupil_diameter(landmarks)
                ear = calculate_ear(landmarks)
                if ear < BLINK_THRESHOLD:
                    blink_status = 1
                    blink_counter += 1

                # Head Pose & AOI
                yaw, pitch, roll = get_head_pose_angles(landmarks, w, h)
                current_aoi = get_aoi_metrics(gaze_screen_x, gaze_screen_y, SAMPLE_AOIS)

                # Visual feedback on eyes (iris centers)
                for iris_idx in (468, 473):
                    pt = landmarks[iris_idx]
                    cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 2, (0, 255, 255), -1)

                # rPPG Pulse / Heart Rate
                if rppg_tracker is not None:
                    current_bpm, _ = rppg_tracker.process_frame(frame, landmarks, timestamp=time.time())

            # --- Off-Screen Attention / Distraction Detection ---
            # Triggers ONLY when face is absent, or head turns/tilts far away
            is_offscreen_now = False
            if not face_detected:
                is_offscreen_now = True
            elif yaw is not None and (abs(yaw) > 35 or pitch < -40):
                is_offscreen_now = True

            attention_stats = attention_tracker.update(is_offscreen_now, current_time)

            # Record numerical metrics
            doc = {
                "timestamp": current_time,
                "gaze_x": gaze_screen_x,
                "gaze_y": gaze_screen_y,
                "pupil_diameter": pupil_d,
                "ear": ear,
                "blink_status": blink_status,
                "bpm": current_bpm,
                "head_yaw": yaw,
                "head_pitch": pitch,
                "head_roll": roll,
                "aoi": current_aoi,
                "is_offscreen": attention_stats["is_offscreen"],
                "offscreen_count": attention_stats["offscreen_count"],
                "offscreen_max_duration_sec": attention_stats["max_offscreen_sec"],
                "offscreen_avg_duration_sec": attention_stats["avg_offscreen_sec"],
                "offscreen_avg_interval_sec": attention_stats["avg_interval_sec"],
                "offscreen_current_duration_sec": attention_stats["current_offscreen_sec"]
            }
            numerical_records.append(doc)

            if not args.no_db:
                try:
                    db_queue.put_nowait({"type": "numerical", "data": doc})
                except queue.Full:
                    pass

            # --- HUD Display & Mini Screen Radar ---
            display = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
            dw, dh = display.shape[1], display.shape[0]

            # 1. Attention Status Header Banner
            if attention_stats["is_offscreen"]:
                att_text = f"ATTENTION: OFF-SCREEN ({attention_stats['current_offscreen_sec']}s)"
                att_color = (0, 0, 255)  # Red Alert
            else:
                att_text = "ATTENTION: ON-SCREEN"
                att_color = (0, 255, 0)  # Green Focused

            cv2.putText(display, att_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, att_color, 2)
            cv2.putText(display, f"Gaze: ({gaze_screen_x}, {gaze_screen_y}) | AOI: {current_aoi or 'None'}", (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

            bpm_str = f"{current_bpm:.0f}" if current_bpm else "Measuring..."
            cv2.putText(display, f"EAR: {ear:.2f} | Blinks: {blink_counter} | Pulse: {bpm_str} BPM", (15, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 200, 0), 1)

            cv2.putText(display, f"Off-Screen: {attention_stats['offscreen_count']}x | Max Stay: {attention_stats['max_offscreen_sec']}s | Frequency: {attention_stats['avg_interval_sec']}s",
                        (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 255), 1)

            if yaw is not None:
                cv2.putText(display, f"Head: Yaw={yaw:.1f} Pitch={pitch:.1f}", (15, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 0), 1)

            # 2. Live Mini-Monitor Radar (Bottom-Right Corner)
            # Shows where on your 1366x768 screen your eyes are looking in real-time
            rw, rh = 160, 90
            rx_pos = dw - rw - 15
            ry_pos = dh - rh - 15
            cv2.rectangle(display, (rx_pos, ry_pos), (rx_pos + rw, ry_pos + rh), (25, 25, 25), -1)
            cv2.rectangle(display, (rx_pos, ry_pos), (rx_pos + rw, ry_pos + rh), (180, 180, 180), 1)
            cv2.putText(display, "MONITOR", (rx_pos + 6, ry_pos + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1)

            if gaze_screen_x >= 0 and gaze_screen_y >= 0:
                radar_gx = rx_pos + int(np.clip(gaze_screen_x * rw / SCREEN_WIDTH, 0, rw))
                radar_gy = ry_pos + int(np.clip(gaze_screen_y * rh / SCREEN_HEIGHT, 0, rh))
                gaze_history.append((radar_gx, radar_gy))
                if len(gaze_history) > 15:
                    gaze_history.pop(0)

                # Draw recent gaze path in mini monitor
                for i in range(1, len(gaze_history)):
                    cv2.line(display, gaze_history[i-1], gaze_history[i], (0, 255, 128), 1)

                # Draw current target dot & crosshair
                cv2.drawMarker(display, (radar_gx, radar_gy), (0, 0, 255), cv2.MARKER_CROSS, 8, 1)
                cv2.circle(display, (radar_gx, radar_gy), 4, (0, 255, 255), -1)

            cv2.imshow("Live Gaze & Behavioral Analysis", display)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # Teardown
    cv2.destroyAllWindows()
    stop_event.set()

    if audio_thread:
        audio_thread.join(timeout=2.0)
    if db_writer:
        db_writer.join(timeout=3.0)

    # Export Session Artifacts
    export_numerical_csv(numerical_records)
    if last_frame is not None:
        export_attention_heatmap(heatmap_accumulator, last_frame)
        export_gaze_path(gaze_history, last_frame.shape[1], last_frame.shape[0])

    print("\nSession complete. Data and plots exported.")

if __name__ == "__main__":
    main()
