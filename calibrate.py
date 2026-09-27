import cv2
import mediapipe as mp
import numpy as np
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT,
    CALIBRATION_FILE
)
from core.camera import CameraStream
from core.gaze import get_gaze_features, GazeCalibrator

def run_calibration():
    margin_x = SCREEN_WIDTH // 10
    margin_y = SCREEN_HEIGHT // 10

    # 5 High-contrast anchor points (fast, 10-second calibration with head pose decoupling)
    calib_points = {
        "top-left": (margin_x, margin_y),
        "top-right": (SCREEN_WIDTH - margin_x, margin_y),
        "bottom-left": (margin_x, SCREEN_HEIGHT - margin_y),
        "bottom-right": (SCREEN_WIDTH - margin_x, SCREEN_HEIGHT - margin_y),
        "center": (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    }

    # 4 edge midpoints to validate accuracy across screen
    val_points = {
        "mid-top": (SCREEN_WIDTH // 2, margin_y),
        "mid-bottom": (SCREEN_WIDTH // 2, SCREEN_HEIGHT - margin_y),
        "mid-left": (margin_x, SCREEN_HEIGHT // 2),
        "mid-right": (SCREEN_WIDTH - margin_x, SCREEN_HEIGHT // 2)
    }

    mp_face_mesh = mp.solutions.face_mesh
    calibrator = GazeCalibrator()

    with CameraStream() as stream:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            cv2.namedWindow("Calibration", cv2.WINDOW_AUTOSIZE)

            # --- Phase 1: Calibration with Head-Pose Decoupling ---
            calib_features = []
            screen_coords = []

            for name, pos in calib_points.items():
                print(f"\n>>> Look at {name} dot. Click the 'Calibration' window and press SPACE (or ENTER).")
                while True:
                    ret, frame = stream.read()
                    if not ret or frame is None:
                        continue

                    h, w, _ = frame.shape
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)

                    display = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
                    dh, dw, _ = display.shape

                    dot_x = int(pos[0] * dw / SCREEN_WIDTH)
                    dot_y = int(pos[1] * dh / SCREEN_HEIGHT)
                    cv2.circle(display, (dot_x, dot_y), 15, (0, 0, 255), -1)
                    cv2.putText(display, f"Look at {name} & press SPACE/ENTER",
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                    if results.multi_face_landmarks:
                        cv2.putText(display, "[READY] Face Detected", (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    else:
                        cv2.putText(display, "[WAIT] Face not detected", (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                    cv2.imshow("Calibration", display)

                    key = cv2.waitKey(10) & 0xFF
                    if key == 27:
                        print("Calibration cancelled.")
                        cv2.destroyAllWindows()
                        return
                    elif key in (32, 13, 10, ord(' '), ord('\r'), ord('\n')):
                        if results.multi_face_landmarks:
                            burst_feats = []
                            BURST_TOTAL = 15
                            print(f"Holding gaze on '{name}', capturing {BURST_TOTAL} frames...")
                            for _ in range(BURST_TOTAL):
                                b_ret, b_frame = stream.read()
                                if not b_ret or b_frame is None:
                                    continue
                                b_rgb = cv2.cvtColor(b_frame, cv2.COLOR_BGR2RGB)
                                b_res = face_mesh.process(b_rgb)
                                if b_res.multi_face_landmarks:
                                    b_lm = b_res.multi_face_landmarks[0].landmark
                                    b_f = get_gaze_features(b_lm, w, h)
                                    if b_f is not None:
                                        burst_feats.append(b_f)

                                b_disp = cv2.resize(b_frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
                                cv2.circle(b_disp, (dot_x, dot_y), 18, (0, 255, 0), -1)
                                cv2.putText(b_disp, f"Capturing: {len(burst_feats)}/{BURST_TOTAL} (Hold Still)",
                                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                                cv2.imshow("Calibration", b_disp)
                                cv2.waitKey(25)

                            if len(burst_feats) >= 8:
                                avg_feat = np.median(burst_feats, axis=0)
                                calib_features.append(avg_feat)
                                screen_coords.append(pos)
                                print(f"Captured '{name}': EyeRatio=({avg_feat[0]:.3f}, {avg_feat[1]:.3f}), Head=(Yaw={avg_feat[2]:.1f}, Pitch={avg_feat[3]:.1f})")
                                break
                            else:
                                print("Face lost during capture, please hold still and press SPACE/ENTER again.")
                        else:
                            print("No face detected, please look directly at camera.")

            # Fit Head-Pose Decoupled Model
            calibrator.fit(calib_features, screen_coords)
            print("\nCalibration matrix calculated successfully with Head-Pose Decoupling.")

            # --- Phase 2: Quantitative Validation ---
            val_samples = []
            print("\n--- Starting Accuracy Validation (4 Points) ---")
            for name, pos in val_points.items():
                print(f">>> Look at validation dot '{name}'. Press SPACE/ENTER.")
                while True:
                    ret, frame = stream.read()
                    if not ret or frame is None:
                        continue

                    h, w, _ = frame.shape
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb)

                    display = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
                    dh, dw, _ = display.shape

                    dot_x = int(pos[0] * dw / SCREEN_WIDTH)
                    dot_y = int(pos[1] * dh / SCREEN_HEIGHT)
                    cv2.circle(display, (dot_x, dot_y), 15, (255, 165, 0), -1)
                    cv2.putText(display, f"Validation: Look at {name} & press SPACE/ENTER",
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 0), 2)
                    cv2.imshow("Calibration", display)

                    key = cv2.waitKey(10) & 0xFF
                    if key == 27:
                        break
                    elif key in (32, 13, 10, ord(' '), ord('\r'), ord('\n')):
                        if results.multi_face_landmarks:
                            burst_feats = []
                            BURST_TOTAL = 15
                            print(f"Validating '{name}', measuring {BURST_TOTAL} frames...")
                            for _ in range(BURST_TOTAL):
                                b_ret, b_frame = stream.read()
                                if not b_ret or b_frame is None:
                                    continue
                                b_rgb = cv2.cvtColor(b_frame, cv2.COLOR_BGR2RGB)
                                b_res = face_mesh.process(b_rgb)
                                if b_res.multi_face_landmarks:
                                    b_lm = b_res.multi_face_landmarks[0].landmark
                                    b_f = get_gaze_features(b_lm, w, h)
                                    if b_f is not None:
                                        burst_feats.append(b_f)

                                b_disp = cv2.resize(b_frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
                                cv2.circle(b_disp, (dot_x, dot_y), 18, (0, 255, 255), -1)
                                cv2.putText(b_disp, f"Validating: {len(burst_feats)}/{BURST_TOTAL} (Hold Still)",
                                            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                                cv2.imshow("Calibration", b_disp)
                                cv2.waitKey(25)

                            if len(burst_feats) >= 8:
                                avg_feat = np.median(burst_feats, axis=0)
                                px, py = calibrator.predict(avg_feat)
                                err = np.linalg.norm(np.array(pos) - np.array([px, py]))
                                val_samples.append((avg_feat, pos))
                                print(f"  Target '{name}': Target={pos}, Pred=({px}, {py}), Error={err:.1f}px")
                                break
                            else:
                                print("Face lost during validation. Please hold still and press SPACE/ENTER again.")
                        else:
                            print("No face detected.")

            cv2.destroyAllWindows()

            # Report Score
            mean_err, rel_err = calibrator.evaluate(val_samples)
            print("\n" + "=" * 46)
            print("🎯 ACCURACY BENCHMARK RESULT:")
            print(f"   Mean Absolute Error: {mean_err:.1f} pixels")
            print(f"   Relative Error:      {rel_err:.1f}% of screen width")
            print("=" * 46 + "\n")

            calibrator.save(CALIBRATION_FILE)
            print(f"✅ Calibration saved to '{CALIBRATION_FILE}'. Ready for main.py!")

if __name__ == "__main__":
    run_calibration()
