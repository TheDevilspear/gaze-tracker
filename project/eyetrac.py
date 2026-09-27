import cv2
import mediapipe as mp
import numpy as np
import csv
import pyautogui
from collections import deque

# Screen size
screen_w, screen_h = pyautogui.size()

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [362, 263, 386, 374]
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

def get_eye_ratio(landmarks, eye_points, iris_points, w, h):
    left = int(landmarks[eye_points[0]].x * w)
    right = int(landmarks[eye_points[1]].x * w)
    top = int(landmarks[eye_points[2]].y * h)
    bottom = int(landmarks[eye_points[3]].y * h)

    iris_x = np.mean([landmarks[i].x for i in iris_points]) * w
    iris_y = np.mean([landmarks[i].y for i in iris_points]) * h

    # Prevent division by zero
    if right == left or bottom == top:
        return None, None, None

    x_ratio = (iris_x - left) / (right - left)
    y_ratio = (iris_y - top) / (bottom - top)

    return x_ratio, y_ratio, (int(iris_x), int(iris_y))  # return iris center too


# Store calibration data
margin_x, margin_y = screen_w // 10, screen_h // 10
calibration_points = {
    "top-left": (margin_x, margin_y),
    "top-right": (screen_w - margin_x, margin_y),
    "bottom-left": (margin_x, screen_h - margin_y),
    "bottom-right": (screen_w - margin_x, screen_h - margin_y),
    "center": (screen_w // 2, screen_h // 2)
}

calib_data = {}

# Open camera with V4L2 and MJPG codec to fix green screen in WSL/USB/IP
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera (source 0) could not be opened.")
    print("In WSL, ensure your webcam is attached via USB/IP (e.g. 'usbipd wsl attach') or a virtual camera device is configured.")
    exit(1)

# Motion-JPEG at 320x240 fits completely within the USB/IP transfer buffer
# (640x480 exceeds the USB/IP ISO buffer limit, causing the bottom 70% to truncate to black)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Discard first few warmup frames
for _ in range(5):
    cap.read()

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    cv2.namedWindow("Calibration", cv2.WINDOW_AUTOSIZE)

    # Calibration
    for name, screen_pos in calibration_points.items():
        print(f"\n>>> Step: Look at {name} dot. Click the 'Calibration' window and press SPACE (or ENTER).")
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # Upscale 2x for clean, clear display without black bars
            display = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
            dh, dw, _ = display.shape

            # --- Draw calibration dot on display ---
            dot_x = int(screen_pos[0] * dw / screen_w)
            dot_y = int(screen_pos[1] * dh / screen_h)
            cv2.circle(display, (dot_x, dot_y), 15, (0, 0, 255), -1)  # red dot
            cv2.putText(display, f"Look at {name} & press SPACE/ENTER",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            # Real-time visual feedback on face detection status
            if results.multi_face_landmarks:
                cv2.putText(display, "[READY] Face Detected - Press SPACE or ENTER",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            else:
                cv2.putText(display, "[WAIT] No Face Detected - Look directly at camera",
                            (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            cv2.imshow("Calibration", display)

            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # ESC = exit
                print("Calibration cancelled by user.")
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
            elif key in (32, 13, 10, ord(' '), ord('\r'), ord('\n')):  # Space or Enter
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    lx, ly, _ = get_eye_ratio(lm, LEFT_EYE, LEFT_IRIS, w, h)
                    rx, ry, _ = get_eye_ratio(lm, RIGHT_EYE, RIGHT_IRIS, w, h)
                    if lx is not None and rx is not None:
                        calib_data[name] = ((lx+rx)/2, (ly+ry)/2, screen_pos)
                        print(f"Calibration point '{name}' captured successfully.")
                        break
                    else:
                        print("Iris not detected clearly, please hold still and press SPACE/ENTER again.")
                else:
                    print("No face detected, please look directly at the camera and press SPACE/ENTER.")

    print("Calibration complete ✅")

    # Extract calibration arrays
    eye_ratios = np.array([(v[0], v[1]) for v in calib_data.values()])
    screen_coords = np.array([v[2] for v in calib_data.values()])

    # Fit linear mapping
    X = np.hstack([eye_ratios, np.ones((eye_ratios.shape[0],1))])
    A_x, _, _, _ = np.linalg.lstsq(X, screen_coords[:,0], rcond=None)
    A_y, _, _, _ = np.linalg.lstsq(X, screen_coords[:,1], rcond=None)
    A = np.vstack([A_x, A_y]).T  # shape (3,2)

    print("Calibration matrix:\n", A)

    # --- Quantitative Accuracy Validation ---
    val_points = {
        "mid-top": (screen_w // 2, margin_y),
        "mid-bottom": (screen_w // 2, screen_h - margin_y),
        "mid-left": (margin_x, screen_h // 2),
        "mid-right": (screen_w - margin_x, screen_h // 2)
    }
    val_errors = []
    print("\n--- Starting Accuracy Validation (4 Points) ---")
    for name, screen_pos in val_points.items():
        print(f">>> Look at validation dot '{name}'. Click window & press SPACE/ENTER.")
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            display = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
            dh, dw, _ = display.shape

            dot_x = int(screen_pos[0] * dw / screen_w)
            dot_y = int(screen_pos[1] * dh / screen_h)
            cv2.circle(display, (dot_x, dot_y), 15, (255, 165, 0), -1)  # Orange/cyan dot
            cv2.putText(display, f"Validation: Look at {name} & press SPACE/ENTER",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 0), 2)
            cv2.imshow("Calibration", display)

            key = cv2.waitKey(10) & 0xFF
            if key == 27:
                break
            elif key in (32, 13, 10, ord(' '), ord('\r'), ord('\n')):
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    lx, ly, _ = get_eye_ratio(lm, LEFT_EYE, LEFT_IRIS, w, h)
                    rx, ry, _ = get_eye_ratio(lm, RIGHT_EYE, RIGHT_IRIS, w, h)
                    if lx is not None and rx is not None:
                        val_ratio = np.array([(lx+rx)/2, (ly+ry)/2, 1.0])
                        pred_pos = np.dot(val_ratio, A)
                        err = float(np.linalg.norm(np.array(screen_pos) - pred_pos))
                        val_errors.append(err)
                        print(f"  Target '{name}': Target={screen_pos}, Pred=({int(pred_pos[0])}, {int(pred_pos[1])}), Error={err:.1f}px")
                        break
                    else:
                        print("Iris not clear, please hold still and press SPACE/ENTER.")
                else:
                    print("No face detected, please look directly at the camera.")

    if val_errors:
        mean_err = np.mean(val_errors)
        print("\n" + "=" * 44)
        print("🎯 ACCURACY BENCHMARK RESULT:")
        print(f"   Mean Absolute Error: {mean_err:.1f} pixels")
        print(f"   Relative Error:      {(mean_err / screen_w) * 100:.1f}% of screen width")
        print("=" * 44 + "\n")

    cv2.destroyAllWindows()

    
    
    # Open CSV file
    with open("eye_gaze_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["left_x", "left_y", "right_x", "right_y", "gaze_x", "gaze_y"])

        mouse_control = False  # toggle flag
        gaze_history = deque(maxlen=7)  # keeps last 7 points

        # Live gaze tracking
        cv2.namedWindow("Gaze Tracking", cv2.WINDOW_AUTOSIZE)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark

                # Draw mesh grid
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.multi_face_landmarks[0],
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                lx, ly, left_iris = get_eye_ratio(lm, LEFT_EYE, LEFT_IRIS, w, h)
                rx, ry, right_iris = get_eye_ratio(lm, RIGHT_EYE, RIGHT_IRIS, w, h)

                if lx is None or rx is None:
                    continue

                raw_gaze_ratio = np.array([[(lx+rx)/2, (ly+ry)/2, 1]])
                raw_gaze_point = raw_gaze_ratio @ A
                    
                gaze_history.append(raw_gaze_point[0])

                # Compute the smoothed gaze point
                smoothed_gaze_point = np.mean(gaze_history, axis=0)
                gx, gy = int(smoothed_gaze_point[0]), int(smoothed_gaze_point[1])

                # Save to CSV
                writer.writerow([lx, ly, rx, ry, gx, gy])

                # === Mouse control toggle ===
                if mouse_control:
                    pyautogui.moveTo(gx, gy)

                # === Overlay gaze point and info ===
                # Rescale gaze to webcam frame coords
                gx_cam = int(gx * w / screen_w)
                gy_cam = int(gy * h / screen_h)

                # Red dot (camera coords → matches lines)
                cv2.circle(frame, (gx_cam, gy_cam), 10, (0,0,255), -1)

                # Lines from both iris centers → gaze point
                cv2.line(frame, left_iris, (gx_cam, gy_cam), (255,0,0), 2)
                cv2.line(frame, right_iris, (gx_cam, gy_cam), (255,0,0), 2)

                # Text still shows screen coords (useful for CSV/mouse)
                cv2.putText(frame, f"Gaze: ({gx},{gy})", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Mouse: {'ON' if mouse_control else 'OFF'}",
                            (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)


            display = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Gaze Tracking", display)
            key = cv2.waitKey(5) & 0xFF
            if key == 27:  # ESC = exit
                break
            elif key == ord('m'):  # toggle mouse control
                mouse_control = not mouse_control
                print("Mouse control:", "ON ✅" if mouse_control else "OFF ❌")

cap.release()
cv2.destroyAllWindows()
print("Gaze data saved to eye_gaze_data.csv")
