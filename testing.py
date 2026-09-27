import sys
import os
import ctypes

# Automatically preload Homebrew's PortAudio library if on Linux/WSL
for brew_lib in ["/home/linuxbrew/.linuxbrew/lib/libportaudio.so.2", "/home/linuxbrew/.linuxbrew/lib/libportaudio.so"]:
    if os.path.exists(brew_lib):
        try:
            ctypes.CDLL(brew_lib, mode=ctypes.RTLD_GLOBAL)
            break
        except Exception:
            pass

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from math import hypot
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError
from pydub import AudioSegment
from pydub.playback import play
from transformers import pipeline
import threading
import queue
import pyaudio
import wave
from dotenv import load_dotenv
import logging

# --- 1. Basic Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 2. Load Environment Variables ---
load_dotenv() 



FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg") 

# Set this environment variable so that pydub can find the executable
os.environ["FFMPEG_BINARY"] = FFMPEG_PATH

# --- Configuration for MongoDB ---
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "EyetrackingAI"
COLLECTION_NAME_TEXT = "transcripts"
COLLECTION_NAME_NUMERICAL = "numerical data"


# --- Constants for Eye Tracking ---
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_IRIS_INDICES = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]
BLINK_THRESHOLD = 0.25

# --- Sample AOIs ---
SAMPLE_AOIS = {
    "Headline": (50, 50, 300, 100),
    "ProductImage": (350, 150, 600, 400),
    "CallToAction": (700, 500, 900, 550)
}

# --- Audio Recording Constants ---
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "live_transcript_chunk.wav"

# --- Asynchronous Database Writer Thread ---
class DBWriterThread(threading.Thread):
    """Batches MongoDB inserts asynchronously so video and audio threads never block."""
    def __init__(self, db_queue, stop_event, mongo_uri=MONGO_URI):
        super().__init__(daemon=True)
        self.db_queue = db_queue
        self.stop_event = stop_event
        self.client = None
        self.db = None
        if mongo_uri:
            try:
                self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
                self.client.admin.command('ping')
                self.db = self.client[DB_NAME]
                logging.info("DBWriterThread: Successfully connected to MongoDB.")
            except Exception as e:
                logging.warning(f"DBWriterThread: MongoDB connection failed ({e}). Offline mode active.")
                self.client = None

    def run(self):
        batch = []
        last_flush = time.time()
        while not self.stop_event.is_set() or not self.db_queue.empty():
            try:
                item = self.db_queue.get(timeout=0.2)
                batch.append(item)
            except queue.Empty:
                pass

            # Flush batch every 1.0s or when reaching 50 items
            if batch and (len(batch) >= 50 or (time.time() - last_flush) > 1.0 or self.stop_event.is_set()):
                if self.db is not None:
                    num_docs = [d["data"] for d in batch if d.get("type") == "numerical"]
                    txt_docs = [d["data"] for d in batch if d.get("type") == "text"]
                    try:
                        if num_docs:
                            self.db[COLLECTION_NAME_NUMERICAL].insert_many(num_docs, ordered=False)
                        if txt_docs:
                            self.db[COLLECTION_NAME_TEXT].insert_many(txt_docs, ordered=False)
                    except Exception as e:
                        logging.error(f"DBWriterThread batch write error: {e}")
                batch.clear()
                last_flush = time.time()

        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        logging.info("DBWriterThread stopped.")

# --- Decoupled Audio Transcriber ---
class AudioTranscriber(threading.Thread):
    """Continuously records audio without dropping frames while running inference in worker."""
    def __init__(self, db_queue, stop_event):
        super().__init__(daemon=True)
        self.db_queue = db_queue
        self.stop_event = stop_event
        self.chunk_queue = queue.Queue(maxsize=10)
        self.asr_pipeline = None

        try:
            torch.set_num_threads(2)
            self.asr_pipeline = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en")
        except Exception as e:
            logging.warning(f"AudioTranscriber: Could not load ASR model ({e}).")

        self.p = None
        self.stream = None
        try:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=FORMAT,
                                     channels=CHANNELS,
                                     rate=RATE,
                                     input=True,
                                     frames_per_buffer=CHUNK)
        except Exception as e:
            logging.warning(f"AudioTranscriber: Mic initialization failed ({e}). Audio disabled.")

        self.worker_thread = threading.Thread(target=self._transcription_worker, daemon=True)

    def _transcription_worker(self):
        """Processes audio chunks in background so recording stream never overflows."""
        while not self.stop_event.is_set():
            try:
                raw_bytes = self.chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self.asr_pipeline is None:
                continue

            temp_filename = f"live_chunk_{int(time.time()*1000)}.wav"
            try:
                wf = wave.open(temp_filename, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # paInt16 = 2 bytes
                wf.setframerate(RATE)
                wf.writeframes(raw_bytes)
                wf.close()

                transcription = self.asr_pipeline(temp_filename)
                text = transcription.get('text', '').strip()
                if text:
                    print(f"\n[Transcribed]: {text}")
                    doc = {"timestamp": time.time(), "text": text}
                    try:
                        self.db_queue.put_nowait({"type": "text", "data": doc})
                    except queue.Full:
                        pass
            except Exception as e:
                logging.error(f"Transcription error: {e}")
            finally:
                if os.path.exists(temp_filename):
                    try:
                        os.remove(temp_filename)
                    except OSError:
                        pass

    def run(self):
        if self.stream is None:
            return

        self.worker_thread.start()
        print("Real-time audio recording started.")
        frames = []
        last_chunk_time = time.time()

        while not self.stop_event.is_set():
            try:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

                if (time.time() - last_chunk_time) >= RECORD_SECONDS:
                    chunk_data = b''.join(frames)
                    frames = []
                    last_chunk_time = time.time()
                    try:
                        self.chunk_queue.put_nowait(chunk_data)
                    except queue.Full:
                        logging.warning("Audio queue full, dropping oldest audio segment.")
            except IOError as e:
                logging.debug(f"Audio stream error: {e}")

        # Cleanup
        try:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
        except Exception:
            pass
        logging.info("AudioTranscriber stopped.")

def get_landmark_coords(landmarks, indices, frame_width, frame_height):
    return [(int(landmarks[idx].x * frame_width), int(landmarks[idx].y * frame_height)) for idx in indices]

def calculate_ear(landmarks):
    def euclidean_distance(point1, point2):
        return hypot(point2[0] - point1[0], point2[1] - point1[1])
    
    left_eye_points = get_landmark_coords(landmarks, LEFT_EYE_INDICES, 1, 1)
    right_eye_points = get_landmark_coords(landmarks, RIGHT_EYE_INDICES, 1, 1)
    
    denom_left = 2 * euclidean_distance(left_eye_points[0], left_eye_points[3])
    denom_right = 2 * euclidean_distance(right_eye_points[0], right_eye_points[3])
    
    left_ear = (euclidean_distance(left_eye_points[1], left_eye_points[5]) + euclidean_distance(left_eye_points[2], left_eye_points[4])) / denom_left if denom_left > 0 else 0.0
    right_ear = (euclidean_distance(right_eye_points[1], right_eye_points[5]) + euclidean_distance(right_eye_points[2], right_eye_points[4])) / denom_right if denom_right > 0 else 0.0
    
    return (left_ear + right_ear) / 2.0

def get_head_pose_angles(landmarks, frame_width, frame_height):
    # (Existing head pose function, no changes needed)
    image_points = np.array([
        get_landmark_coords(landmarks, [33], frame_width, frame_height)[0],
        get_landmark_coords(landmarks, [263], frame_width, frame_height)[0],
        get_landmark_coords(landmarks, [1], frame_width, frame_height)[0],
        get_landmark_coords(landmarks, [61], frame_width, frame_height)[0],
        get_landmark_coords(landmarks, [291], frame_width, frame_height)[0],
        get_landmark_coords(landmarks, [199], frame_width, frame_height)[0]
    ], dtype="double")
    
    model_points = np.array([
        (0.0, 0.0, 0.0), 
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])
    
    center = (frame_height/2, frame_width/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if success:
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        head_pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(head_pose_matrix, camera_matrix, dist_coeffs)
        
        yaw = euler_angles[1, 0]
        pitch = euler_angles[0, 0]
        roll = euler_angles[2, 0]
        
        return yaw, pitch, roll
    return None, None, None

def get_aoi_metrics(gaze_x, gaze_y, aoi_regions):
    current_aoi = None
    for aoi_name, (x1, y1, x2, y2) in aoi_regions.items():
        if x1 <= gaze_x <= x2 and y1 <= gaze_y <= y2:
            current_aoi = aoi_name
            break
    return current_aoi

def process_video(source=0):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Check video source
    if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(source))
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            for _ in range(5):
                cap.read()
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Could not open video source '{source}'.")
        if source == 0 or source == "0":
            print("In WSL, attach your webcam via USB/IP ('usbipd wsl attach') or run with a video file: python testing.py --source sample.mp4")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cv2.namedWindow('Live Video Feed', cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow('Live Video Feed', 100, 100)

    stop_event = threading.Event()
    db_queue = queue.Queue(maxsize=2000)

    # Start asynchronous DB writer and audio transcription worker
    db_writer = DBWriterThread(db_queue, stop_event)
    db_writer.start()

    audio_thread = AudioTranscriber(db_queue, stop_event)
    audio_thread.start()

    numerical_data = []
    gaze_history = []
    heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)

    blink_status = 0
    blink_counter = 0
    start_time = time.time()
    image = None

    print("Video processing started. Press 'q' in the video window to stop.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
        
        current_time = time.time() - start_time
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        gaze_x, gaze_y = -1, -1
        pupil_diameter = -1
        head_yaw, head_pitch, head_roll = None, None, None
        current_aoi = None
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # Ensure all 478 landmarks are present before proceeding
                if len(landmarks) < 478:
                    continue
                
                left_iris_center_coords = get_landmark_coords(landmarks, [473], frame_width, frame_height)
                right_iris_center_coords = get_landmark_coords(landmarks, [468], frame_width, frame_height)
                
                if left_iris_center_coords and right_iris_center_coords:
                    left_iris_center = left_iris_center_coords[0]
                    right_iris_center = right_iris_center_coords[0]
                    gaze_x = int((left_iris_center[0] + right_iris_center[0]) / 2)
                    gaze_y = int((left_iris_center[1] + right_iris_center[1]) / 2)

                    left_pupil_points = get_landmark_coords(landmarks, [474, 476], 1, 1)
                    right_pupil_points = get_landmark_coords(landmarks, [469, 471], 1, 1)
                    if left_pupil_points and right_pupil_points:
                        left_pupil_diameter = hypot(left_pupil_points[0][0] - left_pupil_points[1][0], left_pupil_points[0][1] - left_pupil_points[1][1])
                        right_pupil_diameter = hypot(right_pupil_points[0][0] - right_pupil_points[1][0], right_pupil_points[0][1] - right_pupil_points[1][1])
                        pupil_diameter = (left_pupil_diameter + right_pupil_diameter) / 2
                    
                    ear = calculate_ear(landmarks)
                    blink_status = 1 if ear < BLINK_THRESHOLD else 0
                    if blink_status == 1:
                        blink_counter += 1
                    
                    head_yaw, head_pitch, head_roll = get_head_pose_angles(landmarks, frame_width, frame_height)
                    current_aoi = get_aoi_metrics(gaze_x, gaze_y, SAMPLE_AOIS)

                    cv2.circle(image, (gaze_x, gaze_y), 5, (0, 255, 0), -1)
                    cv2.putText(image, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Blinks: {blink_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Pupil: {pupil_diameter:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if head_yaw:
                        cv2.putText(image, f"Head Pose: YAW={head_yaw:.1f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if current_aoi:
                        cv2.putText(image, f"AOI: {current_aoi}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.circle(heatmap_accumulator, (gaze_x, gaze_y), 20, 1, -1)
                    gaze_history.append((gaze_x, gaze_y))

                    doc = {
                        "timestamp": current_time,
                        "gaze_x": gaze_x,
                        "gaze_y": gaze_y,
                        "pupil_diameter": pupil_diameter,
                        "blink_status": blink_status,
                        "head_yaw": head_yaw,
                        "head_pitch": head_pitch,
                        "head_roll": head_roll,
                        "aoi": current_aoi
                    }
                    numerical_data.append(doc)
                    try:
                        db_queue.put_nowait({"type": "numerical", "data": doc})
                    except queue.Full:
                        pass

        display_img = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('Live Video Feed', display_img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Clean up and join background worker threads
    cap.release()
    cv2.destroyAllWindows()
    stop_event.set()
    audio_thread.join(timeout=2.0)
    db_writer.join(timeout=3.0)

    if numerical_data:
        try:
            df = pd.DataFrame(numerical_data)
            df.to_csv("numerical_eye_data.csv", index=False)
            print("Numerical data successfully saved to numerical_eye_data.csv")
        except Exception as e:
            logging.error(f"Failed to export numerical data to CSV: {e}")
    
    if heatmap_accumulator.max() > 0 and image is not None:
        heatmap_normalized = cv2.normalize(heatmap_accumulator, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_normalized = np.uint8(heatmap_normalized)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        heatmap_blended = cv2.addWeighted(image, 0.5, heatmap_colored, 0.5, 0)
        cv2.imwrite("eye_attention_heatmap.png", heatmap_blended)
        print("Heatmap saved to eye_attention_heatmap.png")
    
    gaze_map = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    if gaze_history:
        for i in range(1, len(gaze_history)):
            cv2.line(gaze_map, gaze_history[i-1], gaze_history[i], (0, 255, 0), 2)
    cv2.imwrite("gaze_path.png", gaze_map)
    print("Gaze path map saved to gaze_path.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live Eye Tracking & Speech Analysis")
    parser.add_argument("--source", default=0, help="Camera index (e.g. 0) or path to video file")
    args = parser.parse_args()

    # Convert numeric string to integer if applicable
    src = int(args.source) if str(args.source).isdigit() else args.source
    process_video(source=src)