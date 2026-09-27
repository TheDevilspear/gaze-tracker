# Multimodal Eye Tracking, Behavioral Analysis & rPPG Suite

A modular, real-time computer vision and multimodal behavioral analysis application optimized for Linux/WSL. It unifies screen-calibrated gaze tracking, head pose estimation (solvePnP), Eye Aspect Ratio (EAR) blink detection, remote photoplethysmography (rPPG heart rate), live speech transcription (Whisper ASR), and asynchronous MongoDB batch logging.

---

## 🚀 Quick Start & How to Run

### 1. Prerequisites & Environment Setup

Activate the pre-configured virtual environment:
```bash
source .venv/bin/activate
```
*(If setting up on a new machine, install dependencies with `pip install -r requirements.txt` and ensure `portaudio` is installed).*

### 2. WSL / USB Camera Attachment (If on WSL2)

In an **Administrator PowerShell** on Windows:
```powershell
usbipd list
usbipd attach --wsl --busid 1-3
```

In your WSL terminal:
```bash
sudo chmod 666 /dev/video0
```

### 3. Step 1: Calibration & Accuracy Benchmark
Run the dedicated calibration tool to map your eye gaze to screen pixels and get an immediate quantitative accuracy score:
```bash
python calibrate.py
```
1. **5-Point Calibration**: Look at each red dot on the screen, click the window to focus, and press **SPACE** or **ENTER**.
2. **4-Point Validation Benchmark**: Look at each orange dot and press **SPACE** or **ENTER**.
3. The benchmark will output your **Mean Absolute Error (in pixels)** and save `calibration_matrix.npy`.

### 4. Step 2: Run Full Live Session
Launch the unified tracker:
```bash
python main.py
```

#### Optional CLI Flags:
- `--no-audio`: Disable microphone recording and Whisper transcription.
- `--no-db`: Run in offline mode without MongoDB logging.
- `--no-rppg`: Disable heart rate tracking.
- `--source <path_or_index>`: Specify a video file or different camera index (default: `0`).

Press **'q'** inside the video window to stop and automatically export session CSVs, heatmaps, and gaze paths.

---

## 🏗 High-Level Architectural Breakdown

```text
eyetrackingproject/
├── config.py              # Centralized configuration (camera, screen, landmarks, AOIs)
├── calibrate.py           # Standalone calibration & accuracy evaluation script
├── main.py                # Unified live tracker orchestrator
├── core/
│   ├── camera.py          # CameraStream (V4L2 + MJPG buffer management)
│   ├── gaze.py            # GazeCalibrator, eye ratios, regression math & validation
│   ├── head_pose.py       # EAR (safe division), solvePnP pose, pupil diameter, AOIs
│   ├── attention.py       # Off-screen attention tracker (duration, frequency, max stay)
│   └── rppg.py            # Real-time POS pulse extraction (rPPG heart rate)
└── services/
    ├── audio.py           # AudioTranscriber (lazy-loaded Whisper ASR worker)
    └── storage.py         # DBWriterThread (async MongoDB batching) & CSV/heatmap exports
```

---

## 🔍 Module Breakdown

### 1. Core Tracking Engine (`core/`)

- **[`core/gaze.py`](core/gaze.py)**:
  - Computes horizontal and vertical iris position ratios relative to eye corner boundaries using MediaPipe FaceMesh landmarks.
  - Fits a linear least-squares transformation matrix ($[\text{ratio}_x, \text{ratio}_y, 1] \times A = [\text{screen}_x, \text{screen}_y]$).
  - Evaluates Euclidean pixel error against ground truth screen targets.
  - Persists and loads the calibration matrix via `calibration_matrix.npy`.

- **[`core/head_pose.py`](core/head_pose.py)**:
  - **Head Pose Estimation**: Uses 6 standard 3D facial feature coordinates and Perspective-n-Point (`cv2.solvePnP` + `decomposeProjectionMatrix`) to compute Pitch, Yaw, and Roll.
  - **Eye Aspect Ratio (EAR)**: Computes vertical-to-horizontal eyelid distances with zero-division protection to detect blinks.
  - **Pupil Diameter**: Measures Euclidean pupil span across left and right irises.
- **[`core/attention.py`](core/attention.py)**:
  - **Off-Screen Attention Tracking**: Monitors when the user looks away from the screen, looks down at a phone/keyboard, or when the face leaves the frame.
  - **Analytics**: Calculates live off-screen event counts, highest continuous duration away (`max_duration`), average event duration, and event frequency (mean interval between distractions).

- **[`core/rppg.py`](core/rppg.py)**:
  - Real-time remote photoplethysmography (rPPG) extracted from the [`rPPG-Toolbox`](rPPG-Toolbox).
  - Isolates a clean skin Region of Interest (ROI) on the forehead using facial landmarks (`[109, 10, 338, 297, 9, 67]`), avoiding hair and eyebrows.
  - Implements the **Plane-Orthogonal-to-Skin (POS)** algorithm to extract blood volume pulse (BVP) from RGB channel variations.
  - Uses a 1024-point zero-padded FFT and Butterworth bandpass filter ($45–160\text{ BPM}$) to estimate heart rate in real time without heavy dependencies or GPU requirements.

- **[`core/camera.py`](core/camera.py)**:
  - Manages video capture using `cv2.CAP_V4L2` and Motion-JPEG (`MJPG`) encoding at 320×240.
  - Prevents USB/IP packet buffer truncation (resolving black screen / green screen driver issues under WSLg).

---

### 2. Background Services (`services/`)

- **[`services/audio.py`](services/audio.py)**:
  - Preloads user-space PortAudio libraries under Linux/WSL.
  - Uses PyAudio to stream microphone input in 5-second segments via a dedicated thread.
  - Lazy-loads Hugging Face `transformers` Whisper ASR (`distil-whisper/distil-small.en`) to run transcription asynchronously without stalling video frames.

- **[`services/storage.py`](services/storage.py)**:
  - **`DBWriterThread`**: Non-blocking worker thread that flushes queued data in batches (every 1 second or 50 records) to MongoDB (`transcripts` and `numerical data` collections). Automatically switches to offline mode if MongoDB is unreachable.
  - **Artifact Exporters**: Generates `numerical_eye_data.csv`, blended attention heatmaps (`eye_attention_heatmap.png`), and continuous gaze trajectories (`gaze_path.png`).

---

### 3. Entry Points

- **[`calibrate.py`](calibrate.py)**: Standalone 5-point calibration + 4-point validation benchmark. Run this first to generate your `calibration_matrix.npy`.
- **[`main.py`](main.py)**: Main application orchestrating camera capture, calibrated gaze mapping, head pose, blink detection, rPPG heart rate, audio transcription, HUD overlays, and database logging.
