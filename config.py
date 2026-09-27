import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- Display & Screen Settings ---
DEFAULT_SCREEN_WIDTH = 1366
DEFAULT_SCREEN_HEIGHT = 768

try:
    import pyautogui
    SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
except Exception:
    SCREEN_WIDTH, SCREEN_HEIGHT = DEFAULT_SCREEN_WIDTH, DEFAULT_SCREEN_HEIGHT

# --- Camera Configuration (WSL / USB/IP Optimized) ---
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))
FRAME_WIDTH = 320   # 320x240 fits reliably within USB/IP ISO packet limits
FRAME_HEIGHT = 240
CODEC = "MJPG"
WARMUP_FRAMES = 5

# --- MediaPipe Landmark Indices ---
# Face mesh eye corners & contours
LEFT_EYE_RATIO_INDICES = [33, 133, 159, 145]     # [left_corner, right_corner, top, bottom]
RIGHT_EYE_RATIO_INDICES = [362, 263, 386, 374]   # [left_corner, right_corner, top, bottom]

LEFT_IRIS_INDICES = [468, 469, 470, 471]
RIGHT_IRIS_INDICES = [473, 474, 475, 476]

# Comprehensive indices for EAR and Pupil calculation
LEFT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_PUPIL_INDICES = [474, 476]
RIGHT_PUPIL_INDICES = [469, 471]

BLINK_THRESHOLD = 0.25

# --- Areas of Interest (AOIs) ---
SAMPLE_AOIS = {
    "Headline": (50, 50, 300, 100),
    "ProductImage": (350, 150, 600, 400),
    "CallToAction": (700, 500, 900, 550)
}

# --- Audio Configuration ---
CHUNK = 1024
FORMAT_CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
ASR_MODEL_NAME = "distil-whisper/distil-small.en"

# --- Database & Storage Configuration ---
MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = "EyetrackingAI"
COLLECTION_TRANSCRIPTS = "transcripts"
COLLECTION_NUMERICAL = "numerical data"

CSV_OUTPUT_FILE = "numerical_eye_data.csv"
CALIBRATION_FILE = "calibration_matrix.npy"
