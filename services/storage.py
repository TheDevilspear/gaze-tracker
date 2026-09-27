import os
import time
import queue
import logging
import threading
import cv2
import numpy as np
import pandas as pd
from pymongo import MongoClient
from config import (
    MONGO_URI, DB_NAME, COLLECTION_TRANSCRIPTS, COLLECTION_NUMERICAL,
    CSV_OUTPUT_FILE
)

class DBWriterThread(threading.Thread):
    """Batches MongoDB inserts asynchronously so video and audio loops never block."""
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
                logging.warning(f"DBWriterThread: MongoDB connection failed ({e}). Running in offline mode.")
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

            if batch and (len(batch) >= 50 or (time.time() - last_flush) > 1.0 or self.stop_event.is_set()):
                if self.db is not None:
                    num_docs = [d["data"] for d in batch if d.get("type") == "numerical"]
                    txt_docs = [d["data"] for d in batch if d.get("type") == "text"]
                    try:
                        if num_docs:
                            self.db[COLLECTION_NUMERICAL].insert_many(num_docs, ordered=False)
                        if txt_docs:
                            self.db[COLLECTION_TRANSCRIPTS].insert_many(txt_docs, ordered=False)
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

def export_numerical_csv(data_list, output_path=CSV_OUTPUT_FILE):
    """Saves numerical metrics list to CSV."""
    if not data_list:
        return
    try:
        df = pd.DataFrame(data_list)
        df.to_csv(output_path, index=False)
        logging.info(f"Numerical data exported to {output_path}")
    except Exception as e:
        logging.error(f"Failed to export CSV: {e}")

def export_attention_heatmap(accumulator, base_image, output_path="eye_attention_heatmap.png"):
    """Normalizes accumulation grid and blends JET colormap over the base image."""
    if accumulator.max() <= 0 or base_image is None:
        return
    norm = cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(np.uint8(norm), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(base_image, 0.5, heatmap_colored, 0.5, 0)
    cv2.imwrite(output_path, blended)
    logging.info(f"Heatmap saved to {output_path}")

def export_gaze_path(history, width, height, output_path="gaze_path.png"):
    """Draws connected gaze path over a black canvas."""
    if not history:
        return
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(1, len(history)):
        p1 = history[i-1]
        p2 = history[i]
        if p1[0] >= 0 and p2[0] >= 0:
            cv2.line(canvas, p1, p2, (0, 255, 0), 2)
    cv2.imwrite(output_path, canvas)
    logging.info(f"Gaze path saved to {output_path}")
