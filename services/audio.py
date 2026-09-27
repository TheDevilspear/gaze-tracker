import os
import time
import wave
import queue
import ctypes
import logging
import threading
from config import CHUNK, FORMAT_CHANNELS, RATE, RECORD_SECONDS, ASR_MODEL_NAME

# Preload Homebrew PortAudio library if on WSL / Linux
for brew_lib in ["/home/linuxbrew/.linuxbrew/lib/libportaudio.so.2", "/home/linuxbrew/.linuxbrew/lib/libportaudio.so"]:
    if os.path.exists(brew_lib):
        try:
            ctypes.CDLL(brew_lib, mode=ctypes.RTLD_GLOBAL)
            break
        except Exception:
            pass

try:
    import pyaudio
except ImportError:
    pyaudio = None

class AudioTranscriber(threading.Thread):
    """Continuously records microphone audio and transcribes in a background worker."""
    def __init__(self, db_queue, stop_event, model_name=ASR_MODEL_NAME):
        super().__init__(daemon=True)
        self.db_queue = db_queue
        self.stop_event = stop_event
        self.chunk_queue = queue.Queue(maxsize=10)
        self.asr_pipeline = None

        try:
            import torch
            from transformers import pipeline
            torch.set_num_threads(2)
            self.asr_pipeline = pipeline("automatic-speech-recognition", model=model_name)
        except Exception as e:
            logging.warning(f"AudioTranscriber: ASR model load failed ({e}). Speech-to-text disabled.")

        self.p = None
        self.stream = None
        if pyaudio is not None:
            try:
                self.p = pyaudio.PyAudio()
                self.stream = self.p.open(
                    format=pyaudio.paInt16,
                    channels=FORMAT_CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                )
            except Exception as e:
                logging.warning(f"AudioTranscriber: Mic initialization failed ({e}). Audio disabled.")
                self.stream = None

        self.worker_thread = threading.Thread(target=self._transcription_worker, daemon=True)

    def _transcription_worker(self):
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
                wf.setnchannels(FORMAT_CHANNELS)
                wf.setsampwidth(2)
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
        logging.info("Audio recording thread started.")
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
                logging.debug(f"Audio read warning: {e}")

        # Cleanup
        try:
            self.stream.stop_stream()
            self.stream.close()
            if self.p:
                self.p.terminate()
        except Exception:
            pass
        logging.info("AudioTranscriber stopped.")
