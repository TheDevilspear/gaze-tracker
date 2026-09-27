import numpy as np

class OffScreenTracker:
    """
    Tracks off-screen attention and distraction events with temporal debouncing.
    Prevents false triggers from single-frame glitches or quick natural blinks.
    """
    def __init__(self, min_event_duration=0.6, debounce_frames=6):
        self.min_event_duration = min_event_duration
        self.debounce_frames = debounce_frames
        self.offscreen_streak = 0
        self.onscreen_streak = 0

        self.is_offscreen = False
        self.offscreen_start = None
        self.current_duration = 0.0
        self.max_duration = 0.0
        self.total_offscreen_time = 0.0

        self.event_durations = []
        self.event_start_times = []
        self.offscreen_count = 0

    def update(self, is_offscreen_now, current_time):
        """
        Updates tracker state with debouncing.
        is_offscreen_now: bool
        current_time: float (seconds)
        """
        if is_offscreen_now:
            self.offscreen_streak += 1
            self.onscreen_streak = 0
            if self.offscreen_streak >= self.debounce_frames:
                if not self.is_offscreen:
                    self.is_offscreen = True
                    # Backdate to start of streak
                    self.offscreen_start = current_time - (self.debounce_frames * 0.033)
                    self.current_duration = 0.0
        else:
            self.onscreen_streak += 1
            self.offscreen_streak = 0
            if self.onscreen_streak >= 3:
                if self.is_offscreen:
                    event_duration = current_time - self.offscreen_start
                    if event_duration >= self.min_event_duration:
                        self.event_durations.append(event_duration)
                        self.event_start_times.append(self.offscreen_start)
                        self.total_offscreen_time += event_duration
                        self.offscreen_count += 1
                        if event_duration > self.max_duration:
                            self.max_duration = event_duration

                    self.is_offscreen = False
                    self.offscreen_start = None
                    self.current_duration = 0.0

        if self.is_offscreen and self.offscreen_start is not None:
            self.current_duration = max(0.0, current_time - self.offscreen_start)
            if self.current_duration > self.max_duration:
                self.max_duration = self.current_duration

        return self.get_stats(current_time)

    def get_stats(self, session_time=None):
        """Computes current summary metrics."""
        avg_dur = float(np.mean(self.event_durations)) if self.event_durations else 0.0

        if self.offscreen_count > 0 and session_time is not None and session_time > 0:
            avg_interval = session_time / float(self.offscreen_count)
        else:
            avg_interval = 0.0

        return {
            "is_offscreen": self.is_offscreen,
            "offscreen_count": self.offscreen_count,
            "current_offscreen_sec": round(self.current_duration, 1),
            "max_offscreen_sec": round(self.max_duration, 1),
            "avg_offscreen_sec": round(avg_dur, 1),
            "avg_interval_sec": round(avg_interval, 1),
            "total_offscreen_sec": round(self.total_offscreen_time, 1)
        }
