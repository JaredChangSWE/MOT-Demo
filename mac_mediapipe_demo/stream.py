"""Low-latency RTSP frame grabber.

``cv2.VideoCapture`` buffers frames internally; if the control loop reads
slower than the camera produces, that buffer fills and you end up tracking a
target that has already moved. This reader runs the capture in its own thread
and only ever keeps the *latest* frame, so ``read()`` returns near-real-time.
"""

from __future__ import annotations

import os
import threading
import time

# Force FFmpeg to pull RTSP over TCP with a low-latency, small buffer. UDP (the
# default) drops packets on Wi-Fi and stalls 0.5-1s waiting for the next
# keyframe; TCP retransmits and keeps the stream steady. Must be set before the
# first VideoCapture is created.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;500000",
)

import cv2  # noqa: E402  (import after setting the FFmpeg env)

from applog import get_logger

_log = get_logger("stream")

# A gap longer than this between decoded frames counts as a stall worth logging
# (this is what a viewer perceives as a 0.5-1s freeze).
_STALL_SECONDS = 0.4


class LatestFrameReader:
    def __init__(self, rtsp_url: str, name: str = "cam") -> None:
        self.name = name
        self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        # Ask the backend to keep its own buffer tiny too (best-effort).
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._latest_frame = None
        self._seq = 0                 # bumped on every decoded frame
        self._running = True
        self._lock = threading.Lock()
        # stall / health stats (read via stats()).
        self.stall_count = 0
        self.max_gap = 0.0
        self.read_failures = 0
        self._logged_open = False
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        _log.info(f"[{name}] opening {rtsp_url.rsplit('@', 1)[-1]} "
                  f"(opened={self.cap.isOpened()})")

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def _update(self) -> None:
        last_ok = time.monotonic()
        while self._running:
            ok, frame = self.cap.read()
            now = time.monotonic()
            if ok:
                gap = now - last_ok
                last_ok = now
                if not self._logged_open:
                    # First frame: startup buffering, not a stall.
                    self._logged_open = True
                    h, w = frame.shape[:2]
                    _log.info(f"[{self.name}] first frame {w}x{h}")
                elif gap > _STALL_SECONDS:
                    self.stall_count += 1
                    self.max_gap = max(self.max_gap, gap)
                    _log.warning(f"[{self.name}] stream stall: "
                                 f"{gap:.2f}s gap (stall #{self.stall_count})")
                with self._lock:
                    self._latest_frame = frame
                    self._seq += 1
            else:
                self.read_failures += 1
                if self.read_failures % 20 == 1:
                    _log.warning(f"[{self.name}] read() failed "
                                 f"(#{self.read_failures}) — reconnecting/backoff")
                time.sleep(0.1)

    def stats(self) -> dict:
        return {"name": self.name, "stalls": self.stall_count,
                "max_gap": self.max_gap, "read_failures": self.read_failures}

    def read(self):
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def latest(self) -> tuple:
        """Return (frame_copy_or_None, seq). ``seq`` lets a consumer skip work
        when the frame hasn't changed since it last looked."""
        with self._lock:
            if self._latest_frame is None:
                return None, self._seq
            return self._latest_frame.copy(), self._seq

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()
