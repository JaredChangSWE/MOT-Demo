"""Dedicated asynchronous worker thread for ONVIF PTZ commands.

Isolates ONVIF SOAP network I/O from the MediaPipe detection thread so that camera
network latency or HTTP request stalls never delay AI inference or frame decoding.
"""

from __future__ import annotations

import threading
import time

from applog import get_logger
from ptz import PTZController

_log = get_logger("ptz_worker")


class PTZCommandWorker(threading.Thread):
    def __init__(self, ptz: PTZController, min_interval: float = 0.04) -> None:
        super().__init__(daemon=True)
        self.ptz = ptz
        self.min_interval = min_interval  # Cap ONVIF HTTP commands to ~25 req/s to prevent socket buffer lag
        self._running = True
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)

        # Pending command state
        self._target_move: tuple[float, float, float] | None = None  # (pan, tilt, timeout)
        self._target_stop: bool = False
        self._last_exec_t = 0.0

    def send_move(self, pan: float, tilt: float, timeout: float) -> None:
        """Queue a continuous move command (non-blocking, returns instantly)."""
        with self._cond:
            self._target_move = (pan, tilt, timeout)
            self._target_stop = False
            self._cond.notify()

    def send_stop(self, force: bool = True) -> None:
        """Queue an immediate stop command (non-blocking, returns instantly)."""
        with self._cond:
            self._target_stop = True
            self._target_move = None
            self._cond.notify()

    def run(self) -> None:
        _log.info("PTZCommandWorker thread started")
        is_stopped = True

        while self._running:
            move_cmd = None
            stop_cmd = False

            with self._cond:
                # Wait until there is a command or thread is stopping
                while self._running and self._target_move is None and not self._target_stop:
                    self._cond.wait(timeout=0.05)

                if not self._running:
                    break

                if self._target_stop:
                    stop_cmd = True
                    self._target_stop = False
                elif self._target_move is not None:
                    move_cmd = self._target_move
                    self._target_move = None

            now = time.time()
            if stop_cmd:
                _log.debug("executing PTZ stop (immediate)")
                self.ptz.stop(force=True)
                is_stopped = True
                self._last_exec_t = time.time()
            elif move_cmd is not None:
                # Pacing check: prevent flooding camera socket buffer
                elapsed = now - self._last_exec_t
                if elapsed < self.min_interval:
                    time.sleep(self.min_interval - elapsed)

                pan, tilt, timeout = move_cmd
                _log.debug(f"executing PTZ move pan={pan:+.2f} tilt={tilt:+.2f}")
                self.ptz.move(pan, tilt, timeout)
                is_stopped = False
                self._last_exec_t = time.time()

        # Ensure motor stops when worker thread terminates
        try:
            self.ptz.stop(force=True)
        except Exception:
            pass
        _log.info("PTZCommandWorker thread stopped")

    def stop_and_join(self) -> None:
        _log.info("stopping PTZCommandWorker")
        with self._cond:
            self._running = False
            self._cond.notify_all()
        self.join(timeout=1.0)
