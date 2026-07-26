"""File logging so runtime behavior can be diagnosed after the fact.

Each run writes a timestamped log under ``logs/`` (gitignored). Events (stream
open/stall, ONVIF connect, detector/tracker switches, target lock, PTZ
start/stop, watchdog, periodic profiling, exceptions) go to the file at DEBUG;
the console keeps only concise INFO lines so the terminal stays readable.

Usage:
    from applog import setup_logging, get_logger
    setup_logging()          # once, at startup
    log = get_logger(__name__)
    log.info("something happened")
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path(__file__).with_name("logs")
_configured = False
_log_path: Path | None = None


def setup_logging(level: int = logging.DEBUG) -> Path:
    """Configure root logging to a fresh timestamped file + console. Idempotent."""
    global _configured, _log_path
    if _configured:
        return _log_path  # type: ignore[return-value]

    _LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_path = _LOG_DIR / f"ptz_{stamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.FileHandler(_log_path, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(ch)

    # Silence very chatty third-party libraries (zeep/onvif emit thousands of
    # DEBUG lines) so the log stays about OUR program.
    for noisy in ("zeep", "urllib3", "requests", "PIL", "matplotlib",
                  "mediapipe", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    logging.getLogger(__name__).info(f"logging to {_log_path}")
    return _log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_path() -> Path | None:
    return _log_path
