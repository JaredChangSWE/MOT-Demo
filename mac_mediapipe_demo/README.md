# Mac MediaPipe Auto-Tracker

Real-camera PTZ auto-tracking demo that runs locally on macOS. It finds an
ONVIF PTZ camera (e.g. TP-Link Tapo) on your LAN, pulls its RTSP video, detects
people with **MediaPipe Face detection** (or full-body Pose), tracks them with a
**selectable multi-object tracker** (SORT / ByteTrack / DeepSORT / OC-SORT /
BoT-SORT), and steers the camera to keep the followed person centered — all
tunable live from an on-screen control panel.

Detection defaults to **face** (BlazeFace: fast — ~2–3 ms on a downscaled frame
— and accurate for who's in view); switch to full-body **pose** live from the
panel. RTSP is pulled over **TCP** and detection runs on a downscaled copy to
avoid stutter on Wi-Fi.

It ports the sibling `ptz_tracking/` simulation's approach (pluggable tracking
algorithms + live parameter panel) onto a **physical camera** over ONVIF.

This target is a **dual-lens device**: `stream1` = camera 1, `stream6` =
camera 2. Everything lives in **one window** — both lens views side by side with
the tuning sliders on top — and auto-tracking runs on one lens (default
camera 2 / `stream6`; set `TAPO_TRACK_CAM=1` to switch).

## How it works

```
WS-Discovery -> pick camera -> ONVIF PTZ + two RTSP lenses (threaded, latest-frame)
   -> MediaPipe multi-pose detector (person bboxes)
   -> pluggable MOT tracker (stable IDs)   [reused from ptz_tracking.trackers]
   -> target selection (lock onto one, lost-grace re-acquire)
   -> eased velocity controller (SmoothDamp + engage hysteresis + lead)
   -> ONVIF ContinuousMove
```

| Module             | Responsibility                                              |
| ------------------ | ---------------------------------------------------------- |
| `discovery.py`     | ONVIF WS-Discovery — finds camera IP/port, stdlib only     |
| `stream.py`        | Threaded RTSP reader that only keeps the newest frame      |
| `ptz.py`           | ONVIF PTZ wrapper (`move(pan, tilt)` / `stop()`)           |
| `face_detector.py` | MediaPipe face detection → `Detection` bboxes (default)     |
| `pose_detector.py` | MediaPipe multi-pose → person `Detection` bboxes           |
| `detector.py`      | Factory: pick face or pose from the live params            |
| `params.py`        | All live-tunable parameters (one shared object)            |
| `target_select.py` | Which track to follow (lock + lost-grace)                  |
| `rt_controller.py` | Eased velocity control (SmoothDamp + hysteresis + lead)    |
| `hud.py`           | Tracking overlay: boxes, IDs, trails, status              |
| `controls_rt.py`   | On-screen "PTZ Controls" trackbar panel                    |
| `config.py`        | Credentials + device/stream settings (from env/`.env`)     |
| `main.py`          | Orchestration (dual view, tracking, control loop)          |

The five tracking algorithms are **reused verbatim** from `ptz_tracking/trackers`
(imported from the sibling package) — the MediaPipe detector produces the same
`Detection` bbox contract the simulator's trackers already consume.

## Setup

```bash
# From the repo root, reuse the existing venv (Python 3.13):
.venv/bin/pip install -r mac_mediapipe_demo/requirements.txt

# Provide camera credentials (the account you set in the Tapo/ONVIF app):
cp mac_mediapipe_demo/.env.example mac_mediapipe_demo/.env
# then edit .env  ->  TAPO_USER / TAPO_PASS
```

On first run the needed MediaPipe model is downloaded automatically and cached
(gitignored): `blaze_face_short_range.tflite` for face mode, or
`pose_landmarker_lite.task` for pose mode.

> **MediaPipe note:** recent wheels (0.10.x, incl. the Python 3.13 build)
> ship only the new **Tasks API**, so this uses `vision.PoseLandmarker`
> rather than the legacy `mp.solutions.pose`.

> **Tapo note:** create a dedicated *Camera Account* in the Tapo app
> (Advanced Settings → Camera Account). That username/password is what ONVIF
> and RTSP use — it is **not** your TP-Link cloud login.

## Run

```bash
cd mac_mediapipe_demo

.venv/bin/python main.py              # discover, pick, and track
.venv/bin/python main.py --list-only  # just list ONVIF cameras and exit
.venv/bin/python main.py --no-display # headless, no preview window
```

Press `q` in the preview window (or `Ctrl-C`) to stop — the motors are halted
and resources released on exit.

### First-time direction calibration (recommended)

ONVIF's pan/tilt sign isn't the same on every camera. If yours steers the wrong
way it will chase the target off-screen and spin. Run the guided calibrator once
— it nudges the camera while showing the live view and writes the right invert
flags to `.env`:

```bash
.venv/bin/python calibrate.py
```

You can also set them by hand in `.env`: `TAPO_INVERT_PAN=1` / `TAPO_INVERT_TILT=1`.

A safety **watchdog** also stops the camera if it ever moves for more than
`max_move_seconds` (3s) without settling, so a wrong sign can't spin forever.

## Logs (for diagnosis)

Every run writes a timestamped log to `logs/ptz_<timestamp>.log` (gitignored):
stream open/first-frame, **stream stalls** (gap > 0.4 s — the perceived
freezes), ONVIF connect, detector/tracker switches, target lock/lost, PTZ
start/stop and any ONVIF faults, a `profile` line every 5 s (worker fps +
per-stage ms), and shutdown stats (`stalls`, `max_gap`, `read_failures` per
lens). If something misbehaves, share that file — it says what happened.

ML inference/decisions run at **`inference_fps`** (default 5 Hz, slider-tunable);
the display still shows every camera frame. 5 fps is plenty for tracking and
keeps CPU low.

## Live control panel

Everything is in one **"PTZ Tracker"** window: the sliders sit on top and the
two camera views render side by side below (the tracked lens carries the HUD).
Sliders retune the pipeline on the next frame (no restart); a one-line legend for
every slider is also printed to the console. Highlights:

- **Tracker (0–4)** — switch algorithm live: SORT / ByteTrack / DeepSORT /
  OC-SORT / BoT-SORT (switching resets IDs). The HUD shows the active tracker and
  its measured per-frame latency.
- **Follow (0=big 1=center)** — follow the largest (nearest) person, or the one
  closest to frame center.
- **Detector (0=face 1=pose)** — switch detection method live.
- **Detect conf / scale width** — detection score threshold (raise to reject
  weak/false hits) and the downscale width (lower = faster).
- **Camera follow** — pan/tilt gain, ease time, max PTZ speed, aim lead, start/
  stop-follow error (hysteresis), re-acquire delay.
- **Tracker** — box steadiness, IoU match threshold, track max-age, min-hits.

All defaults live in `params.py`; direction/`.env` settings in `config.py`.

## Troubleshooting

- **No cameras found** — camera and Mac must be on the same subnet; enable
  ONVIF on the camera. macOS may prompt for local-network permission the first
  time; allow it.
- **`[Errno 65] No route to host` during discovery** — on macOS 15+/26 this is
  almost always **Local Network Privacy**: the terminal you launched Python from
  hasn't been granted local-network access. Enable your terminal app
  (Terminal / iTerm / Warp / VS Code) under **System Settings → Privacy &
  Security → Local Network**, then fully quit and reopen it. (It works when run
  from an already-approved app like Claude, which is why the same code can
  succeed there and fail in a bare terminal.)
- **`No route to host` with a VPN active** — discovery already pins the probe to
  your real LAN interface via `IP_BOUND_IF`, below the routing table. If
  auto-detection still picks wrong, set `TAPO_BIND_IP` to your Mac's LAN IP
  (find it with `ipconfig getifaddr en0` / `en1`), e.g. `TAPO_BIND_IP=10.0.0.165`.
- **ONVIF connection failed** — almost always wrong Camera Account
  username/password (not the cloud account).
- **Camera hunts / oscillates around a centered target** — gain too high for the
  ONVIF latency. Lower **Pan/Tilt gain**, keep **Aim lead = 0** (lead amplifies a
  stationary face's velocity noise), raise **Ease time**, and widen
  **Stop-follow err** so it settles into the hold band. Defaults are already
  tuned gently for this.
- **Camera turns the wrong way / spins** — run `calibrate.py` (or set
  `TAPO_INVERT_PAN=1` / `TAPO_INVERT_TILT=1`); the watchdog caps any runaway.
- **Frames freezing 0.5–1s / stutter** — RTSP now runs over **TCP** (retransmits
  instead of stalling on Wi-Fi packet loss) and detection runs downscaled. If it
  still stutters: lower **Detect scale width**, keep the detector on **face**
  (much lighter than pose), or point the tracked lens at an SD substream via
  `TAPO_CAM2_PATH`.
- **Faces not detected when far away** — BlazeFace short-range tops out ~2 m;
  move closer, lower **Detect conf**, or switch the detector to **pose**.
