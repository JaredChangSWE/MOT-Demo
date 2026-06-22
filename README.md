# PTZ Camera Auto-Tracking Demo

A self-contained Python demo of **PTZ (pan/tilt/zoom) camera auto-tracking**. A
fixed camera watches a simulated 3D scene where 10 people wander randomly. The
system detects people, assigns stable tracking IDs, locks onto the first person
it recognizes, and continuously rotates the camera's pan/tilt to keep that
person centered in the frame.

Everything runs on just **NumPy + OpenCV** — no GPU, no model download, no heavy
3D engine.

## What it demonstrates

1. **3D scene simulation** — 10 people doing random-waypoint walks on a ground
   plane (`world.py`).
2. **Fixed PTZ camera** — pinhole projection; only pan (yaw) and tilt (pitch)
   change, never position (`camera.py`). Mounted at the **center of the scene**
   with a circular **keep-out zone** so people never walk onto the lens. Pan is
   **unlimited by default** — the camera rotates continuously (360°+) to follow
   a target all the way around (`--limit-pan` re-enables a mechanical stop).
   While a target is briefly out of view the camera keeps rotating after its
   Kalman-extrapolated position (HUD state `COASTING`) and re-centers when it
   reappears, so a fast target circling the camera is followed continuously.
3. **Object detection** — per-frame person detections with bounding box,
   confidence, and class (`detector.py`).
4. **Multi-object tracking** — five selectable algorithms (`trackers/`), each
   giving people stable IDs + motion trails, switchable live in the UI:
   **SORT**, **ByteTrack**, **DeepSORT**, **OC-SORT**, **BoT-SORT**. The current
   algorithm's per-frame **latency** is shown in the HUD. They share a Kalman
   box filter and assignment utilities (`trackers/base.py`); each adds its own
   association policy. See the "Trackers" section below.
5. **Auto-tracking control** — eased pan/tilt control (`controller.py`):
   - **Ease-in / ease-out** via a critically-damped SmoothDamp filter, so the
     camera accelerates gently from rest and decelerates smoothly onto the
     target instead of jerking.
   - **Engage range (hysteresis)** — the camera only starts moving once the
     centering error exceeds `engage`, and keeps tracking until it drops below
     `release`. Small wobbles are ignored; no chatter at the threshold.
6. **Visualization** — camera view with bounding boxes, IDs, trails, the locked
   target highlighted, a center crosshair, the error vector, and a status HUD
   (`renderer.py` + `overlay.py`).

## Detection note (why not a real YOLO?)

The scene is rendered synthetically, and a COCO-trained YOLO does **not** reliably
detect rendered figures — that would make the demo flaky. So `detector.py` uses a
**ground-truth detector** that derives detections from the simulator's true
geometry, with optional realistic noise (bbox jitter, varying confidence,
occasional missed detections). It implements a clean `detect(camera, world) ->
list[Detection]` interface, so a real `YOLODetector` could be dropped in later
without touching the rest of the pipeline.

## Trackers

Pick the algorithm with `--tracker NAME` (or the **"tracker 0-4"** slider in the
control panel — switching resets IDs). The HUD shows the active tracker and its
measured per-frame latency.

| Tracker | Mechanism | Appearance? | Notes |
|---------|-----------|-------------|-------|
| **SORT** | Kalman + IoU + Hungarian | no | the classic baseline |
| **ByteTrack** | SORT + two-stage (high/low-confidence) association | no | recovers tracks from low-score boxes |
| **DeepSORT** | Kalman + appearance + matching cascade | yes | prioritizes recently-seen tracks |
| **OC-SORT** | observation-centric momentum (OCM) + recovery (OCR) | no | robust to occlusion / nonlinear motion |
| **BoT-SORT** | ByteTrack + appearance fusion + camera-motion compensation | yes | uses the known camera motion for CMC |

**Two honest caveats for this demo:**

1. **Appearance = colour histogram, not a deep re-ID embedding.** Faithful
   DeepSORT/BoT-SORT use a CNN feature extractor (torch + weights). To stay
   dependency-light, the appearance feature here is an HSV histogram of the box
   crop — the *structure* of each algorithm is faithful; the descriptor is a
   lightweight stand-in.
2. **Detection is idealized (ground-truth).** Because detections are clean and
   never merge two people, the motion-only trackers (SORT / OC-SORT) already do
   very well, and the appearance-based ones (DeepSORT / BoT-SORT) mostly add
   latency without improving identity here. On real, noisy detections the
   appearance/CMC machinery is where DeepSORT/BoT-SORT earn their keep.

Indicative comparison (900 frames, default scene; latency on this machine):

| tracker | latency | distinct IDs | ID switches |
|---------|--------:|-------------:|------------:|
| SORT | ~0.07 ms | 36 | 27 |
| ByteTrack | ~0.07 ms | 36 | 27 |
| DeepSORT | ~0.20 ms | 38 | 35 |
| OC-SORT | ~0.10 ms | 24 | 29 |
| BoT-SORT | ~0.21 ms | 48 | 71 |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Interactive (opens a window):

```bash
python -m ptz_tracking.main
```

Keys: `q` quit · `space` pause · `r` re-pick target · `n` toggle detection noise

A **"PTZ Controls" window** opens alongside the demo with clearly-labelled live
sliders — tracker selector, people count, walk speed, keep-out radius, camera
ease time, max pan speed, aim lead, start/stop-follow error, re-acquire delay,
box steadiness, detection jitter and miss rate. Each label states its unit, and
a full plain-English legend is printed to the console when the panel opens. Drag
a slider and the change takes effect on the next frame (changing the people
count rebuilds the scene). Launch with `--no-ui` to hide the panel.

Headless (no window — runs for a duration, writes an mp4, prints metrics):

```bash
python -m ptz_tracking.main --headless --duration 30 --record out.mp4
```

## Parameters

Everything is tunable from the command line (`--help` lists all flags with
defaults). The knobs map to a single `Params` object (`params.py`) threaded
through every component. Highlights:

| Flag | Meaning |
|------|---------|
| `--people N` | number of people in the scene (default 10) |
| `--speed-scale X` | multiply walking speed (default 2.5; e.g. `--speed-scale 10` for 10×) |
| `--world-size M` | half-size of the square walkable area (m) |
| `--exclusion R` | keep-out radius around the camera, m (default 3.0) |
| `--camera-x/y/z` | camera mount position; default `(0,0,2.6)` = scene center |
| `--bbox-smoothing A` | EMA weight for box/trail; **lower = less jitter** (default 0.35) |
| `--smooth-time S` | ease-curve time; **larger = gentler**, smaller = snappier (default 0.20) |
| `--max-speed D` | cap on pan/tilt speed, deg/s |
| `--lead F` | aim F frames ahead along the target's velocity (keeps fast targets centered; default 6) |
| `--limit-pan` | re-enable the mechanical pan limit (default: unlimited 360°) |
| `--engage E` | start tracking once normalized error exceeds this (default 0.15) |
| `--release R` | stop tracking once error drops below this (default 0.04) |
| `--fovy`, `--pan-limit`, `--tilt-min/max` | camera optics & mechanical limits |
| `--jitter`, `--drop`, `--no-noise` | detection-noise model |
| `--tracker NAME` | SORT / ByteTrack / DeepSORT / OC-SORT / BoT-SORT |
| `--iou`, `--max-age`, `--trail`, `--match-dist` | tracker association |
| `--duration S` / `--frames N` | headless run length |
| `--seed N` | world RNG seed |

Examples:

```bash
# Fast, lively scene with a gentle camera
python -m ptz_tracking.main --speed-scale 6 --smooth-time 0.5

# Ultra-smooth boxes, camera only reacts to large excursions
python -m ptz_tracking.main --bbox-smoothing 0.2 --engage 0.25 --release 0.05

# 45-second headless soak test with video
python -m ptz_tracking.main --headless --duration 45 --record out.mp4
```

## Layout

```
ptz_tracking/
├── config.py      # fixed constants + coordinate conventions
├── params.py      # Params dataclass: all CLI-tunable knobs (single source)
├── world.py       # 3D scene: Person agents + random-waypoint walking
├── camera.py      # fixed PTZ camera: pinhole projection + pan/tilt
├── detector.py    # Detection contract + ground-truth detector
├── trackers/      # pluggable trackers (selectable at runtime)
│   ├── base.py       # shared Track, Kalman box filter, assignment + appearance utils
│   ├── sort.py · bytetrack.py · deepsort.py · ocsort.py · botsort.py
│   └── __init__.py   # registry + make_tracker() + TRACKER_NAMES
├── controller.py  # eased (SmoothDamp) pan/tilt controller w/ engage range
├── renderer.py    # draws the camera's view of the scene
├── overlay.py     # draws boxes, IDs, trails, target, crosshair, HUD
├── controls.py    # on-screen slider panel (live Params tuning)
└── main.py        # orchestration loop + keyboard controls + CLI
```
