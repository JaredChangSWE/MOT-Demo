"""Central configuration for the PTZ auto-tracking demo.

All tunable constants live here so the rest of the package stays declarative.
Coordinate convention (right-handed world frame):
    X -> right, Y -> forward / depth into the scene, Z -> up.
Distances are in meters, angles in radians unless a name ends in `_deg`.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Image / display
# ---------------------------------------------------------------------------
IMAGE_W = 960
IMAGE_H = 540
FPS = 30
DT = 1.0 / FPS  # simulation timestep (seconds)

# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------
NUM_PEOPLE = 10
# Walkable ground area: x,y in [-HALF, HALF].
WORLD_HALF = 8.0
PERSON_HEIGHT_RANGE = (1.6, 1.9)   # meters
PERSON_SPEED_RANGE = (0.6, 1.6)    # meters / second
WAYPOINT_REACHED_DIST = 0.4        # meters

# ---------------------------------------------------------------------------
# Camera (PTZ: fixed position, pan + tilt only)
# ---------------------------------------------------------------------------
CAMERA_POS = (0.0, -11.0, 2.6)     # fixed mounting point (x, y, z)
CAMERA_FOVY_DEG = 50.0             # vertical field of view
CAMERA_NEAR = 0.2                  # near plane (meters); closer points are culled

# Pan = yaw about world Z (0 = looking along +Y; positive turns toward +X / right).
# Tilt = pitch (0 = horizontal; positive looks up, negative looks down).
# Initial pose only; mechanical pan/tilt limits live in params.Params.
PAN_INIT = 0.0
TILT_INIT = math.radians(-18.0)

# ---------------------------------------------------------------------------
# Detection (ground-truth simulator)
# ---------------------------------------------------------------------------
DET_BBOX_JITTER_PX = 2.0           # std-dev of gaussian noise added to bbox edges
DET_CONF_RANGE = (0.70, 0.98)
DET_DROP_PROB = 0.05               # per-person chance of a missed detection (noise mode)
DET_MIN_BBOX_PX = 8.0              # ignore detections smaller than this (height)
DET_MIN_VISIBLE_FRAC = 0.35        # person bbox must overlap image by at least this fraction

# ---------------------------------------------------------------------------
# Tracker (SORT-style IoU association)
# ---------------------------------------------------------------------------
TRK_IOU_THRESHOLD = 0.2
TRK_MAX_AGE = 15                   # frames a track survives without a match
TRK_MIN_HITS = 2                   # consecutive hits before a track is "confirmed"
TRK_TRAIL_LEN = 40                 # number of trajectory points kept per track
# Centroid-distance fallback: after IoU matching, also match a track to the
# nearest unmatched detection whose center is within this multiple of the
# track's bbox size. Keeps IDs stable when camera ego-motion shifts boxes
# farther than their own width (so IoU alone would drop to 0).
TRK_MATCH_DIST_FACTOR = 1.5

# ---------------------------------------------------------------------------
# PTZ controller
# ---------------------------------------------------------------------------
# NOTE: the controller uses a SmoothDamp ease-curve + engage/release hysteresis.
# Its live tunables (ctrl_smooth_time, ctrl_max_speed_deg, ctrl_engage_error,
# ctrl_release_error) live in params.Params and are exposed as CLI flags.
CTRL_TARGET_LOST_GRACE = 30        # frames to hold before re-acquiring a new target
