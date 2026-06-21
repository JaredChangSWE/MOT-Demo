"""3D scene with people that wander via random waypoints.

World frame: X right, Y forward (depth), Z up; ground plane at Z=0.
Each Person walks on the ground toward a random waypoint, picking a new one on
arrival. The world is a flat square of side 2*WORLD_HALF centered at the origin.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config
from .params import Params


@dataclass
class Person:
    """A single walking agent on the ground plane."""

    pid: int
    pos: np.ndarray            # shape (2,), ground position (x, y) in meters
    height: float              # meters
    speed: float               # meters / second
    color: tuple[int, int, int]  # BGR, for rendering
    waypoint: np.ndarray = field(default=None)  # shape (2,)

    @property
    def feet(self) -> np.ndarray:
        """3D world point at the feet (x, y, 0)."""
        return np.array([self.pos[0], self.pos[1], 0.0])

    @property
    def head(self) -> np.ndarray:
        """3D world point at the top of the head (x, y, height)."""
        return np.array([self.pos[0], self.pos[1], self.height])


def _rand_ground(rng: np.random.Generator, half: float,
                 cam_xy: np.ndarray, exclusion: float) -> np.ndarray:
    """Random ground point inside the square but OUTSIDE the keep-out disc."""
    for _ in range(64):
        pt = rng.uniform(-half, half, size=2)
        if np.linalg.norm(pt - cam_xy) >= exclusion:
            return pt
    # Fallback: push the last sample out to the ring.
    d = pt - cam_xy
    n = np.linalg.norm(d) or 1.0
    return cam_xy + d / n * exclusion


# Distinct-ish BGR palette so people are visually separable.
_PALETTE = [
    (66, 135, 245), (60, 200, 90), (40, 40, 220), (200, 160, 40),
    (180, 60, 200), (40, 200, 200), (220, 120, 60), (120, 80, 200),
    (90, 190, 160), (200, 90, 140),
]


class World:
    """Holds all people and advances the simulation."""

    def __init__(self, params: Params | None = None, seed: int | None = 0):
        self.p = params or Params()
        self.rng = np.random.default_rng(seed)
        half = self.p.world_half
        cam_xy = self._cam_xy()
        excl = self.p.exclusion_radius
        self.people: list[Person] = []
        for i in range(self.p.num_people):
            # `speed` is the BASE speed; speed_scale is applied live in step().
            speed = self.rng.uniform(self.p.person_speed_min, self.p.person_speed_max)
            p = Person(
                pid=i,
                pos=_rand_ground(self.rng, half, cam_xy, excl),
                height=float(self.rng.uniform(*config.PERSON_HEIGHT_RANGE)),
                speed=float(speed),
                color=_PALETTE[i % len(_PALETTE)],
            )
            p.waypoint = _rand_ground(self.rng, half, cam_xy, excl)
            self.people.append(p)

    def _cam_xy(self) -> np.ndarray:
        return np.array([self.p.camera_x, self.p.camera_y])

    def step(self, dt: float | None = None) -> None:
        """Advance every person toward its waypoint by one timestep."""
        dt = self.p.dt if dt is None else dt
        half = self.p.world_half
        cam_xy = self._cam_xy()
        excl = self.p.exclusion_radius
        for p in self.people:
            to_wp = p.waypoint - p.pos
            dist = float(np.linalg.norm(to_wp))
            if dist < config.WAYPOINT_REACHED_DIST:
                p.waypoint = _rand_ground(self.rng, half, cam_xy, excl)
                continue
            direction = to_wp / dist
            p.pos = p.pos + direction * (p.speed * self.p.speed_scale) * dt
            # Keep inside the square.
            p.pos = np.clip(p.pos, -half, half)
            # Keep outside the camera keep-out disc: if a step would enter it,
            # push back onto the ring and retarget away from the camera.
            offset = p.pos - cam_xy
            d = float(np.linalg.norm(offset))
            if d < excl:
                n = offset / (d or 1.0)
                p.pos = cam_xy + n * excl
                p.waypoint = _rand_ground(self.rng, half, cam_xy, excl)
