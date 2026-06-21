"""Fixed-position PTZ camera with a pinhole projection model.

The camera never translates; only `pan` (yaw about world Z) and `tilt` (pitch)
change. Projection maps world points -> image pixels using the camera basis
derived from pan/tilt.

Camera basis at pan=p, tilt=t:
    forward = (sin p cos t,  cos p cos t,  sin t)   # pan=0,tilt=0 -> +Y
    right   = (cos p,       -sin p,        0)        # horizontal
    up      = right x forward                        # ~+Z when tilt=0
Image pixel coords: u to the right, v downward, origin top-left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config
from .params import Params
from .world import Person, World


@dataclass
class BBox:
    """Axis-aligned bounding box in image pixels plus camera-space depth."""

    x1: float
    y1: float
    x2: float
    y2: float
    depth: float  # camera-forward distance (meters), for occlusion sorting

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))

    def as_int(self) -> tuple[int, int, int, int]:
        return (int(round(self.x1)), int(round(self.y1)),
                int(round(self.x2)), int(round(self.y2)))


class Camera:
    def __init__(
        self,
        params: Params | None = None,
        pos: tuple[float, float, float] | None = None,
        pan: float = config.PAN_INIT,
        tilt: float = config.TILT_INIT,
    ):
        self.p = params or Params()
        if pos is None:
            pos = (self.p.camera_x, self.p.camera_y, self.p.camera_z)
        self.pos = np.array(pos, dtype=float)
        self.pan = pan
        self.tilt = tilt
        self.image_w = self.p.image_w
        self.image_h = self.p.image_h
        self.fovy = math.radians(self.p.fovy_deg)
        # Square pixels: focal length in pixels from vertical FOV.
        self.focal = (self.image_h / 2.0) / math.tan(self.fovy / 2.0)

    # -- geometry ----------------------------------------------------------
    @property
    def fovx(self) -> float:
        """Horizontal field of view (radians), derived from aspect ratio."""
        return 2.0 * math.atan((self.image_w / 2.0) / self.focal)

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p, t = self.pan, self.tilt
        cp, sp, ct, st = math.cos(p), math.sin(p), math.cos(t), math.sin(t)
        forward = np.array([sp * ct, cp * ct, st])
        right = np.array([cp, -sp, 0.0])
        up = np.cross(right, forward)
        return right, up, forward

    def clamp(self) -> None:
        """Clamp tilt to mechanical limits; pan is either wrapped (unlimited
        360° rotation) or clamped. Read live so the UI can change them."""
        tilt_low = math.radians(self.p.tilt_limit_low_deg)
        tilt_high = math.radians(self.p.tilt_limit_high_deg)
        self.tilt = float(np.clip(self.tilt, tilt_low, tilt_high))
        if self.p.pan_unlimited:
            # Wrap to [-pi, pi] so the value stays bounded while allowing the
            # camera to spin continuously in either direction.
            self.pan = float((self.pan + math.pi) % (2.0 * math.pi) - math.pi)
        else:
            pan_limit = math.radians(self.p.pan_limit_deg)
            self.pan = float(np.clip(self.pan, -pan_limit, pan_limit))

    # -- projection --------------------------------------------------------
    def project_point(self, world_xyz: np.ndarray) -> tuple[float, float, float] | None:
        """Project a 3D world point to (u, v, depth). None if behind near plane."""
        right, up, forward = self._basis()
        d = np.asarray(world_xyz, dtype=float) - self.pos
        zc = float(np.dot(d, forward))
        if zc <= config.CAMERA_NEAR:
            return None
        xc = float(np.dot(d, right))
        yc = float(np.dot(d, up))
        u = self.image_w / 2.0 + self.focal * (xc / zc)
        v = self.image_h / 2.0 - self.focal * (yc / zc)
        return u, v, zc

    def project_person(self, person: Person) -> BBox | None:
        """Project a person to an image bbox. None if not (sufficiently) visible.

        The box spans projected head->feet vertically; width comes from a
        ~0.5 m shoulder span at the person's depth.
        """
        feet = self.project_point(person.feet)
        head = self.project_point(person.head)
        if feet is None or head is None:
            return None
        u_feet, v_feet, z_feet = feet
        u_head, v_head, z_head = head
        depth = 0.5 * (z_feet + z_head)
        u = 0.5 * (u_feet + u_head)
        half_w = 0.5 * (0.5 * self.focal / depth)  # 0.5 m shoulder width
        x1, x2 = u - half_w, u + half_w
        y1, y2 = min(v_head, v_feet), max(v_head, v_feet)
        bbox = BBox(x1, y1, x2, y2, depth)

        if bbox.height < config.DET_MIN_BBOX_PX:
            return None
        if self._visible_fraction(bbox) < config.DET_MIN_VISIBLE_FRAC:
            return None
        return bbox

    def _visible_fraction(self, b: BBox) -> float:
        """Fraction of the bbox area that lies inside the image frame."""
        ix1, iy1 = max(b.x1, 0.0), max(b.y1, 0.0)
        ix2, iy2 = min(b.x2, self.image_w), min(b.y2, self.image_h)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        area = b.width * b.height
        return (iw * ih) / area if area > 0 else 0.0

    @property
    def frame_center(self) -> tuple[float, float]:
        return (self.image_w / 2.0, self.image_h / 2.0)


def visible_people(camera: Camera, world: World) -> list[tuple[Person, BBox]]:
    """All people currently visible, sorted far -> near (painter's order)."""
    out: list[tuple[Person, BBox]] = []
    for p in world.people:
        b = camera.project_person(p)
        if b is not None:
            out.append((p, b))
    out.sort(key=lambda pb: pb[1].depth, reverse=True)
    return out
