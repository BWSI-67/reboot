"""Perception for labs/cone_slalom_GF.py.

The slalom here never computes a path around a cone. Two independent questions
get asked each frame:

    nearest_cone(image) -> "ORANGE" / "GREEN" / None      which SIDE to pass on
    pick_gap(scan, side) -> steering angle in rc degrees   WHERE to actually go

A cone is a lidar obstacle, so the gap beside it already exists in the scan.
Aiming at the leftmost open run while a green cone is ahead *is* passing on the
left -- no geometry, no offset setpoint, no turn timer, and nothing that breaks
when the cone leaves the camera's view mid-pass, which it always does.

Contrast with the usual approach (bias the camera setpoint by N pixels, then
turn open-loop at full lock until some beam says the cone is behind you): that
needs the cone tracked continuously right up to the turn, needs a dwell timer
to stop it oscillating, and cannot tell a passed cone from a wall.
"""

import cv2
import numpy as np

# Measured on this car (see deploys/cone/cone_hsv.py --sweep). S floor 80, not
# 120: the lit face of a cone washes toward white and its saturation drops to
# ~95 while the hue holds; at 120 only the shaded middle of the cone is seen.
# A grey floor reads S < 40, so 80 still clears the background comfortably.
# Do NOT raise the orange hue ceiling past 18 -- a yellow track sits at H 20-29
# and is large enough to beat every cone.
BANDS = {
    "ORANGE": ((3, 80, 60), (18, 255, 255)),      # measured H~8  S~255 V~203
    "GREEN": ((40, 70, 60), (85, 255, 255)),      # measured H=53 S=193 V=128
}
CROP_TOP = 200        # cones stand up; a line-following crop is far too low
MIN_AREA = 500        # a cone is a solid object, so this can be an order of
                      # magnitude above the value a thin line needs


def nearest_cone(image):
    """Which cone colour dominates the view, or None. Area is the proxy for
    'nearest' -- it is only used to choose a side, never as a distance."""
    hsv = cv2.cvtColor(image[CROP_TOP:], cv2.COLOR_BGR2HSV)
    best, best_area = None, MIN_AREA
    for name, (lo, hi) in BANDS.items():
        mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        contours = cv2.findContours(mask, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > best_area:
                best, best_area = name, area
    return best


def pick_gap(scan_rc, side, open_cm, fov_deg, min_run_deg):
    """Angle of the leftmost (side < 0) or rightmost (side > 0) open run.

    scan_rc is the rc-convention array: index 0 straight ahead, increasing
    clockwise, centimetres, 0 meaning no return. A no-return is NOT open space
    -- on this lidar it means a dark or grazing surface, and treating it as far
    away is what makes a gap follower drive at walls.

    Falls back to the widest run when the requested side has none, and to
    straight ahead when nothing is open at all.
    """
    n = len(scan_rc)
    span = int(fov_deg * n / 360)
    idx = np.concatenate([np.arange(n - span, n), np.arange(0, span + 1)])
    angles = np.linspace(-fov_deg, fov_deg, len(idx))
    depth = scan_rc[idx]

    open_mask = depth > open_cm
    runs, start = [], None
    for i, is_open in enumerate(open_mask):
        if is_open and start is None:
            start = i
        elif not is_open and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(open_mask) - 1))

    step = 2.0 * fov_deg / max(len(idx) - 1, 1)
    runs = [r for r in runs if (r[1] - r[0] + 1) * step >= min_run_deg]
    if not runs:
        return 0.0
    chosen = runs[-1] if side > 0 else runs[0]
    return float((angles[chosen[0]] + angles[chosen[1]]) / 2.0)


def scan_to_rc(ranges_m):
    """Raw ROS /scan -> rc convention. The lidar is mounted rotated on this
    car; the flip is the only conversion that puts index 0 at the front."""
    r = np.asarray(ranges_m, dtype=float)
    return np.flip(np.where(np.isfinite(r), r, 0.0)) * 100.0
