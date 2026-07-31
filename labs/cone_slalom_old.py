"""The pre-ets cone slalom, kept runnable as a fallback.

Frozen copy of the version driven on 2026-07-30 before the steering was
swapped to rb_ets: camera area-vote picks the side (0.2 s latch, no debounce,
no wall filters), pick_gap thresholds the scan into open runs and aims at the
centre of the leftmost/rightmost one. Self-contained on purpose -- it does NOT
import cone_sight, so the main lab can keep evolving without touching this.

    python3 cone_slalom_old.py        # car
    python3 cone_slalom_old.py -s     # sim (red/blue cones)
"""

MAX_SPEED, CRUISE = 0.35, 0.8  # ceiling, then the FRACTION of it to cruise at
kP, TRIM = 0.8, 0.04
ANGLE_MAX = 0.6                # hard steering ceiling: no situation (fallback
                               # centres included) is allowed to full-lock
SLOW, SPEED_MIN = 0.7, 0.4     # shed speed with steering, floor as a FRACTION
ORANGE_SIDE = 1                # +1 = pass ORANGE on its right, GREEN on its
                               # left; flip this one number to swap the course
OPEN_CM, FOV, MIN_RUN = 150, 105, 16   # a run must be this wide to count, in deg
MARGIN = 18                    # aim this many degrees past the cone-side edge
                               # of the chosen run, NOT at the run centre: a
                               # wide run's centre sits 60+ deg out and
                               # saturates the steering into a giant arc, a
                               # narrow run's centre hugs the cone -- a fixed
                               # edge offset gives every cone the same
                               # clearance and never full-locks for open space
FRONT_CM, FRONT_ARC = 60, 25   # closer than this dead ahead: crawl
SIDE_HOLD_S = 0.5              # after a side switch, ignore the camera this long
N_CONFIRM = 3                  # camera frames (~10 Hz) that must agree before
                               # a switch: seeing the NEXT gate mid-pass must
                               # not instantly flip the side into the cone
                               # being passed

angle = 0.0
side = ORANGE_SIDE
side_locked_until = 0.0
pend_side, pend_count = 0, 0
last_frame = None
import os, sys, time
import cv2
import numpy as np
sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar()   # real on the car, sim with -s
CONES = ("RED", "BLUE") if "-s" in sys.argv[1:] else ("ORANGE", "GREEN")
time.sleep(2)

BANDS = {
    "ORANGE": (((3, 80, 60), (18, 255, 255)),),
    "GREEN": (((40, 70, 60), (85, 255, 255)),),
    "RED": (((0, 120, 60), (6, 255, 255)), ((170, 120, 60), (180, 255, 255))),
    "BLUE": (((100, 120, 60), (130, 255, 255)),),
}
CROP_TOP = 200
MIN_AREA = 500


def nearest_cone(image, names):
    """Colour of the NEAREST cone: the contour whose bottom edge sits lowest
    in the frame. Floor-standing objects are lower when closer, so the far
    gate cone -- fully visible and often bigger in AREA than the half-cropped
    near one -- can never outvote the cone still being passed."""
    hsv = cv2.cvtColor(image[CROP_TOP:], cv2.COLOR_BGR2HSV)
    best, best_bottom = None, -1
    for name in names:
        mask = None
        for lo, hi in BANDS[name]:
            m = cv2.inRange(hsv, np.array(lo, np.uint8),
                            np.array(hi, np.uint8))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        contours = cv2.findContours(mask, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            if cv2.contourArea(contour) < MIN_AREA:
                continue
            _, y, _, h = cv2.boundingRect(contour)
            if y + h > best_bottom:
                best, best_bottom = name, y + h
    return best


def pick_gap(scan_rc, side, open_cm, fov_deg, min_run_deg):
    """Centre of the leftmost (side < 0) or rightmost (side > 0) open run.
    Zeros stay closed, but lone zero gaps (< ~2 deg) are bridged so salt
    noise cannot chop a real opening below min_run_deg."""
    n = len(scan_rc)
    span = int(fov_deg * n / 360)
    idx = np.concatenate([np.arange(n - span, n), np.arange(0, span + 1)])
    angles = np.linspace(-fov_deg, fov_deg, len(idx))
    depth = scan_rc[idx]

    max_gap = max(1, int(2.0 * n / 360))
    i, m = 0, len(depth)
    while i < m:
        if depth[i] == 0:
            j = i
            while j < m and depth[j] == 0:
                j += 1
            if 0 < i and j < m and (j - i) <= max_gap:
                depth[i:j] = min(depth[i - 1], depth[j])
            i = j
        else:
            i += 1

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
    lo, hi = angles[chosen[0]], angles[chosen[1]]
    # Fixed offset past the cone-side edge (see MARGIN above). When that edge
    # is the FOV boundary instead of a real obstacle -- an all-open scan --
    # fall back to the centre, which is then near straight ahead.
    if side > 0:
        if chosen[0] > 0:
            return float(min(lo + MARGIN, hi))
        return float((lo + hi) / 2.0)
    if chosen[1] < len(angles) - 1:
        return float(max(hi - MARGIN, lo))
    return float((lo + hi) / 2.0)


def start():
    rc.drive.stop()
    rc.set_update_slow_time(0.1)
    rc.drive.set_max_speed(MAX_SPEED)


def update():
    global angle, side, side_locked_until, pend_side, pend_count, last_frame

    try:
        image = rc.camera.get_color_image_no_copy()
    except ValueError:
        image = None
    if image is not None and id(image) != last_frame:
        last_frame = id(image)
        cone = nearest_cone(image, CONES)
        if cone is not None:
            vote = ORANGE_SIDE if cone == CONES[0] else -ORANGE_SIDE
            if vote == side:
                pend_count = 0          # current side re-confirmed
            elif vote == pend_side:
                pend_count += 1
            else:
                pend_side, pend_count = vote, 1
            now = time.monotonic()
            if pend_count >= N_CONFIRM and now >= side_locked_until:
                side = pend_side
                pend_count = 0
                side_locked_until = now + SIDE_HOLD_S

    try:
        scan = rc.lidar.get_samples()
    except ValueError:
        return
    if len(scan) == 0:
        return rc.drive.stop()

    gap = pick_gap(scan, side, OPEN_CM, FOV, MIN_RUN)
    angle = float(np.clip(kP * gap / FOV, -ANGLE_MAX, ANGLE_MAX))
    speed = max(SPEED_MIN, CRUISE - SLOW * abs(angle))

    arc = int(FRONT_ARC * len(scan) / 360)
    front = np.concatenate([scan[-arc:], scan[:arc + 1]])
    front = front[front > 0]
    if front.size and front.min() < FRONT_CM:
        speed = SPEED_MIN * 0.5
    rc.drive.set_speed_angle(speed, rc_utils.clamp(angle + TRIM, -1.0, 1.0))


def update_slow():   # gap 0.0 every frame = no run passed MIN_RUN
    print(f"side {'R' if side > 0 else 'L'}  angle {angle:+.2f}")


rc.set_start_update(start, update, update_slow)
rc.go()
