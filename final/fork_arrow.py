"""
MIT BWSI Autonomous RACECAR -- Mirror-fork section (blue arrow decides)

State flow:
    Cruise   max-gap follow, arrow votes  -> Commit  when close + clean vote
                                                     (>= K one side, ZERO opposite)
    Commit   gap search restricted to the -> Done    after FORK_TIME elapses
             arrow's half of the scan
    Done     release control

Call fork() once per frame from update(). It returns (speed, angle),
or None once the section is finished. fork_ready() tells the dispatcher
the prop is in sight (used to end the transit leg before this section).

Why it is built this way -- verified on venue footage (754 frames, zero
wrong-direction reads):
  * The mirrors defeat every range sensor: LiDAR beams bounce off specularly
    (dropouts / phantom far readings) and to a camera the mirror shows open
    track. The arrow is the only honest signal on the prop.
  * The arrow gets flipped BY HAND between runs and mid-run. Hence the
    conflict veto: one opposite vote inside the window blocks the commit and
    the car creeps to buy frames instead of gambling.
  * After commit the max-gap search only sees the arrow's half, so a mirror
    faking an open gap on the wrong side can never win -- the arrow decides
    the side, the gap follower only aims within it.
  * Direction needs two independent shape cues to agree, else the frame
    abstains. (x-skewness was tried and measures ~0 on the real arrow: the
    flared tail balances the wide head. Do not resurrect it.)
"""

import numpy as np
import cv2 as cv
import racecar_utils as rc_utils


# =====================================================================
#  TUNING -- FORK
# =====================================================================

# --- blue arrow (OpenCV: H 0-179; measured H 105-112 off venue video) ---
BLUE_LOW        = np.array([100, 60, 80])
BLUE_HIGH       = np.array([118, 255, 255])

# --- glare rejection (bright AND colorless = specular highlight) ---
GLARE_V_MIN     = 225
GLARE_S_MAX     = 70
GLARE_DILATE_PX = 5

# --- arrow ROI (fractions of frame height) ---
ARROW_TOP_FRAC    = 0.15   # skip ceiling lights
ARROW_BOTTOM_FRAC = 0.75   # skip glossy floor

# --- arrow shape gates ---
MIN_AREA        = 80       # blob px at 640x480; below = too far to call
ASPECT_MIN      = 1.2      # mid-rotation arrow fails these and abstains
ASPECT_MAX      = 4.5
SOLIDITY        = (0.50, 0.97)   # tail notch keeps a real arrow under 0.97

# --- direction cues (both must agree, else abstain) ---
TIP_RATIO       = 0.75     # cue 1: tip-side outer-20% cols vs tail side
TIP_PEAK        = 0.35     # cue 2: tip end must drop under this x peak
TAIL_PEAK       = 0.45     #        while tail end stays above this x peak

# --- voting and commit ---
VOTE_N          = 7        # sliding window (frames)
VOTE_K          = 4        # same-direction frames required
TRIGGER_AREA    = 600      # blob px = close enough (~3 m); depth overrides
FORK_TRIGGER_CM = 300      # metric gate when the depth stream is usable
FORK_TIME       = 2.5      # s the gap search stays on the chosen half
CRUISE_SPEED    = 0.50
COMMIT_SPEED    = 0.45
PENDING_SPEED   = 0.20     # at the gate without a clean vote: creep, re-read

# --- max-gap steering ---
GAP_WINDOW_DEG  = 30
KP              = 1.1
SECTOR_OVERLAP  = 15       # deg past center kept in the allowed half

# --- depth standby (OFF until measured on the car) ---
# Park the car 2-3 m from the prop and check the scan for base echoes.
# Lidar sees the low base -> leave off. Lidar slice passes over it -> flip
# on: the depth image is folded into the scan so findMaxW cannot aim at it.
USE_DEPTH       = False
DEPTH_FOV_DEG   = 87.0     # horizontal FOV of the depth stream; check spec
DEPTH_ROWS      = (0.35, 0.60)   # rows folded in: below = floor, above = stands
DEPTH_MIN_CM    = 30       # closer = sensor noise, ignore

DEBUG_PRINT     = True


# =====================================================================
#  STATE -- persists between calls
# =====================================================================

fork_state   = "Cruise"
_votes       = []          # last VOTE_N per-frame reads ("LEFT"/"RIGHT"/None)
_fork_dir    = None
_fork_timer  = 0.0
_last_error  = 0.0
_last_fid    = None        # id() of the last processed camera frame
_dup_speed   = 0.50        # speed to hold on duplicate frames (lidar steers)


def fork_reset():
    """Re-arm the whole section. Call from start()."""
    global fork_state, _votes, _fork_dir, _fork_timer, _last_error
    global _last_fid, _dup_speed
    fork_state = "Cruise"
    _votes = []
    _fork_dir = None
    _fork_timer = 0.0
    _last_error = 0.0
    _last_fid = None
    _dup_speed = CRUISE_SPEED


def fork_ready(image=None):
    """Prop in sight? Ends the transit leg. A false engage is harmless:
    Cruise here is the same max-gap follow the transit runs."""
    image = rc.camera.get_color_image()
    if image is None:
        return False
    _direction, _area, bbox = _read_arrow(image)
    return bbox is not None


def _goto(new_state):
    global fork_state
    if new_state != fork_state and DEBUG_PRINT:
        print("[fork] {} -> {}".format(fork_state, new_state))
    fork_state = new_state


# =====================================================================
#  VISION -- read the arrow
# =====================================================================

def _read_arrow(image):
    """Returns (direction, area, bbox). direction "LEFT"/"RIGHT"/None.

    None covers: no blue, too small, wrong shape, or the two direction cues
    disagreed. Abstaining costs one of 30 frames/s; guessing costs a vote.
    """
    height, width = image.shape[:2]
    top = int(height * ARROW_TOP_FRAC)
    bottom = int(height * ARROW_BOTTOM_FRAC)
    hsv = cv.cvtColor(image[top:bottom], cv.COLOR_BGR2HSV)

    blue = cv.inRange(hsv, BLUE_LOW, BLUE_HIGH)
    _, s, v = cv.split(hsv)
    glare = cv.bitwise_and(cv.inRange(v, GLARE_V_MIN, 255),
                           cv.inRange(s, 0, GLARE_S_MAX))
    kernel = np.ones((GLARE_DILATE_PX, GLARE_DILATE_PX), np.uint8)
    mask = cv.bitwise_and(blue, cv.bitwise_not(cv.dilate(glare, kernel)))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = rc_utils.get_largest_contour(contours, MIN_AREA)
    if contour is None:
        return None, 0, None

    x, y, w, h = cv.boundingRect(contour)
    area = rc_utils.get_contour_area(contour)
    bbox = (x, y + top, x + w, y + h + top)

    if not (ASPECT_MIN <= w / max(h, 1) <= ASPECT_MAX):
        return None, area, bbox
    hull = cv.convexHull(contour)
    hull_a = cv.contourArea(hull)
    sol = area / hull_a if hull_a > 0 else 1.0
    if not (SOLIDITY[0] <= sol <= SOLIDITY[1]):
        return None, area, bbox

    blob = np.zeros(mask.shape, np.uint8)
    cv.drawContours(blob, [contour], -1, 255, cv.FILLED)
    cols = (blob[y:y + h, x:x + w] > 0).sum(axis=0).astype(np.float32)

    # cue 1: outer-column ratio -- the tip side is the skinny side
    k = max(3, int(0.2 * w))
    lmean, rmean = float(cols[:k].mean()), float(cols[-k:].mean())
    ratio_dir = ("LEFT" if lmean < TIP_RATIO * rmean else
                 "RIGHT" if rmean < TIP_RATIO * lmean else None)

    # cue 2: taper-to-peak -- the tip must really come to a point
    peak = float(cols.max())
    k2 = max(2, int(0.08 * w))
    l_end, r_end = float(cols[:k2].mean()), float(cols[-k2:].mean())
    taper_dir = ("RIGHT" if r_end < TIP_PEAK * peak and l_end > TAIL_PEAK * peak else
                 "LEFT" if l_end < TIP_PEAK * peak and r_end > TAIL_PEAK * peak else None)

    direction = ratio_dir if ratio_dir is not None and ratio_dir == taper_dir else None
    return direction, area, bbox


# =====================================================================
#  LIDAR -- max-gap steering, depth standby
# =====================================================================

def _find_max_gap(scan, min_a, max_a):
    """Angle (deg, 0 = forward, + = right) of the widest opening in the sector."""
    length = scan.shape[0]
    origin = min_a
    start = int((min_a % 360) * length / 360)
    end = int((max_a % 360) * length / 360)
    s_scan = (np.append(scan[start:], scan[:end + 1]) if start > end
              else scan[start:end + 1])
    w_size = int(GAP_WINDOW_DEG * length / 360)
    sums = np.convolve(s_scan, np.ones(w_size), "valid")
    return origin + (int(sums.argmax()) + w_size // 2) * 360 / length


def _merge_depth(scan):
    """Fold the depth camera's vertical FOV into the planar scan (standby).

    Each depth column's nearest return (floor/ceiling rows excluded) is
    min()-ed into the scan at that bearing: a low base the lidar slice
    passes over is a near wall in depth, so the merged scan can never call
    that direction a gap. Invalid depth (0/NaN: mirror, out of range)
    contributes nothing -- a mirror must not fake free space, and keeping
    mirrors out of play entirely is the sector restriction's job.
    """
    depth = _get_depth()
    if depth is None:
        return scan, None
    h, w = depth.shape[:2]
    band = depth[int(h * DEPTH_ROWS[0]):int(h * DEPTH_ROWS[1])].astype(np.float32)
    band = np.where(np.isfinite(band) & (band > DEPTH_MIN_CM), band, np.inf)
    col_min = band.min(axis=0)
    scan = np.array(scan, np.float32)
    ang = (np.arange(w) / (w - 1) - 0.5) * DEPTH_FOV_DEG
    idx = np.round(ang / 360.0 * scan.shape[0]).astype(int) % scan.shape[0]
    ok = np.isfinite(col_min)
    np.minimum.at(scan, idx[ok], col_min[ok])
    return scan, depth


def _get_depth():
    if not USE_DEPTH:
        return None
    try:
        return rc.camera.get_depth_image()
    except Exception:
        return None


# =====================================================================
#  MAIN STATE MACHINE
# =====================================================================

def fork():
    """One frame of the fork section. Returns (speed, angle), or None when Done."""
    global _votes, _fork_dir, _fork_timer, _last_error

    if fork_state == "Done":
        return None

    dt = rc.get_delta_time()

    scan = rc.lidar.get_samples()
    scan, depth = _merge_depth(scan)

    # steering sector: full width while cruising, the arrow's half once latched
    if fork_state == "Commit" and _fork_dir == "LEFT":
        gap = _find_max_gap(scan, -90, SECTOR_OVERLAP)
    elif fork_state == "Commit" and _fork_dir == "RIGHT":
        gap = _find_max_gap(scan, -SECTOR_OVERLAP, 90)
    else:
        gap = _find_max_gap(scan, -90, 90)
    error = gap / 180
    d_error = (error - _last_error) / dt if dt > 0 else 0.0
    _last_error = error
    angle = float(np.clip(error * KP + d_error * 0.02, -1.0, 1.0))

    # -----------------------------------------------------------------
    # Commit -- ride the restricted sector through the gap
    # -----------------------------------------------------------------
    if fork_state == "Commit":
        _fork_timer -= dt
        if _fork_timer <= 0.0:
            _goto("Done")
            return None
        return (COMMIT_SPEED, angle)

    # -----------------------------------------------------------------
    # Cruise -- gap follow, read the arrow, commit on a clean vote.
    # Duplicate camera frames (camera ~23-30 fps, loop 60 Hz) skip the
    # vision block entirely; the lidar steering above already ran.
    # -----------------------------------------------------------------
    global _last_fid, _dup_speed
    try:
        image = rc.camera.get_color_image_no_copy()
    except AttributeError:
        image = rc.camera.get_color_image()
    if image is None:
        return (CRUISE_SPEED, angle)
    if id(image) == _last_fid:
        return (_dup_speed, angle)
    _last_fid = id(image)

    direction, area, bbox = _read_arrow(image)
    _votes = (_votes + [direction])[-VOTE_N:]

    # close-enough gate: metric when depth is usable, pixel area otherwise
    close = area >= TRIGGER_AREA
    if depth is not None and bbox is not None:
        x1, y1, x2, y2 = bbox
        d = depth[max(0, y1):y2, max(0, x1):x2]
        d = d[np.isfinite(d) & (d > DEPTH_MIN_CM)]
        if d.size:
            close = float(np.median(d)) < FORK_TRIGGER_CM

    if close:
        n_l = _votes.count("LEFT")
        n_r = _votes.count("RIGHT")
        # conflict veto: ANY opposite vote in the window blocks the commit --
        # the arrow gets flipped by hand, and a flip mid-window looks exactly
        # like this. Creep and re-read; never gamble on a changing sign.
        pick = ("LEFT" if n_l >= VOTE_K and n_r == 0 else
                "RIGHT" if n_r >= VOTE_K and n_l == 0 else None)
        if pick:
            _fork_dir = pick
            _fork_timer = FORK_TIME
            _votes = []
            _goto("Commit")
            if DEBUG_PRINT:
                print("[fork] arrow says {} -> committing".format(pick))
            return (COMMIT_SPEED, angle)
        if DEBUG_PRINT:
            print("[fork] pending | area {:.0f} | votes {}".format(area, _votes))
        _dup_speed = PENDING_SPEED
        return (PENDING_SPEED, angle)

    _dup_speed = CRUISE_SPEED
    return (CRUISE_SPEED, angle)
