"""
Blue arrow fork — read which way the arrow points, then take that side with the
depth camera.

Call blue_arrow() once per frame from update(). It returns (speed, angle), or
None once the section is finished.

STATES
    SEEK     no verdict yet: drive the corridor on depth, keep reading the arrow
    ALIGN    verdict latched: look ONLY at that side, find the doorway, aim at
             the middle of it. If that side is shut, hold at standoff -- the
             gate reopens within ~10 s
    THROUGH  lined up and at the doorway: drive through
    DONE     released

EVERYTHING HERE IS IN CENTIMETRES, NOT PIXELS
Depth pixels get reprojected to 3D and then filtered by HEIGHT ABOVE THE FLOOR.
The car cares about one thing: is there something in the slab of air its own
body sweeps through. That question has the same answer at 1 m and at 4 m, so it
needs no distance-dependent tuning, and the numbers that define it -- how tall
the car is, how wide it is -- come off a ruler instead of off the track.

This is what kills the two mirror problems geometrically:

  the panel overhead   A gate lifts to open, so once raised it hangs ABOVE the
                       slab and is dropped on height alone. It never matters
                       that it mirrors the corridor and reads far.

  the floor            Floor points sit at height ~0, below the slab, at every
                       range. A fixed row band could only ever exclude the floor
                       over one narrow range window; a height slab excludes it
                       everywhere.

WHY DEPTH AND NOT LIDAR
The panels are mirrors. LIDAR hit obliquely loses the beam entirely, so a shut
gate reads "no return"; hit square it reads the range to whatever is behind the
car. Neither number describes the panel. The depth camera is fooled the same
way, but it is an IMAGE: the lie has a height, and a height can be cropped.

MISSING RETURNS
The library reports "no return" as 0.0, and mirrors make many. A column with too
little data is UNKNOWN, and unknown is treated as BLOCKED, never as open. Wrong
that way stops the car; wrong the other way drives it into glass.

WHAT A MIRROR CAN STILL DO
A shut panel hit near square reflects the corridor and returns long ranges at
car height -- an opening that is not there. Three things fight it:

  most of the pattern never comes back. Glass at anything but normal incidence
    throws the projector's IR away, so a shut panel is mostly dropouts. A bin
    has to return a real FRACTION of its pixels before it is believed at all,
    and crucially that fraction is measured only over the rows between the
    horizon and the gate plane. Measured over the whole column instead, the
    perfectly well-behaved floor IN FRONT of the panel supplies enough good
    pixels to make a blind column look healthy.

  a doorway has a jamb. At least one edge of the opening must be real structure
    at the gate's own range. An opening with no jamb at either end is the frame
    edge, or open ground, or a mirror face -- not a door.

  the arrow already ruled out the wrong side.

A flawless mirror would still beat all three, because the reflected corridor
lands at car height exactly where a real one would. That is physics, not a bug,
and it is why the panel is approached at a creep with STOP_CM live.
"""

import sys
import math
from collections import deque

import numpy as np
import cv2 as cv

sys.path.insert(1, "../../library")
import racecar_utils as rc_utils



########################################################################################
# MEASURED ONCE WITH A RULER — these are facts about the car, not track tuning
########################################################################################

CAR_WIDTH_CM = 23.0          # widest point, wheel to wheel
CAR_LENGTH_CM = 34.0
WHEELBASE_CM = 25.0          # front axle to rear axle
MAX_STEER_RAD = math.radians(28.0)   # road-wheel angle at full lock

CAM_HEIGHT_CM = 15.0         # lens above the floor. FALLBACK ONLY -- the floor
CAM_PITCH_DEG = 0.0          # fit below measures both of these live
HFOV_DEG = 69.4              # depth stream horizontal FOV (D435 = 69.4)

# The slab of air the car actually sweeps through.
SLAB_LOW_CM = 8.0            # under this is floor, and floor is not an obstacle
SLAB_HIGH_CM = 22.0          # over this passes above the car -- including a
                             # raised gate panel, which is the whole point
SIDE_CLEARANCE_CM = 8.0      # air wanted either side of the car in a doorway

# Where the sign lives, measured off its base. The arrow stands upright on the
# gate's plinth, so it occupies a known, narrow band of height -- and nothing
# else in the hall does. Height is a far stronger filter than an image crop:
# spectators, bins, background props and the panels themselves are all excluded
# by geometry rather than by hoping their hue misses the blue window.
SIGN_LOW_CM = 5.0
SIGN_HIGH_CM = 55.0
SIGN_MAX_RANGE_CM = 350.0

# Derived. Do not hand-set these.
NEEDED_GAP_CM = CAR_WIDTH_CM + 2.0 * SIDE_CLEARANCE_CM
PASS_MARGIN_CM = CAR_LENGTH_CM      # deeper than the gate plane by this = through it
MAX_DOOR_CM = 220.0                 # wider than this is not a doorway, it is a
                                    # phantom off a flat mirror


########################################################################################
# TUNE THESE — the short list
########################################################################################

# --- Blue detection. The only genuinely venue-dependent block in the file. ------------
# The arrow fabric is a tight hue band around H 105-112 (OpenCV H is 0-179).
# Re-probe on the car camera at the venue: white balance shifts the band.
BLUE_HSV = ((100, 60, 80), (118, 255, 255))

# Mirror glare: bright AND colorless. Same trick as YELLOWFORK.
GLARE_HSV = ((0, 0, 225), (179, 70, 255))
GLARE_DILATE_PX = 5

MIN_CONTOUR_AREA = 80        # noise floor only; the commit test is range, not area

# --- Speeds. A choice about how hard to push, not a calibration. ----------------------
CRUISE_SPEED = 0.35
PENDING_SPEED = 0.20         # arrow in range, vote not in yet
APPROACH_SPEED = 0.22        # closing on the doorway
DRIVE_SPEED = 0.40           # through it

# --- Pure pursuit ---------------------------------------------------------------------
LOOKAHEAD_CM = 80.0          # how far down the path to aim. Larger = lazier, smoother
ANGLE_SMOOTHING = 0.5        # fraction of the gap to the new command closed per frame


########################################################################################
# THE REST — derived, or structural. Left alone they should be fine.
########################################################################################

ASPECT_MIN, ASPECT_MAX = 1.2, 4.5    # a horizontal arrow is wide; a rotating one
                                     # falls outside and simply does not vote
TIP_RATIO = 0.75             # tip side outer-20% column mean, over tail side's

VOTE_N, VOTE_K = 5, 3        # K same-direction frames out of the last N
ARROW_COMMIT_CM = 250.0      # close enough to act on the arrow
ARROW_AREA_BACKSTOP = 600    # used ONLY if the arrow returns no depth at all
                             # (dark fabric can swallow the IR pattern)

N_BINS = 64                  # bearing columns the frame is pooled into
ROW_STRIDE = 2               # every other row; halves the cost, changes nothing
MIN_VALID_FRAC = 0.35        # share of a bin's pixels that must come back before
                             # the bin is believed at all. A FRACTION, not a count:
                             # a mostly-holes bin is a mirror throwing the pattern
                             # away, and it stays mostly-holes at any resolution,
                             # whereas any absolute count is cleared by the debris
                             # of a big enough bin. Matte surfaces return >90%;
                             # glass at a glancing angle returns well under 30%.
MIN_SLAB_HITS = 6            # per bin, below this there is no obstacle, just speckle
RANGE_PERCENTILE = 10        # per bin, percentile of slab ranges. Not min(): one
                             # flyer would own the bin

STANDOFF_CM = 110.0          # where to wait when our side is shut. Far enough back
                             # that both door jambs stay inside the FOV
STOP_CM = 45.0               # hard guard, any state
FRONT_SECTOR_DEG = 8.0       # half width of the "straight ahead" window
CENTERED_DEG = 15.0          # doorway bearing this small = lined up
THROUGH_TIME = 2.0           # s of driving to clear the gate once committed
BLIND_FRAC = 0.60            # this share of bins unknown = the frame told us
                             # nothing; creep instead of cruising into it
CALIBRATE_EVERY = 15         # frames between floor fits once one has taken. The
                             # lens does not change height between frames, and the
                             # fit is the most expensive thing in the loop.


########################################################################################
# STATE
########################################################################################

_votes = deque(maxlen=VOTE_N)
_state = "SEEK"
_commit_dir = None
_through_timer = 0.0
_angle = 0.0
_cam_height = CAM_HEIGHT_CM
_cam_pitch = math.radians(CAM_PITCH_DEG)
_calibrated = False
_frames = 0


def reset_blue_arrow():
    """Call from start(), or when entering this track section."""
    global _state, _commit_dir, _through_timer, _angle
    global _cam_height, _cam_pitch, _calibrated, _frames
    _votes.clear()
    _frames = 0
    _state = "SEEK"
    _commit_dir = None
    _through_timer = 0.0
    _angle = 0.0
    _cam_height = CAM_HEIGHT_CM
    _cam_pitch = math.radians(CAM_PITCH_DEG)
    _calibrated = False


########################################################################################
# CAMERA GEOMETRY
########################################################################################

def _intrinsics(shape):
    """(fx, fy, cx, cy) in pixels, from the FOV spec. Square pixels."""
    height, width = shape[:2]
    fx = (width / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
    return fx, fx, (width - 1) / 2.0, (height - 1) / 2.0


def _horizon_row(shape):
    """Image row the horizon falls on. Rays above it never meet the floor."""
    _, fy, _, cy = _intrinsics(shape)
    return cy - fy * math.tan(_cam_pitch)


def calibrate_floor(depth):
    """Measure camera height and pitch off the floor. Returns True if it took.

    Fits a plane to the lower part of the frame. The floor is the only large flat
    surface the camera is guaranteed to see, and its plane gives both unknowns at
    once: distance from the lens is the height, tilt is the pitch. Every result is
    range-checked before it is believed, so a frame filled by a wall -- or by a
    mirror -- leaves the previous calibration standing.
    """
    global _cam_height, _cam_pitch, _calibrated

    height_px, width_px = depth.shape[:2]
    fx, fy, cx, cy = _intrinsics(depth.shape)

    rows = np.arange(int(height_px * 0.60), height_px, 2)
    cols = np.arange(0, width_px, 4)
    z = np.asarray(depth[rows[:, None], cols[None, :]], dtype=np.float64)
    keep = z > 0.0
    if keep.sum() < 200:
        return False

    x = (cols[None, :] - cx) / fx * z
    y = (rows[:, None] - cy) / fy * z
    pts = np.stack([x[keep], y[keep], z[keep]], axis=1)

    # Fit n . p = d as p . w = 1, then normalise. Two rounds of residual
    # rejection so the odd wall or wheel in the bottom corner cannot tilt it.
    w = None
    for _ in range(3):
        try:
            w, *_ = np.linalg.lstsq(pts, np.ones(len(pts)), rcond=None)
        except np.linalg.LinAlgError:
            return False
        norm = float(np.linalg.norm(w))
        if norm <= 1e-9:
            return False
        resid = np.abs(pts.dot(w) - 1.0) / norm
        inliers = resid < 3.0                      # cm from the plane
        if inliers.sum() < 150:
            return False
        pts = pts[inliers]

    norm = float(np.linalg.norm(w))
    normal = w / norm
    dist = 1.0 / norm
    if normal[1] < 0:                              # point it "down" in camera y
        normal, dist = -normal, -dist

    pitch = math.atan2(float(normal[2]), float(normal[1]))
    # Sanity gates. A wall, a mirror, or a bad frame fails at least one.
    if not (5.0 <= dist <= 60.0):
        return False
    if abs(math.degrees(pitch)) > 25.0 or abs(float(normal[0])) > 0.15:
        return False

    if _calibrated:                                # ease in, do not jump
        _cam_height += 0.15 * (dist - _cam_height)
        _cam_pitch += 0.15 * (pitch - _cam_pitch)
    else:
        _cam_height, _cam_pitch = dist, pitch
        _calibrated = True
    return True


########################################################################################
# DEPTH -> PER-BEARING OBSTACLE PROFILE
########################################################################################

def _profile(depth):
    """Collapse the frame into one obstacle range per bearing bin.

    Returns (bearings, edges, ranges, known):
        bearings  deg, bin centres; negative left, positive right
        edges     deg, bin boundaries, length N_BINS + 1
        ranges    cm to the nearest thing IN THE SLAB in that direction;
                  np.inf where the direction is clear
        known     False where too little data came back to say anything

    A bin that is `known` and infinite is open road. A bin that is not `known`
    is a mirror, or shadow, or nothing at all -- and is treated as solid.
    """
    height_px, width_px = depth.shape[:2]
    fx, fy, cx, cy = _intrinsics(depth.shape)

    # Rows above the horizon can only hold slab points very close to the car, so
    # the frame is cut there. Pure speed; the slab test below is what decides.
    top = max(0, int(_horizon_row(depth.shape) - 0.15 * height_px))
    rows = np.arange(top, height_px, ROW_STRIDE)
    if rows.size == 0:
        rows = np.arange(0, height_px, ROW_STRIDE)

    # Odd columns are split between the two edges rather than dropped off the
    # right, which would put a one-sided blind spot where an edge-on gate sits.
    bin_w = max(1, width_px // N_BINS)
    usable = bin_w * N_BINS
    x0 = (width_px - usable) // 2

    z = np.asarray(depth[rows[:, None], np.arange(x0, x0 + usable)[None, :]],
                   dtype=np.float32)
    q = ((rows - cy) / fy).astype(np.float32)[:, None]
    y_down = q * z
    cos_p, sin_p = math.cos(_cam_pitch), math.sin(_cam_pitch)
    h_cm = _cam_height - (y_down * cos_p + z * sin_p)
    ground = z * cos_p - y_down * sin_p

    valid = z > 0.0
    in_slab = valid & (ground > 0.0) & (h_cm >= SLAB_LOW_CM) & (h_cm <= SLAB_HIGH_CM)

    shape = (rows.size, N_BINS, bin_w)
    # Valid-pixel count per (row, bin). Kept row-resolved because whether a bin
    # returned usable data can only be judged over the rows that decide the
    # question -- see _known below.
    row_valid = valid.reshape(shape).sum(axis=2)

    # Percentile of the in-slab ranges per bin, taken over the LIVE entries only.
    # Padding with zeros instead would drag every percentile to zero and call the
    # whole frame solid.
    pool = np.where(in_slab, ground, np.inf).reshape(shape).transpose(1, 0, 2)
    pool = pool.reshape(N_BINS, -1)
    hits = np.isfinite(pool).sum(axis=1)
    pool.sort(axis=1)
    idx = np.minimum((hits * (RANGE_PERCENTILE / 100.0)).astype(np.int32),
                     np.maximum(hits - 1, 0))
    ranges = pool[np.arange(N_BINS), idx]
    ranges[hits < MIN_SLAB_HITS] = np.inf          # speckle is not an obstacle

    # Bearings straight off the intrinsics. atan, not a linear pixel-to-degree
    # scale, whose error is worst exactly at the frame edges where a gate first
    # appears.
    def bearing_of(px):
        return np.degrees(np.arctan((px - cx) / fx))

    bearings = bearing_of(x0 + (np.arange(N_BINS) + 0.5) * bin_w).astype(np.float32)
    edges = bearing_of(x0 + np.arange(N_BINS + 1) * bin_w).astype(np.float32)
    return bearings, edges, ranges.astype(np.float32), row_valid, rows, bin_w


def _floor_row(shape, ground_cm):
    """Image row where the floor at this ground range appears.

    Nearer floor sits lower in the frame, so this row rises toward the horizon as
    ground_cm grows.
    """
    _, fy, _, cy = _intrinsics(shape)
    c, s = math.cos(_cam_pitch), math.sin(_cam_pitch)
    y_down = _cam_height * c - ground_cm * s
    z = _cam_height * s + ground_cm * c
    if z <= 1e-6:
        return float(shape[0])
    return cy + fy * y_down / z


def _known(row_valid, rows, shape, gate_cm, bin_w):
    """Which bins returned enough data to be believed, over the rows that matter.

    The rows that matter run from the top of the window down to where the floor
    meets the gate plane. Below that line a column is looking at the floor
    BETWEEN the car and the gate, which comes back cleanly off any surface and
    says nothing about whether the gate is open.

    Measuring over the whole column instead is what lets a mirror pass: a shut
    mirrored panel is mostly dropouts, but the well-behaved floor in front of it
    is plenty of valid pixels, and the two average out to a healthy-looking
    column that is in fact blind exactly where it needed to see.
    """
    if np.isfinite(gate_cm):
        cutoff = _floor_row(shape, gate_cm)
    else:
        cutoff = float(shape[0])
    use = rows <= cutoff
    if use.sum() < 4:                              # gate underfoot: nothing to judge
        use = np.ones(rows.shape, bool)
    counted = row_valid[use].sum(axis=0)
    return counted >= MIN_VALID_FRAC * use.sum() * bin_w


def _analyse(depth):
    """One frame of depth -> everything the driving code asks about it.

    Returns (bearings, edges, ranges, known, gate). Ordered this way because the
    gate plane has to be found from the slab hits BEFORE bin validity can be
    judged: validity is only meaningful over the rows in front of that plane.
    """
    bearings, edges, ranges, row_valid, rows, bin_w = _profile(depth)
    gate = _gate_range(bearings, ranges)
    known = _known(row_valid, rows, depth.shape, gate, bin_w)
    return bearings, edges, ranges, known, gate


def _blocked(ranges, known, limit_cm):
    """Bins that hold something solid closer than limit_cm. Unknown counts."""
    return (~known) | (ranges <= limit_cm)


def _lateral(bearings, ranges):
    """Sideways offset of each bin's obstacle from the car's centreline, cm."""
    return ranges * np.tan(np.radians(bearings))


def _gate_range(bearings, ranges):
    """Range to the nearest structure IN THE CAR'S PATH, cm. inf if none.

    This is the gate plane, and everything deeper than it by a car length counts
    as through it -- so the open/closed threshold sets itself from the scene
    instead of being a number guessed at the venue.

    "In the path" is a lateral test, not an angular one. An angular sector cannot
    express it: a corridor wall a fixed distance to the side keeps entering any
    fixed sector at a fixed range (90 cm out, a 25 deg sector picks it up at 193
    cm), so the walls get mistaken for the gate and every gate beyond that range
    becomes invisible. Lateral offset does not care how far away the wall is.
    """
    usable = np.isfinite(ranges)
    if not usable.any():
        return float("inf")
    in_path = usable & (np.abs(_lateral(bearings, ranges)) <= NEEDED_GAP_CM)
    return float(ranges[in_path].min()) if in_path.any() else float("inf")


def _widest_run(flags):
    """(start, length) of the longest True run. Length 0 if none."""
    best_start, best_len, run = 0, 0, None
    for i, f in enumerate(flags):
        if f and run is None:
            run = i
        elif not f and run is not None:
            if i - run > best_len:
                best_start, best_len = run, i - run
            run = None
    if run is not None and len(flags) - run > best_len:
        best_start, best_len = run, len(flags) - run
    return best_start, best_len


def _find_door(bearings, edges, ranges, known, gate, side, require_jambs=True):
    """Widest passable opening on `side`. Returns (found, bearing_deg, width_cm).

    An opening qualifies only if it is wider than the car plus clearance,
    measured in CM at the gate's own range, and if at least one JAMB is visible
    -- a bin of real structure right beside it, at the gate plane. An opening
    with no jamb at either end is not a doorway; it is the frame edge, or open
    ground, or a flat mirror returning long ranges across its whole face.

    The returned bearing is where to AIM, which is not always the middle of what
    can be seen. With both jambs visible the middle is right. With only one, the
    opening runs off the edge of the frame and its apparent centre is somewhere
    out in the unseen; the car hugs the jamb it can actually measure, at the
    minimum safe passing offset.
    """
    if not np.isfinite(gate):
        return False, 0.0, 0.0                     # open ground, no gate here

    solid = _blocked(ranges, known, gate + PASS_MARGIN_CM)
    open_bins = ~solid

    # Bridge single-bin dropouts so one bad column cannot split one real doorway
    # into two runs that each fail the width test.
    bridged = open_bins.copy()
    bridged[1:-1] |= open_bins[:-2] & open_bins[2:]

    if side == "LEFT":
        bridged &= bearings <= 10.0
    elif side == "RIGHT":
        bridged &= bearings >= -10.0

    start, length = _widest_run(bridged)
    if length == 0:
        return False, 0.0, 0.0
    end = start + length

    lo, hi = float(edges[start]), float(edges[end])
    x_lo = gate * math.tan(math.radians(lo))
    x_hi = gate * math.tan(math.radians(hi))
    width_cm = x_hi - x_lo

    # A jamb is structure in the bin next door, tested against the UNMASKED
    # solid map -- otherwise the side mask's own edge would pass for a wall.
    jamb_lo = start > 0 and bool(solid[start - 1])
    jamb_hi = end < len(solid) and bool(solid[end])

    half = NEEDED_GAP_CM / 2.0
    if jamb_lo and jamb_hi:
        if width_cm > MAX_DOOR_CM:
            return False, 0.0, width_cm            # too wide to be a door
        target_x = 0.5 * (x_lo + x_hi)
    elif jamb_lo:
        target_x = x_lo + half                     # hug the one edge we can see
    elif jamb_hi:
        target_x = x_hi - half
    elif require_jambs:
        return False, 0.0, width_cm                # no jamb either side
    else:
        target_x = 0.5 * (x_lo + x_hi)

    bearing = math.degrees(math.atan2(target_x, gate))
    return width_cm >= NEEDED_GAP_CM, bearing, width_cm


def _front_range(bearings, ranges, known):
    """Nearest slab obstacle straight ahead, cm. inf if clear, 0.0 if unknown."""
    sector = np.abs(bearings) <= FRONT_SECTOR_DEG
    if not known[sector].any():
        return 0.0
    vals = ranges[sector & known]
    return float(vals.min()) if vals.size else float("inf")


def _corridor_bearing(bearings, ranges, known):
    """Bearing that re-centres the car between whatever is left and right of it.

    Works on lateral offsets in cm, not on raw ranges: a wall alongside the car
    is close in x and far in z, and comparing z would barely notice it.
    """
    usable = known & np.isfinite(ranges)
    if not usable.any():
        return 0.0
    lateral = ranges * np.tan(np.radians(bearings))
    left = lateral[usable & (lateral < 0.0)]
    right = lateral[usable & (lateral > 0.0)]
    if left.size == 0 or right.size == 0:
        return 0.0                                 # one wall only: steering blind
                                                   # is worse than going straight
    centre = 0.5 * (float(left.max()) + float(right.min()))
    return math.degrees(math.atan2(centre, LOOKAHEAD_CM))


def _pursue(bearing_deg, range_cm=None):
    """Steering command for a target at a bearing. Pure pursuit, Ackermann.

    Curvature comes out of the geometry and turns into a servo command by
    dividing by the car's real lock. That is why there is no steering gain in
    this file to get wrong.
    """
    global _angle
    look = LOOKAHEAD_CM if range_cm is None else max(LOOKAHEAD_CM, min(range_cm, 300.0))
    alpha = math.radians(bearing_deg)
    curvature = 2.0 * math.sin(alpha) / look
    steer = math.atan(WHEELBASE_CM * curvature)
    target = rc_utils.clamp(steer / MAX_STEER_RAD, -1.0, 1.0)
    _angle += ANGLE_SMOOTHING * (target - _angle)
    return rc_utils.clamp(_angle, -1.0, 1.0)


########################################################################################
# ARROW READING
########################################################################################

def _sign_gate(depth, color_shape):
    """Pixels that could belong to the sign, on height above the floor.

    Pixels with NO depth are LEFT IN, which is the opposite of how the obstacle
    code treats them, and deliberately so. There the question is "might this hurt
    the car", and silence has to mean yes. Here the question is "might this be
    the sign", and throwing away every pixel the projector failed on would lose a
    dark or angled arrow completely. Nothing is driven off this mask; it only
    decides where to look for blue.
    """
    fx, fy, cx, cy = _intrinsics(depth.shape)
    z = np.asarray(depth, dtype=np.float32)
    q = ((np.arange(depth.shape[0]) - cy) / fy).astype(np.float32)[:, None]
    y_down = q * z
    c, s = math.cos(_cam_pitch), math.sin(_cam_pitch)
    h_cm = _cam_height - (y_down * c + z * s)
    ground = z * c - y_down * s

    gate = (z <= 0.0) | ((h_cm >= SIGN_LOW_CM) & (h_cm <= SIGN_HIGH_CM) &
                         (ground <= SIGN_MAX_RANGE_CM))
    if gate.shape != tuple(color_shape[:2]):       # streams at different sizes
        gate = cv.resize(gate.astype(np.uint8), (color_shape[1], color_shape[0]),
                         interpolation=cv.INTER_NEAREST).astype(bool)
    return gate


def _read_arrow(image, depth=None):
    """Returns (direction, area, range_cm). direction is "LEFT"/"RIGHT"/None.

    None covers: no blue, too small, not horizontal-arrow-shaped, or an
    ambiguous taper. An abstaining frame costs nothing at 30 fps; a wrong frame
    costs a vote. range_cm is inf when the arrow returned no depth.
    """
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    blue = cv.inRange(hsv, BLUE_HSV[0], BLUE_HSV[1])
    glare = cv.inRange(hsv, GLARE_HSV[0], GLARE_HSV[1])
    kernel = np.ones((GLARE_DILATE_PX, GLARE_DILATE_PX), np.uint8)
    mask = cv.bitwise_and(blue, cv.bitwise_not(cv.dilate(glare, kernel)))
    if depth is not None:
        mask = cv.bitwise_and(mask, _sign_gate(depth, image.shape).astype(np.uint8) * 255)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
    if contour is None:
        return None, 0, float("inf")

    x, y, w, h = cv.boundingRect(contour)
    area = rc_utils.get_contour_area(contour)
    if not (ASPECT_MIN <= w / max(h, 1) <= ASPECT_MAX):
        return None, area, float("inf")            # rotating / not an arrow now

    # Column-width profile over the blob: the tip end is the skinny end.
    blob = np.zeros(mask.shape, np.uint8)
    cv.drawContours(blob, [contour], -1, 255, cv.FILLED)
    cols = (blob[y:y + h, x:x + w] > 0).sum(axis=0).astype(np.float32)
    k = max(3, int(0.2 * w))
    lmean, rmean = float(cols[:k].mean()), float(cols[-k:].mean())

    direction = None
    if lmean < TIP_RATIO * rmean:
        direction = "LEFT"
    elif rmean < TIP_RATIO * lmean:
        direction = "RIGHT"

    # How far away it is, straight off the depth stream. Contour AREA would say
    # the same thing only if the arrow's physical size, the resolution and the
    # lens were all fixed and known -- range needs none of that.
    range_cm = float("inf")
    if depth is not None:
        cu, cv_row = x + w // 2, y + h // 2
        half = max(3, min(w, h) // 4)
        patch = depth[max(0, cv_row - half):cv_row + half,
                      max(0, cu - half):cu + half]
        live = patch[patch > 0.0]
        if live.size >= 10:
            range_cm = float(np.median(live))
    return direction, area, range_cm


########################################################################################
# MAIN — call this once per frame
########################################################################################

def blue_arrow():
    """One frame of the fork section. Returns (speed, angle), or None when done."""
    global _state, _commit_dir, _through_timer, _angle, _frames

    if _state == "DONE":
        return None

    # Advance the clock before anything can return early, so a dropped frame
    # cannot strand the car halfway through the gate.
    if _through_timer > 0.0:
        _through_timer -= rc.get_delta_time()

    depth = rc.camera.get_depth_image()
    if depth is None:
        return 0.0, _angle

    _frames += 1
    if not _calibrated or _frames % CALIBRATE_EVERY == 0:
        calibrate_floor(depth)
    bearings, edges, ranges, known, gate = _analyse(depth)
    front = _front_range(bearings, ranges, known)
    too_close = 0.0 < front < STOP_CM

    # -----------------------------------------------------------------
    # SEEK -- run the corridor, read the arrow, latch a side
    # -----------------------------------------------------------------
    if _state == "SEEK":
        area, arrow_cm = 0, float("inf")
        image = rc.camera.get_color_image()
        if image is not None:
            direction, area, arrow_cm = _read_arrow(image, depth)
            _votes.append(direction)

        in_range = arrow_cm <= ARROW_COMMIT_CM
        if not np.isfinite(arrow_cm):
            in_range = area >= ARROW_AREA_BACKSTOP   # arrow gave no depth at all

        if in_range:
            for side in ("LEFT", "RIGHT"):
                if sum(1 for v in _votes if v == side) >= VOTE_K:
                    _commit_dir = side
                    _state = "ALIGN"
                    _votes.clear()
                    print("[fork] arrow says {} at {:.0f}cm -> depth takes over".format(
                        side, arrow_cm))
                    break

        if _state == "SEEK":
            speed = PENDING_SPEED if in_range else CRUISE_SPEED
            # A frame this empty is a mirror throwing the pattern away, not clear
            # road. Creep rather than cruise into it blind.
            if float((~known).mean()) >= BLIND_FRAC:
                speed = min(speed, PENDING_SPEED)
            if too_close:
                speed = 0.0
            return speed, _pursue(_corridor_bearing(bearings, ranges, known))

    # -----------------------------------------------------------------
    # ALIGN -- aim at the middle of the doorway on the arrow's side
    # -----------------------------------------------------------------
    if _state == "ALIGN":
        found, bearing, width = _find_door(bearings, edges, ranges, known, gate,
                                           _commit_dir)

        if not found:
            # Shut, or nothing here yet. Close to standoff and wait -- the gate
            # runs a ~10 s cycle, so waiting costs seconds and guessing costs
            # the run.
            if np.isfinite(gate) and gate <= STANDOFF_CM:
                return 0.0, _angle
            return (0.0 if too_close else APPROACH_SPEED), \
                _pursue(_corridor_bearing(bearings, ranges, known))

        angle = _pursue(bearing, gate)
        if abs(bearing) <= CENTERED_DEG and gate <= STANDOFF_CM:
            _through_timer = THROUGH_TIME
            _state = "THROUGH"
            print("[fork] door {:+.0f}deg, {:.0f}cm wide, {:.0f}cm out -> through".format(
                bearing, width, gate))
            return DRIVE_SPEED, angle
        return (0.0 if too_close else APPROACH_SPEED), angle

    # -----------------------------------------------------------------
    # THROUGH -- drive out, still tracking the middle of the opening
    # -----------------------------------------------------------------
    if _state == "THROUGH":
        if _through_timer <= 0.0:
            _state = "DONE"
            return None
        # Jambs leave the FOV as the car enters the doorway, so stop demanding
        # them here; the approach already proved this was a real opening.
        found, bearing, _ = _find_door(bearings, edges, ranges, known, gate,
                                       _commit_dir, require_jambs=False)
        angle = _pursue(bearing) if found else _angle
        return (0.0 if too_close else DRIVE_SPEED), angle

    return 0.0, _angle


########################################################################################
# DEBUG — drives nothing, prints what the car believes
########################################################################################

def blue_arrow_debug():
    """Park the car at the standoff point and watch a full gate cycle.

    cam        height/pitch the floor fit measured. If height is far off a ruler
               reading, the floor is not being seen -- check the lower frame.
    gate       range to the structure ahead
    door       width of the opening, in cm, at the gate plane. Compare against
               the car: NEEDED_GAP_CM is """ + "{:.0f}".format(NEEDED_GAP_CM) + """ cm.
    unknown    share of bearing bins with too little data. High while the gate is
               DOWN is expected -- that is the mirror. High while it is UP means
               the opening is not being seen at all.
    """
    depth = rc.camera.get_depth_image()
    if depth is None:
        print("[fork] no depth")
        return

    ok = calibrate_floor(depth)
    bearings, edges, ranges, known, gate = _analyse(depth)
    found, bearing, width = _find_door(bearings, edges, ranges, known, gate,
                                       _commit_dir)

    arrow = "-"
    image = rc.camera.get_color_image()
    if image is not None:
        direction, area, arrow_cm = _read_arrow(image, depth)
        arrow = "{}@{:.0f}cm".format(direction or "-", arrow_cm)

    print(
        "[fork] {} | arrow {:>12s} | side {:5s} | cam {:.1f}cm/{:+.1f}deg{} "
        "| gate {:6.0f}cm | door {:+5.1f}deg {:5.0f}cm "
        "| front {:6.0f}cm | unknown {:4.0%}".format(
            "OPEN " if found else "BLOCK", arrow, _commit_dir or "any",
            _cam_height, math.degrees(_cam_pitch), "" if ok else "?",
            gate, bearing, width,
            _front_range(bearings, ranges, known), float((~known).mean()),
        )
    )


########################################################################################
# STANDALONE HARNESS — delete when merging into your multi-section script
########################################################################################

# Set True to park the car and only print. Do this first, once, to confirm the
# floor fit agrees with a ruler.
DEBUG_ONLY = True




########################################################################################
# race.py SECTION CONTRACT ADAPTER
# fork()/fork_reset()/fork_ready()/fork_state -- rc is injected by race.py.
########################################################################################

fork_state = "SEEK"


def fork_reset():
    global fork_state
    reset_blue_arrow()
    fork_state = "SEEK"


def fork_ready(image=None):
    """Blue arrow in color view ends the transit leg. Depth-only SEEK would
    also cruise, but engaging on sight keeps the tuned transit cruise in
    charge until the prop is actually there -- and if the depth stream is
    dead the car parks in SEEK, so do not hand over early."""
    if image is None:
        image = rc.camera.get_color_image()
    if image is None:
        return False
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, np.array(BLUE_HSV[0], np.uint8),
                      np.array(BLUE_HSV[1], np.uint8))
    return cv.countNonZero(mask) >= MIN_CONTOUR_AREA


def fork():
    global fork_state
    out = blue_arrow()
    fork_state = _state
    return out
