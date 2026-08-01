"""
MIT BWSI Autonomous RACECAR -- Elevator section
(merged: team's field-tested WORKING ELEVATOR (c9d22e7) state machine +
 race.py section contract + green-line entry)

State flow:
    Green       green line follow          -> Approach   when the model sees the sign
    Approach    PID-steer at the sign      -> Reading    at AT_DOOR_FRAC of the frame
                (sign lost > SIGN_LOST_TIME -> back to Green)
    Reading     stopped, watch the sign    -> Entering   when it reads GO
    Entering    drive in, steer at sign    -> inElevator when area grows by
                (STOP mid-entry = hold)                  INSIDE_AREA_RATIO from the
                                                         door (+ ENTER_MIN_TIME guard)
    inElevator  stopped, ride              -> WaitToExit when countdown hits 0
    WaitToExit  hardcoded pause            -> Exiting    after EXIT_WAIT_TIME
    Exiting     drive out                  -> Done
    Done        release control

Call elevator() once per frame. It returns (speed, angle), or None once Done.
elevator_ready() ends the transit leg (green line in the floor ROI, or the
sign already in model view).

Field-tested facts baked in (do not "fix" back):
  * ELEVATOR_RIDE_TIME 10.5 s -- measured on the rig, not 40.
  * "Inside" is door_area x INSIDE_AREA_RATIO, never an absolute px count:
    the absolute test could be already-true at the door, which parked the
    car outside for the whole ride. ENTER_MIN_TIME guards the same failure.
  * The LIDAR exit gate was dropped (unreliable on the rig) for a plain
    EXIT_WAIT_TIME pause.
  * gostop_v1 is the ONLY model that reads this LED text panel (99-100%
    detection, median conf ~0.89). The 9-class road-sign models emit noise
    on it. Model loading + TPU diagnosis live in gostop_tpu.py; if no model
    loads, sign reading falls back to HSV (GREEN->GO, RED->STOP).
"""

import cv2 as cv
import numpy as np
import racecar_utils as rc_utils

import gostop_tpu as gt


# =====================================================================
#  TUNING -- GREEN LINE ENTRY (race integration)
# =====================================================================

GLARE_V_MIN          = 235
GLARE_S_MAX          = 45
MASK_CLOSE_PX        = 7

GREEN_LOW            = np.array([40,  70,  50])
GREEN_HIGH           = np.array([85, 255, 255])
RED_LOW_A            = np.array([0,   100, 100])
RED_HIGH_A           = np.array([10,  255, 255])
RED_LOW_B            = np.array([170, 100, 100])
RED_HIGH_B           = np.array([179, 255, 255])

LINE_ROI_TOP         = 300    # floor strip used for line following
LINE_ROI_BOTTOM      = 480
SIGN_ROI_TOP         = 60     # HSV-fallback sign search band
SIGN_ROI_BOTTOM      = 300

SCAN_ROWS            = 12
ROW_MIN_PX           = 6
MIN_ROWS             = 4
LOOKAHEAD            = 0.40
MIN_CONTOUR_AREA     = 300
LINE_KP              = 1.30
GREEN_SPEED          = 0.40

# --- reference line follower (ported from the proven blue-line racer) ---
# Bottom crop + half-res + largest-contour-center PD. Deliberately simple:
# no glare mask, no morphology, no polynomial fit -- the S floor in the HSV
# band rejects glare and the contour center is robust to a ragged line.
LINE_CROP_TOP        = 360    # bottom of the frame
LINE_MIN_AREA        = 300    # px at HALF resolution -- the filter, small
LINE_PD_KP           = 0.5
LINE_PD_KD           = 1.2    # d_error is one CAMERA frame apart (frame
                              # dedup guarantees it), so no dt math
LINE_SLOW            = 0.8    # fraction of speed shed per unit |angle|
LINE_LOST_SPEED      = 0.25   # lost: slow and HOLD the last steer, never park

READY_MIN_AREA       = 700    # green px (at HALF res, bottom crop) needed to
                              # end the wall-follow leg. The line is ~1900 px
                              # there, so 700 is "clearly on the line" without
                              # being so big that nothing ever triggers.

SIGN_ENGAGE_AREA     = 2500   # sign px before the sign is allowed to (a) end
                              # the transit or (b) pull the car off the line
                              # into Approach. A 600 px GO across the hall
                              # once dragged the car into a 15 s crawl.

VERBOSE              = True   # per-frame telemetry line (the team's primary
                              # tuning tool); flip off for the race


# =====================================================================
#  TUNING -- ELEVATOR (team values, field-tested; verbatim from c9d22e7)
# =====================================================================

AT_DOOR_FRAC         = 0.039   # "at the door" as a fraction of the frame
AT_DOOR_AREA         = 12000   # fallback only, if the frame size is unknown

INSIDE_AREA_RATIO    = 10      # sign must grow by this factor from the door
INSIDE_AREA          = 30000   # fallback only, if no door area was captured
ENTER_MIN_TIME       = 0.8     # sec of actual driving before "inside" can fire

SIGN_KP              = 0.90
SIGN_KI              = 0.00
SIGN_KD              = 0.12
SIGN_MAX_STEER       = 0.85
APPROACH_SPEED       = 0.20
SIGN_LOST_TIME       = 1.0     # sec sign may vanish before Approach gives up

ENTERING_SPEED       = 0.20
SLOW_ENTERING_SPEED  = 0.10
ENTER_TIMEOUT        = 5.0
STOP_ON_STOP         = True
ELEVATOR_RIDE_TIME   = 10.5    # MEASURED on the rig
EXITING_SPEED        = 0.20
EXITING_ANGLE        = -0.05
EXIT_DRIVE_TIME      = 2.2
EXIT_WAIT_TIME       = 0.5     # plain pause after the ride (LIDAR gate dropped)


# =====================================================================
#  STATE -- persists between calls
# =====================================================================

elevator_state   = "Green"
elevator_countdown = ELEVATOR_RIDE_TIME
ENTERING_TIMER   = 0.0
EXITING_TIMER    = 0.0
_clock           = 0.0
_angle           = 0.0     # line-follow steering memory
_sign_angle      = 0.0     # sign-PID steering memory
_last_seen       = -999.0
_err_prev        = 0.0
_err_sum         = 0.0
_door_area       = 0.0
_peak_area       = 0.0
_exit_wait_timer = 0.0
_line_err_prev   = 0.0     # reference-PD error memory (per camera frame)
_last_fid        = None    # id() of the last processed camera frame
_pend_dt         = 0.0     # dt banked from duplicate frames (timers stay real)
_last_cmd        = (0.0, 0.0)
_preflighted     = False
_debug           = {}


def elevator_reset():
    """Re-arm the whole section. Call from start()."""
    global elevator_state, elevator_countdown, ENTERING_TIMER, EXITING_TIMER
    global _clock, _angle, _sign_angle, _last_seen, _err_prev, _err_sum
    global _door_area, _peak_area, _exit_wait_timer, _line_err_prev
    global _last_fid, _pend_dt, _last_cmd
    _line_err_prev = 0.0
    _last_fid = None
    _pend_dt = 0.0
    _last_cmd = (0.0, 0.0)
    elevator_state = "Green"
    elevator_countdown = ELEVATOR_RIDE_TIME
    ENTERING_TIMER = 0.0
    EXITING_TIMER = 0.0
    _clock = 0.0
    _angle = 0.0
    _sign_angle = 0.0
    _last_seen = -999.0
    _err_prev = 0.0
    _err_sum = 0.0
    _door_area = 0.0
    _peak_area = 0.0
    _exit_wait_timer = 0.0
    _preflight()


def _preflight():
    """The team's model/TPU preflight report, printed once per process."""
    global _preflighted
    if _preflighted:
        return
    _preflighted = True
    print("=" * 62)
    print("[elevator] PREFLIGHT")
    path = gt._find_model()
    print("  model file : {}".format(path or "NOT FOUND"))
    tpus = gt.list_tpus()
    if tpus is None:
        print("  edge tpus  : cannot enumerate (pycoral not importable)")
    else:
        print("  edge tpus  : {} found{}".format(
            len(tpus), "  " + str(tpus) if tpus else "  <-- none plugged in"))
    d = gt._get_detector()
    print("  backend    : {}".format(d.backend or "none"))
    if d.error:
        print("  error      : {}".format(d.error))
    if not d.ok:
        print("  RESULT     : model DISABLED -- sign reading falls back to HSV")
        if "delegate" in (d.error or "").lower() or "edgetpu" in (d.error or "").lower():
            for line in gt.tpu_diagnosis():
                print("    " + line)
    elif "cpu" in (d.backend or ""):
        print("  RESULT     : running on the CPU (slow!), not the Edge TPU")
    else:
        print("  RESULT     : running on the Edge TPU")
    print("=" * 62)


def elevator_debug():
    return dict(_debug)


def elevator_ready(image=None):
    """Green line in the floor ROI, or the sign already in view -- either
    ends the transit leg. Without this the line follow would stop the car
    dead in a corridor that has no line yet."""
    if image is None:               # dispatcher passes its own cheap frame
        image = rc.camera.get_color_image()
    if image is None:
        return False
    # CHEAP ON PURPOSE: half-res bottom crop, one inRange, one countNonZero.
    # This runs during the wall-follow leg, and wf_lite's model twin needs a
    # steady 60 Hz loop -- calling the Coral model here (~30 ms) dropped the
    # loop to 43 Hz and visibly wrecked its cornering. No model, no contours.
    small = cv.resize(image[LINE_CROP_TOP:], None, fx=0.5, fy=0.5,
                      interpolation=cv.INTER_NEAREST)
    mask = cv.inRange(cv.cvtColor(small, cv.COLOR_BGR2HSV), GREEN_LOW, GREEN_HIGH)
    n = cv.countNonZero(mask)
    _debug["ready_px"] = n
    return n >= READY_MIN_AREA


def _goto(new_state):
    global elevator_state, ENTERING_TIMER, EXITING_TIMER, _err_prev, _err_sum
    if new_state == elevator_state:
        return
    print("[elevator] {} -> {}".format(elevator_state, new_state))
    elevator_state = new_state
    ENTERING_TIMER = 0.0
    EXITING_TIMER = 0.0
    _err_prev = 0.0
    _err_sum = 0.0


# =====================================================================
#  VISION -- line helpers (race integration) + sign detection wrapper
# =====================================================================

def _hsv_and_glare(image, top, bottom):
    cropped = rc_utils.crop(image, (top, 0), (bottom, rc.camera.get_width()))
    hsv = cv.cvtColor(cropped, cv.COLOR_BGR2HSV)
    _, s, v = cv.split(hsv)
    glare = cv.bitwise_and(cv.inRange(v, GLARE_V_MIN, 255),
                           cv.inRange(s, 0, GLARE_S_MAX))
    return hsv, cv.bitwise_not(glare)


def _color_mask(hsv, valid, low, high, low2=None, high2=None):
    m = cv.inRange(hsv, low, high)
    if low2 is not None:
        m = cv.bitwise_or(m, cv.inRange(hsv, low2, high2))
    m = cv.bitwise_and(m, valid)
    return cv.morphologyEx(m, cv.MORPH_CLOSE,
                           np.ones((MASK_CLOSE_PX, MASK_CLOSE_PX), np.uint8))


def _largest_blob(mask, min_area):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest = rc_utils.get_largest_contour(contours, min_area)
    if largest is None:
        return None, 0, None
    blob = np.zeros_like(mask)
    cv.drawContours(blob, [largest], -1, 255, cv.FILLED)
    center = rc_utils.get_contour_center(largest)
    return blob, cv.contourArea(largest), (None if center is None else center[1])


def _follow_line(blob, speed):
    """Second-degree polynomial line follow over a clean single-blob mask."""
    global _angle
    if blob is None:
        return (0.0, _angle)
    h, w = blob.shape
    ys, xs_center = [], []
    for i in range(SCAN_ROWS):
        y = int(h - 1 - (i + 0.5) * h / SCAN_ROWS)
        if y < 0:
            break
        xs = np.flatnonzero(blob[y])
        if xs.size < ROW_MIN_PX:
            continue
        ys.append(float(y))
        xs_center.append(0.5 * (float(xs[0]) + float(xs[-1])))
    if len(ys) < MIN_ROWS:
        return (0.0, _angle)
    fit = np.poly1d(np.polyfit(ys, xs_center, 2))
    y_eval = max(ys) - LOOKAHEAD * (max(ys) - min(ys))
    _angle = rc_utils.clamp(
        LINE_KP * rc_utils.remap_range(float(fit(y_eval)), 0, w, -1, 1), -1, 1)
    return (speed, _angle)


def _find_line_x(image):
    """Reference follower's line finder: bottom crop, half res, largest green
    contour center. Returns full-scale column x, or None."""
    small = cv.resize(image[LINE_CROP_TOP:], None, fx=0.5, fy=0.5,
                      interpolation=cv.INTER_NEAREST)
    hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = rc_utils.get_largest_contour(contours, LINE_MIN_AREA)
    if contour is None:
        return None
    center = rc_utils.get_contour_center(contour)
    return None if center is None else center[1] * 2


def _detect(image):
    """(label, area, center_x, conf): the Coral model, or HSV fallback.

    label is "GO", "STOP", or None -- same contract either way, so the state
    machine never knows which eye it is using.
    """
    d = gt._get_detector()
    if d.ok:
        return d.detect(image)
    hsv, valid = _hsv_and_glare(image, SIGN_ROI_TOP, SIGN_ROI_BOTTOM)
    g_mask = _color_mask(hsv, valid, GREEN_LOW, GREEN_HIGH)
    r_mask = _color_mask(hsv, valid, RED_LOW_A, RED_HIGH_A, RED_LOW_B, RED_HIGH_B)
    _, g_area, g_x = _largest_blob(g_mask, 150)
    _, r_area, r_x = _largest_blob(r_mask, 150)
    if g_area <= 0 and r_area <= 0:
        return None, 0.0, None, 0.0
    if g_area >= r_area:
        return "GO", g_area, g_x, 1.0
    return "STOP", r_area, r_x, 1.0


# =====================================================================
#  STEERING AT THE SIGN (team PID, verbatim)
# =====================================================================

def _at_door_area():
    try:
        w, h = rc.camera.get_width(), rc.camera.get_height()
        if w and h:
            return AT_DOOR_FRAC * float(w) * float(h)
    except Exception:
        pass
    return float(AT_DOOR_AREA)


def _sign_steer(center_x, dt):
    global _err_prev, _err_sum, _sign_angle
    width = rc.camera.get_width()
    error = rc_utils.remap_range(center_x, 0, width, -1, 1)
    _err_sum += error * dt
    derivative = (error - _err_prev) / dt if dt > 0 else 0.0
    _err_prev = error
    _sign_angle = rc_utils.clamp(
        SIGN_KP * error + SIGN_KI * _err_sum + SIGN_KD * derivative,
        -SIGN_MAX_STEER, SIGN_MAX_STEER)
    return _sign_angle


# =====================================================================
#  MAIN STATE MACHINE (team logic, converted to the section contract)
# =====================================================================

def elevator():
    """One frame of the elevator section. Returns (speed, angle), or None.

    The camera publishes ~23-30 fps while update() runs at 60 Hz, so half
    the calls see the SAME image -- reprocessing runs the Coral model twice
    per real frame for nothing. Duplicates return the last command; their dt
    is BANKED and added to the next real frame, so every timer (ride
    countdown, entering, sign memory) still runs on wall time.
    """
    global _last_fid, _pend_dt, _last_cmd

    if elevator_state == "Done":
        return None

    dt = rc.get_delta_time()
    try:
        image = rc.camera.get_color_image_no_copy()
    except AttributeError:
        image = rc.camera.get_color_image()
    if image is None:
        return (0.0, _angle)
    if id(image) == _last_fid:
        _pend_dt += dt
        return _last_cmd
    _last_fid = id(image)

    out = _elevator_step(image, dt + _pend_dt)
    _pend_dt = 0.0
    _last_cmd = out if out is not None else (0.0, 0.0)
    return out


def _elevator_step(image, dt):
    global elevator_countdown, ENTERING_TIMER, EXITING_TIMER
    global _clock, _last_seen, _door_area, _peak_area, _exit_wait_timer

    _clock += dt

    label, area, center_x, conf = _detect(image)
    if label is not None:
        _last_seen = _clock
    lost_for = _clock - _last_seen
    if area > _peak_area:
        _peak_area = area

    need = (_door_area * INSIDE_AREA_RATIO) if _door_area > 0 else INSIDE_AREA
    _debug.update(state=elevator_state, sign=label, area=int(area),
                  conf=round(conf, 2), countdown=round(elevator_countdown, 1),
                  door=int(_at_door_area()), need=int(need))
    if VERBOSE:
        print("[{:.1f}s] state={:<10} sign={:<5} area={:>6} conf={:.2f} "
              "line_x={:>4} door>={:.0f} need>={:.0f} peak={:.0f}".format(
                  _clock, elevator_state, str(label), int(area), conf,
                  int(_debug.get("line_x", -1) or -1), _at_door_area(), need, _peak_area))

    # ---- Green: follow the line until the sign is close enough to matter ----
    # Reference-follower port: largest-contour-center PD on the bottom crop.
    if elevator_state == "Green":
        if label is not None and area >= SIGN_ENGAGE_AREA:
            _goto("Approach")
        else:
            global _angle, _line_err_prev
            col = _find_line_x(image)
            _debug["line_x"] = -1 if col is None else int(col)
            if col is None:
                # lost: slow and hold the last steer -- never park on a gap
                return (LINE_LOST_SPEED, _angle)
            half = rc.camera.get_width() / 2
            error = (col - half) / half
            d_error = error - _line_err_prev
            _line_err_prev = error
            _angle = rc_utils.clamp(
                LINE_PD_KP * error + LINE_PD_KD * d_error, -1.0, 1.0)
            spd = max(0.5 * GREEN_SPEED,
                      GREEN_SPEED * (1.0 - LINE_SLOW * abs(_angle)))
            return (spd, _angle)

    # ---- Approach: PID-steer at the sign, creep to the door ----
    if elevator_state == "Approach":
        if area >= _at_door_area():
            _door_area = area
            print("[elevator] AT THE DOOR: area={:.0f} -> inside when area "
                  ">= {:.0f} (x{:.1f})".format(
                      area, area * INSIDE_AREA_RATIO, INSIDE_AREA_RATIO))
            _goto("Reading")
            return (0.0, _sign_angle)
        if label is None:
            if lost_for >= SIGN_LOST_TIME:
                _goto("Green")         # resume the line instead of idling
                return (0.0, _angle)
            return (0.0, _sign_angle)
        return (APPROACH_SPEED, _sign_steer(center_x, dt))

    # ---- Reading: stopped at the door, waiting for GO ----
    if elevator_state == "Reading":
        if label == "GO":
            _goto("Entering")
        else:
            if label is not None and center_x is not None:
                _sign_steer(center_x, dt)
            return (0.0, _sign_angle)

    # ---- Entering: drive in; inside = door_area grown by the ratio ----
    if elevator_state == "Entering":
        if STOP_ON_STOP and label == "STOP":
            return (0.0, _sign_angle)
        ENTERING_TIMER += dt
        inside_area = (_door_area * INSIDE_AREA_RATIO) if _door_area > 0 else INSIDE_AREA
        slow_area = inside_area * 0.75
        moved_enough = ENTERING_TIMER >= ENTER_MIN_TIME
        if (moved_enough and area >= inside_area) or ENTERING_TIMER >= ENTER_TIMEOUT:
            print("[elevator] INSIDE: area={:.0f} (needed {:.0f}) after {:.1f}s"
                  .format(area, inside_area, ENTERING_TIMER))
            _goto("inElevator")
        else:
            spd = SLOW_ENTERING_SPEED if area >= slow_area else ENTERING_SPEED
            steer = _sign_steer(center_x, dt) if center_x is not None else _sign_angle
            return (spd, steer)

    # ---- inElevator: sit still while it moves ----
    if elevator_state == "inElevator":
        elevator_countdown -= dt
        if elevator_countdown > 0:
            return (0.0, 0.0)
        _exit_wait_timer = 0.0
        _goto("WaitToExit")

    # ---- WaitToExit: plain pause, then straight into Exiting this frame ----
    if elevator_state == "WaitToExit":
        _exit_wait_timer += dt
        if _exit_wait_timer >= EXIT_WAIT_TIME:
            _goto("Exiting")
        else:
            return (0.0, 0.0)

    # ---- Exiting: drive out, then done ----
    if elevator_state == "Exiting":
        EXITING_TIMER += dt
        if EXITING_TIMER >= EXIT_DRIVE_TIME:
            print("[elevator] Done. total {:.1f}s".format(_clock))
            _goto("Done")
            return None
        return (EXITING_SPEED, EXITING_ANGLE)

    return (0.0, _angle)
