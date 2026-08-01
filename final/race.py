"""
MIT BWSI Autonomous RACECAR -- Grand Prix final: top-level state machine

    python3 race.py

Track sections run IN ORDER. Each section is a module exposing, by naming
convention for label <name>:

    <name>_reset()                      re-arm from start()
    <name>()  -> (speed, angle) | None  one frame; None = section finished
    <name>_ready() -> bool              OPTIONAL entry cue; while False the
                                        dispatcher cruises the transit leg

Between sections the track has plain corridor: the previous section is done
but the next one's entry cue (fork: arrow in sight; elevator: blue line in
the floor ROI) has not appeared yet. Handing those frames to the next
section would be wrong -- the elevator's line follow stops the car dead
with no line visible -- so the dispatcher gap-follows each transit leg
itself. WHICH WAY the car goes on a leg is controlled by that leg's scan
sector in TRANSITS below: (-90, 90) follows the widest gap anywhere,
(-90, 15) hugs left, (-15, 90) hugs right.

REORDER `SECTIONS` (and TRANSITS with it) TO MATCH THE ACTUAL TRACK.

Sections share one rc: it is created here and injected into every module
(the modules reference a bare `rc`, harness-style, and never create their
own). Signs/traffic-light response (Coral, ~/signs on the car) is a separate
overlay -- fold it in here only if the course actually mixes signs into
these sections.
"""

import os
import sys
import time

import cv2
import numpy as np

# ---- every run is logged to a timestamped file, terminal untouched ----
LOG_DIR = os.path.expanduser("~/race/logs")


class _Tee(object):
    def __init__(self, f, term):
        self.f = f
        self.t = term

    def write(self, s):
        self.t.write(s)
        self.f.write(s)

    def flush(self):
        self.t.flush()
        self.f.flush()


try:
    os.makedirs(LOG_DIR, exist_ok=True)
    _log_path = os.path.join(LOG_DIR, time.strftime("race_%Y%m%d_%H%M%S.log"))
    # ONE file handle shared by stdout and stderr: two handles appending to
    # the same file interleave and truncate each other, which ate the middle
    # of a traceback exactly when it was needed most.
    _log_f = open(_log_path, "a", buffering=1)
    sys.stdout = _Tee(_log_f, sys.stdout)
    sys.stderr = _Tee(_log_f, sys.stderr)
    print("[race] logging to {}".format(_log_path))
except Exception as _e:
    print("[race] file logging unavailable ({})".format(_e))

# racecar_core runs update() on its own thread; an uncaught exception there
# kills the thread silently and leaves the car coasting. Log it and stop.
import threading as _threading
import traceback as _traceback


def _thread_excepthook(args):
    print("[race] UNCAUGHT in {}:".format(args.thread.name))
    _traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
    try:
        rc.drive.stop()
    except Exception:
        pass


_threading.excepthook = _thread_excepthook

try:                                   # car: library already on PYTHONPATH
    import racecar_core
except ImportError:                    # fall back to the Neo jupyter_ws tree
    sys.path.insert(0, "/home/racecar/jupyter_ws/racecar-neo/library")
    import racecar_core

import cone_slalom
import elevator
import fork_arrow
import fork_depth
import wf_cruise

# Which fork implementation races. "depth" = the team's TEST-PASSING
# height-slab depth version (needs the depth stream alive -- with depth dead
# it parks in SEEK). "color" = the color+lidar sector version. One word to
# switch on race day.
FORK_IMPL = "depth"
_fork = fork_depth if FORK_IMPL == "depth" else fork_arrow

rc = racecar_core.create_racecar()
for _mod in (elevator, fork_arrow, fork_depth, cone_slalom):
    _mod.rc = rc

# Leg-0 cruise = the tuned wf_lite finalloop racer (race_deploy contract);
# holds the corridor center far better than the max-gap PD. Falls back to
# max-gap automatically if wf_lite fails to load.
_wf = wf_cruise.WfCruise(rc)


class _CheapCam(object):
    """Raw compressed-image tap, used ONLY by the transit-leg entry cues.

    racecar_core's camera subscribes LAZILY on the first get_color_image()
    call, and from then on cv_bridge decodes every 640x480 frame at 30 Hz
    inside the same executor that spins the control loop. race_deploy never
    touches the camera, which is exactly why its wall following is faster
    and steadier than ours was: merely asking for one frame taxes the
    controller for the rest of the run.

    So during transit we keep rc.camera untouched and tap the compressed
    topic ourselves: the callback only stashes the JPEG bytes (free), and
    a frame is decoded only on the frames where the cue is evaluated.
    Sections still use rc.camera -- they are vision-bound anyway.
    """

    def __init__(self, rc):
        self.buf = None
        self.ok = False
        try:
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CompressedImage
            rc.physics.node.create_subscription(
                CompressedImage, "/camera/color/image_raw/compressed",
                self._cb, qos_profile_sensor_data)
            self.ok = True
        except Exception as e:
            print("[race] cheap camera tap unavailable ({}) -- entry cues "
                  "will use rc.camera".format(e))

    def _cb(self, msg):
        self.buf = msg.data          # bytes only; no decode in the callback

    def frame(self):
        if self.buf is None:
            return None
        return cv2.imdecode(np.frombuffer(self.buf, np.uint8), cv2.IMREAD_COLOR)


_cam = _CheapCam(rc)

# ---------------- tuning ----------------
# (label, module) in TRACK ORDER. TRANSITS[i] governs the corridor BEFORE
# section i engages: (sector_min_deg, sector_max_deg, speed). The sector is
# the "which way" knob -- narrow it toward one side to force that side.
SECTIONS = [
    ("elevator",     elevator),
    ("cone",         cone_slalom),
]
# Track: wall-follow -> ELEVATOR -> wall-follow -> CONE SLALOM -> wall-follow.
# Every transit leg is the tuned wf_lite wall follower; the sector fallback
# below only runs if wf_lite failed to load.
TRANSITS = [
    (-90, 90, 0.5),        # start -> green line (wf_lite drives; ~5.7 s)
    (-90, 90, 0.5),        # elevator exit -> first cone
]
FINISH = (-90, 90, 0.5)    # after the last section, wall-follow to the line
MAX_SPEED = 1.0
GAP_WINDOW_DEG = 30
KP = 1.1

READY_BLIND_S = 4.5        # leg 0: pure wall-follow this long after the run
                           # actually starts (RIGHT BUMPER pressed AND the
                           # encoder says the car is moving). Measured: the
                           # green line arrives at ~5.7 s, so this must stay
                           # comfortably under that.
USE_BUMPER_START = True    # the run begins on the RIGHT BUMPER, not on START:
                           # START only loads the program, the bumper launches
                           # it, so the clock matches the timed run
READY_FRAMES = 3           # consecutive ready() CHECKS to actually engage
READY_EVERY_N = 6          # evaluate the entry cue only every Nth frame. The
                           # cue is now cheap (one half-res inRange) but the
                           # wall-follow loop rate is still sacred: wf_lite's
                           # model twin steers badly on a jittery loop.
# ----------------------------------------

_idx = 0
_engaged = False
_t0 = 0.0
_ready_count = 0
_frame_n = 0
_moved_at = None           # wall time the car was first MEASURED moving
_armed = not USE_BUMPER_START   # RIGHT BUMPER seen -> the run has begun

MOVE_SPEED_MPS = 0.25      # encoder speed that counts as "the car moved"


def _bumper_down():
    try:
        return rc.controller.is_down(rc.controller.Button.RB)
    except Exception:
        return True         # no controller API -> do not block the run


def _drive(speed, angle):
    """All dispatcher drive commands go through here so the blind window can
    anchor on MEASURED motion (encoder), not on a command: START can be
    pressed while the car is still held, and commands alone must not start
    the clock. If the encoder topic is silent entirely, fall back to the
    first nonzero command."""
    global _moved_at, _t0
    if _moved_at is None:
        v = _wf.measured_speed() if _wf.ok else None
        if v is not None and v > MOVE_SPEED_MPS:
            _moved_at = time.time()
            _t0 = _moved_at        # the race clock starts when the car MOVES
            print("[race] car moving ({:.2f} m/s) -- clock and blind window "
                  "start now".format(v))
        elif v is None and abs(speed) > 0.05:
            _moved_at = time.time()
            _t0 = _moved_at
    rc.drive.set_speed_angle(speed, angle)


def _cruise_angle(min_a, max_a):
    """Max-gap steering inside a sector (deg, 0 = forward, + = right)."""
    scan = rc.lidar.get_samples()
    if scan is None:
        return 0.0
    length = scan.shape[0]
    start = int((min_a % 360) * length / 360)
    end = int((max_a % 360) * length / 360)
    s_scan = (np.append(scan[start:], scan[:end + 1]) if start > end
              else scan[start:end + 1])
    w_size = int(GAP_WINDOW_DEG * length / 360)
    sums = np.convolve(s_scan, np.ones(w_size), "valid")
    gap = min_a + (int(sums.argmax()) + w_size // 2) * 360 / length
    return float(np.clip(gap / 180 * KP, -1.0, 1.0))


def start():
    global _idx, _engaged, _t0, _ready_count, _moved_at, _armed
    _idx = 0
    _engaged = False
    _ready_count = 0
    _moved_at = None
    _armed = not USE_BUMPER_START
    _t0 = time.time()
    if USE_BUMPER_START:
        print("[race] loaded -- HOLD THE RIGHT BUMPER to launch")
    for label, mod in SECTIONS:
        getattr(mod, label + "_reset")()
    _wf.reset()
    # wf_lite's speed model assumes max_speed == its spd_scale; the sections
    # get plain MAX_SPEED back the moment leg 0 hands over.
    rc.drive.set_max_speed(_wf.spd_scale if _wf.ok else MAX_SPEED)
    rc.drive.stop()
    rc.set_update_slow_time(0.5)


_crashed = [False]


def update():
    """Control loop with a guard rail: one bad frame must not kill the thread
    and leave the car driving itself."""
    try:
        _update()
    except Exception:
        if not _crashed[0]:
            _crashed[0] = True
            print("[race] EXCEPTION in update() -- stopping the car:")
            _traceback.print_exc()
        rc.drive.stop()


def _update():
    global _idx, _engaged, _ready_count, _frame_n, _armed, _t0

    # the run does not begin until the RIGHT BUMPER is pressed; before that
    # the car sits still, so the clock and the blind window match the timed run
    if not _armed:
        if _bumper_down():
            _armed = True
            _t0 = time.time()
            print("[race] RIGHT BUMPER -- run starts")
        else:
            rc.drive.stop()
            return

    while _idx < len(SECTIONS):
        label, mod = SECTIONS[_idx]

        # transit leg: cruise until the section's entry cue fires.
        # Leg 0 rides the wf_lite racer; later legs use the sector max-gap.
        # Entry needs READY_FRAMES consecutive hits, checked only every
        # READY_EVERY_N frames (the cue is expensive; the cruise loop rate
        # is sacred), and leg 0 is blind to its cue for READY_BLIND_S after
        # the car starts moving -- the start area shows green.
        if not _engaged:
            _frame_n += 1
            ready = getattr(mod, label + "_ready", None)
            blind = _idx == 0 and (_moved_at is None
                                   or time.time() - _moved_at < READY_BLIND_S)
            if blind:
                _ready_count = 0
            elif ready is not None and _frame_n % READY_EVERY_N == 0:
                # decode a frame ONLY on the frames the cue actually runs,
                # and never through rc.camera (see _CheapCam)
                img = _cam.frame() if _cam.ok else None
                _ready_count = _ready_count + 1 if ready(img) else 0
            if ready is not None and _ready_count < READY_FRAMES:
                if _wf.ok:                  # every transit leg wall-follows
                    out = _wf.update(rc.get_delta_time())
                    if out is not None:
                        _drive(*out)
                        return
                min_a, max_a, speed = TRANSITS[_idx]
                _drive(speed, _cruise_angle(min_a, max_a))
                return
            _engaged = True
            _ready_count = 0
            if _wf.ok:
                rc.drive.set_max_speed(MAX_SPEED)   # leave wf's speed regime
            print("[race] {} engaged at {:.1f}s".format(label, time.time() - _t0))

        out = getattr(mod, label)()
        if out is not None:
            speed, angle = out
            _drive(speed, angle)
            return
        print("[race] {} done at {:.1f}s".format(label, time.time() - _t0))
        _idx += 1                  # next leg/section gets THIS same frame
        _engaged = False
        if _wf.ok:                 # back into wf_lite's speed regime
            rc.drive.set_max_speed(_wf.spd_scale)
            _wf.reset()
    if _wf.ok:                     # final leg: wall-follow to the finish
        out = _wf.update(rc.get_delta_time())
        if out is not None:
            _drive(*out)
            return
    min_a, max_a, speed = FINISH
    _drive(speed, _cruise_angle(min_a, max_a))


_slow_t = [0.0, 0]         # last update_slow wall time, frames at that point


def update_slow():
    if _idx < len(SECTIONS):
        label, mod = SECTIONS[_idx]
        inner = (getattr(mod, label + "_state", "") if _engaged
                 else "transit->" + label)
    else:
        label, inner = "FINISH", ""
    # loop rate: the number that matters for wf_lite. It should sit near
    # 60 Hz on the transit legs; a drop means something expensive crept
    # into the per-frame path.
    now = time.time()
    hz = ""
    if _slow_t[0] > 0 and now > _slow_t[0] and not _engaged:
        hz = " {:.0f}Hz".format((_frame_n - _slow_t[1]) / (now - _slow_t[0]))
    _slow_t[0], _slow_t[1] = now, _frame_n
    # wf_lite sensor health, same fields race_deploy prints: a dead encoder or
    # gyro is silent otherwise, and wf_lite just quietly steers worse
    wf = ""
    if _wf.ok and not _engaged:
        v = _wf.measured_speed()
        g = _wf.measured_gyro()
        wf = " enc={} gyro={} twin={:.2f}".format(
            "{:.2f}".format(v) if v is not None else "N/A",
            "{:+.2f}".format(g) if g is not None else "N/A",
            getattr(_wf.ctrl, "v", 0.0) * getattr(_wf.ctrl, "ms", 1.0))
    t = now - _t0 if _moved_at is not None else 0.0
    print("[race] t={:5.1f}s section={} {}{}{}".format(t, label, inner, hz, wf))


rc.set_start_update(start, update, update_slow)
print("[race] press START on the pad to launch; BACK reclaims manual")
rc.go()
