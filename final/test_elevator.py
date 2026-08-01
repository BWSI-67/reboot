#!/usr/bin/env python3
"""Standalone harness: run ONLY the elevator section, with live internals.

    python3 test_elevator.py           # START on the gamepad begins it
    python3 test_elevator.py --hold    # perception only: prints everything,
                                       # never moves (speed forced to 0)

The 0.5 s status line is elevator_debug(): state, sign color, sign_area,
ride countdown, green credit. sign_area is the number you need for the
three area gates (SIGN_DETECT_AREA / GREEN_SIGN_AREA / SLOW_ENTER_AREA):
park the car at the distances where each should fire and read it off.
"""

import argparse
import os
import sys
import time

# ---- every run is logged to a timestamped file, terminal untouched ----
try:
    _d = os.path.expanduser("~/race/logs")
    os.makedirs(_d, exist_ok=True)
    _p = os.path.join(_d, time.strftime("elev_%Y%m%d_%H%M%S.log"))

    class _Tee(object):
        def __init__(self, path, term):
            self.f = open(path, "a", buffering=1)
            self.t = term

        def write(self, s):
            self.t.write(s)
            self.f.write(s)

        def flush(self):
            self.t.flush()
            self.f.flush()

    sys.stdout = _Tee(_p, sys.stdout)
    sys.stderr = _Tee(_p, sys.stderr)
    print("[test] logging to {}".format(_p))
except Exception as _e:
    print("[test] file logging unavailable ({})".format(_e))

try:                                   # car: library already on PYTHONPATH
    import racecar_core
except ImportError:                    # fall back to the Neo jupyter_ws tree
    sys.path.insert(0, "/home/racecar/jupyter_ws/racecar-neo/library")
    import racecar_core

import elevator

ap = argparse.ArgumentParser()
ap.add_argument("--hold", action="store_true", help="perception only, no motion")
args = ap.parse_args()

rc = racecar_core.create_racecar()
elevator.rc = rc


def start():
    elevator.elevator_reset()
    rc.drive.set_max_speed(0.5)      # section speeds are all <= 0.5 anyway
    rc.drive.stop()
    rc.set_update_slow_time(0.5)


def update():
    out = elevator.elevator()
    if out is None:
        print("[test] elevator Done -- stopping")
        rc.drive.stop()
        return
    speed, angle = out
    if args.hold:
        speed = 0.0
    rc.drive.set_speed_angle(speed, angle)


def update_slow():
    print("[test] ready={} {}".format(elevator.elevator_ready(),
                                      elevator.elevator_debug()))


rc.set_start_update(start, update, update_slow)
print("[test] press START on the pad to begin{}".format(
    " (perception only, holding still)" if args.hold else ""))
rc.go()
