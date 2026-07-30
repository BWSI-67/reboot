MAX_SPEED, CRUISE = 0.5, 1.0   # ceiling, then the FRACTION of it to cruise at
kP, TRIM = 0.9, 0.04
SLOW, SPEED_MIN = 0.7, 0.4     # shed speed with steering, floor as a FRACTION
ORANGE_SIDE = 1                # +1 = pass ORANGE on its right, GREEN on its
                               # left; flip this one number to swap the course
OPEN_CM, FOV, MIN_RUN = 150, 90, 8   # a run must be this wide to count, in deg
FRONT_CM, FRONT_ARC = 60, 12   # closer than this dead ahead: crawl, do not
                               # trust a gap the car cannot reach in time

angle = 0.0
side = ORANGE_SIDE     # which way to bias until a cone says otherwise
last_frame = None
import sys
import time
from pathlib import Path

import numpy as np

# cone_slalom_GF.py is in reboot/labs/
REBOOT_ROOT = Path(__file__).resolve().parents[1]

# RACECAR library is outside reboot.
sys.path.insert(0, str(REBOOT_ROOT.parent / "library"))

# cone_sight.py is in reboot/deploys/cone/
sys.path.insert(0, str(REBOOT_ROOT / "deploys" / "cone"))

import racecar_core
import racecar_utils as rc_utils
from cone_sight import nearest_cone, pick_gap   # same dir on the car

rc = racecar_core.create_racecar(isSimulation=False)
time.sleep(2)

def start():
    rc.drive.stop()
    rc.set_update_slow_time(0.1)
    rc.drive.set_max_speed(MAX_SPEED)

def update():   # 60 Hz, but the camera only refreshes every other tick
    global angle, side, last_frame

    # -- camera: one bit, which side of the next cone to pass on --------------
    image = rc.camera.get_color_image_no_copy()
    if image is not None and id(image) != last_frame:
        last_frame = id(image)          # a repeat frame costs a full HSV pass
        cone = nearest_cone(image)      # and cannot say anything new
        if cone is not None:
            side = ORANGE_SIDE if cone == "ORANGE" else -ORANGE_SIDE

    # -- lidar: where the gap on that side actually is ------------------------
    scan = rc.lidar.get_samples()
    if len(scan) == 0:
        return rc.drive.stop()          # no sweep yet, right after launch

    # The cone is itself an obstacle, so the gap beside it is already in the
    # scan. Taking the leftmost open run IS passing on the left; there is no
    # trajectory to compute and nothing breaks when the cone leaves the frame.
    gap = pick_gap(scan, side, OPEN_CM, FOV, MIN_RUN)
    angle = float(np.clip(kP * gap / FOV, -1.0, 1.0))
    speed = max(SPEED_MIN, CRUISE - SLOW * abs(angle))

    # A wide gap 50 deg away is useless if the cone is already at the bumper.
    # Zeros are dropped, not counted as near or far: they carry no range, and
    # this lidar returns 10-16% of them everywhere. pick_gap makes the opposite
    # call -- there a zero must not read as open space to drive into.
    arc = int(FRONT_ARC * len(scan) / 360)
    front = np.concatenate([scan[-arc:], scan[:arc + 1]])
    front = front[front > 0]
    if front.size and front.min() < FRONT_CM:
        speed = SPEED_MIN * 0.5
    rc.drive.set_speed_angle(speed, rc_utils.clamp(angle + TRIM, -1.0, 1.0))

def update_slow():   # gap 0.0 every frame means no run passed MIN_RUN: the
    # corridor is tighter than OPEN_CM, so lower it
    print(f"side {'R' if side > 0 else 'L'}  angle {angle:+.2f}")

rc.set_start_update(start, update, update_slow)
rc.go()
