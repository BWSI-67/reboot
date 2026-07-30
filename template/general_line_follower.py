MAX_SPEED, CRUISE = 1.0, 1.0   # ceiling, then the FRACTION of it to cruise at
kP, kD = 0.5, 1.2
HSV_MIN, HSV_MAX = (90, 80, 60), (120, 255, 255)  # blue; S>=80 rejects glare
CROP_TOP, MIN_AREA = 360, 300   # bottom of the frame; tape ~1900 px at half
TRIM, SLOW = 0.04, 0.8

angle = 0.0
speed = CRUISE
last_error = 0.0
last_frame = None

import sys, time
from pathlib import Path
import cv2
import numpy as np

for _root in Path(__file__).resolve().parents:
    if (_root / "library").is_dir():
        sys.path.insert(0, str(_root / "library"))
        break
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar(isSimulation=False)
time.sleep(2)

def findLine(image):
    small = cv2.resize(image[CROP_TOP:], None, fx=0.5, fy=0.5,
                       interpolation=cv2.INTER_NEAREST)  # a quarter of the CPU
    contours = rc_utils.find_contours(small, HSV_MIN, HSV_MAX)
    contour = rc_utils.get_largest_contour(contours, MIN_AREA)
    if contour is None:
        return None
    return rc_utils.get_contour_center(contour)[1] * 2   # back to full scale

def start():
    rc.drive.stop()
    rc.set_update_slow_time(0.1)
    rc.drive.set_max_speed(MAX_SPEED)

def update():
    global speed, angle, last_error, last_frame
    image = rc.camera.get_color_image_no_copy()
    if image is None:
        rc.drive.stop()
        return
    if id(image) == last_frame:
        return                    # camera is ~23 fps, update() runs at 60 Hz
    last_frame = id(image)

    column = findLine(image)
    if column is None:
        rc.drive.set_speed_angle(0.3, angle)      # lost: slow, hold last steer
        print("LOST")
        return

    half = rc.camera.get_width() / 2
    error = (column - half) / half
    d_error = error - last_error   # one camera frame apart, so no dt math
    last_error = error
    angle = float(np.clip(error * kP + d_error * kD, -1.0, 1.0))
    speed = max(0.4, CRUISE - SLOW * abs(angle))
    rc.drive.set_speed_angle(speed, rc_utils.clamp(angle + TRIM, -1.0, 1.0))
    print(f"col {column:.0f}  error {error:+.2f}  angle {angle:+.2f}")

rc.set_start_update(start, update)
rc.go()