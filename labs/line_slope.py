MAX_SPEED, CRUISE = 0.25, 1.0   # ceiling, then the straight-line FRACTION
kP, kS, TRIM = 0.6, 1.4, 0.04    # kP on offset, kS on slope (see update)
HSV_MIN, HSV_MAX = (90, 80, 60), (120, 255, 255)  # blue; S>=80 rejects glare
NEAR_TOP, FAR_TOP, BAND = 400, 300, 80   # two bands, BAND px tall each
MIN_AREA, SLOPE_DEADZONE = 150, 0.05
SLOW, SPEED_MIN, LOST_SPEED = 0.8, 0.4, 0.5   # all FRACTIONS of MAX_SPEED

angle = last_side = 0.0
last_frame = None
import sys, time
import cv2
import numpy as np
sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar(isSimulation=False)
time.sleep(2)

def findColumn(image, top):
    """Centre column of the line in the band at `top`, or None. Half scale."""
    band = cv2.resize(image[top:top + BAND], None, fx=0.5, fy=0.5,
                      interpolation=cv2.INTER_NEAREST)
    contours = rc_utils.find_contours(band, HSV_MIN, HSV_MAX)
    contour = rc_utils.get_largest_contour(contours, MIN_AREA)
    if contour is None:
        return None
    return rc_utils.get_contour_center(contour)[1] * 2   # back to full scale

def start():
    rc.drive.stop()
    rc.set_update_slow_time(0.1)
    rc.drive.set_max_speed(MAX_SPEED)

def update():
    global angle, last_side, last_frame
    image = rc.camera.get_color_image_no_copy()
    if image is None:
        rc.drive.stop()
        return
    if id(image) == last_frame:    # camera ~25 fps vs update() 60 Hz: a
        return                     # repeat costs 2 searches, says nothing new
    last_frame = id(image)

    half = rc.camera.get_width() / 2
    near = findColumn(image, NEAR_TOP)
    far = findColumn(image, FAR_TOP)
    if near is None and far is None:      # lost: keep turning the way the
        rc.drive.set_speed_angle(LOST_SPEED, last_side)   # line last went
        print(f"LOST  searching {last_side:+.0f}")
        return

    # offset = where the line is now. slope = which way it LEAVES the frame,
    # i.e. the line's heading, positive when it bends right. Steering on the
    # slope is what turns the car into a corner before the offset has grown --
    # two independent offset terms cannot do that, they only ever react.
    offset = ((near if near is not None else far) - half) / half
    slope = 0.0 if (near is None or far is None) else (far - near) / half
    angle = float(np.clip(kP * offset + kS * slope, -1.0, 1.0))
    if abs(offset) > SLOPE_DEADZONE:      # near centre the sign is noise
        last_side = 1.0 if offset > 0 else -1.0
    speed = max(SPEED_MIN, CRUISE - SLOW * abs(angle))
    rc.drive.set_speed_angle(speed, rc_utils.clamp(angle + TRIM, -1.0, 1.0))
    print(f"off {offset:+.2f} slope {slope:+.2f} a {angle:+.2f} v {speed:.2f}")

rc.set_start_update(start, update)
rc.go()
