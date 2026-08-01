import sys
import cv2 as cv

import racecar_utils as rc_utils

########################################################################################
# Constants
########################################################################################

# --- Color detection ---
GREEN_HSV = ((29, 79, 84), (86, 255, 255))
# ORANGE cone (course uses orange, not red). H capped at 17: YELLOW_GLARE_HSV
# starts at 18 and the glare mask would eat any orange pixels above it.
RED_HSV = ((4, 130, 90), (17, 255, 255))
WHITE_GLARE_HSV = ((0, 0, 230), (179, 80, 255))
YELLOW_GLARE_HSV = ((18, 50, 50), (40, 255, 255))
MIN_CONTOUR_AREA = 1000  # 10 ft (3 m) cone spacing: a cone at 3 m is only
                         # ~700 px^2, so 3000 meant blind-driving until ~1.5 m;
                         # 1000 acquires the next cone at ~2.5 m
TOP_CROP_ROW = 100          # rows removed from the top of the frame

# --- Camera steering ---
KP = 0.0019
GREEN_SETPOINT_FRACTION = 8 / 9
RED_SETPOINT_FRACTION = 1 / 9
GREEN_DEADBAND_X = 640      # was the literal 640 in the old green branch
RED_DEADBAND_X = 40         # was the literal 40 in the old red branch

# --- Open-loop return, applied after a cone leaves the frame ---
PASS_THROUGH_TIME = 0.35    # s, drive straight first so the car clears the cone
RETURN_RAMP_TIME = 0.25     # s, ease into the turn instead of snapping to it
RETURN_ANGLE_AFTER_RED = -0.45   # negative = left
RETURN_ANGLE_AFTER_GREEN = 0.45  # positive = right

SPEED = 0.45

########################################################################################
# Global variables
########################################################################################


last_hash = None
angle = 0.0
tracking_color = None        # color of the cone currently in view, None if none
last_passed_color = None     # color of the last cone tracked, sets the return direction
return_angle = 0.0           # full-strength angle to ease into once clear of the cone
time_since_loss = 0.0        # s since the cone left the frame

########################################################################################
# Helper functions
########################################################################################


def is_new_frame(image):
    """Reject a frame whose contents are identical to the previous one."""
    global last_hash
    if image is None:
        return False
    current_hash = hash(image.tobytes())
    if current_hash == last_hash:
        return False
    last_hash = current_hash
    return True


def glare_mask(hsv_image):
    """Mask of pixels to exclude: overexposed white plus the yellow line color.

    Both wash out or bleed into the green/red HSV ranges under strong lighting,
    the same failure mode the line follower's glare mask corrects for.
    """
    white = cv.inRange(hsv_image, WHITE_GLARE_HSV[0], WHITE_GLARE_HSV[1])
    yellow = cv.inRange(hsv_image, YELLOW_GLARE_HSV[0], YELLOW_GLARE_HSV[1])
    return cv.bitwise_or(white, yellow)


def find_cone(hsv_image, hsv_range, column_limit=None):
    """Largest contour of one color, or None if it is missing or too small.

    hsv_image     already converted to HSV and already cropped
    hsv_range     ((h_min, s_min, v_min), (h_max, s_max, v_max))
    column_limit  optional right-hand column bound, restricting the search
    """
    if column_limit is not None:
        hsv_image = rc_utils.crop(hsv_image, (0, 0), (hsv_image.shape[0], column_limit))

    color_mask = cv.inRange(hsv_image, hsv_range[0], hsv_range[1])
    filtered_mask = cv.bitwise_and(color_mask, cv.bitwise_not(glare_mask(hsv_image)))

    contours, _ = cv.findContours(filtered_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = rc_utils.get_largest_contour(contours)

    if contour is None or rc_utils.get_contour_area(contour) < MIN_CONTOUR_AREA:
        return None
    return contour


def find_cone_green(hsv_image):
    return find_cone(hsv_image, GREEN_HSV)


def find_cone_red(hsv_image):
    return find_cone(hsv_image, RED_HSV, column_limit=rc.camera.get_width() // 2 + 50)


def contour_center(green_contour, red_contour):
    """Take whichever cone is larger, which is normally the closer one.

    Returns (column of the contour center, color), or (None, None) if neither
    cone passed the area threshold.
    """
    green_area = rc_utils.get_contour_area(green_contour) if green_contour is not None else 0
    red_area = rc_utils.get_contour_area(red_contour) if red_contour is not None else 0

    if green_area == 0 and red_area == 0:
        return None, None
    if green_area >= red_area:
        return rc_utils.get_contour_center(green_contour)[1], "green"
    return rc_utils.get_contour_center(red_contour)[1], "red"


def camera_return(contour_center_x, color):
    """Proportional steering that drives the cone toward its target column."""
    if color == "green":
        setpoint = rc.camera.get_width() * GREEN_SETPOINT_FRACTION
        if contour_center_x > GREEN_DEADBAND_X:
            return 0.0
    else:
        setpoint = rc.camera.get_width() * RED_SETPOINT_FRACTION
        if contour_center_x < RED_DEADBAND_X:
            return 0.0

    return (contour_center_x - setpoint) * KP


def open_loop_return():
    """Straight until the car is past the cone, then ramp into the return turn."""
    if last_passed_color is None:
        return 0.0
    if time_since_loss < PASS_THROUGH_TIME:
        return 0.0
    if RETURN_RAMP_TIME <= 0:
        return return_angle

    ramp = min((time_since_loss - PASS_THROUGH_TIME) / RETURN_RAMP_TIME, 1.0)
    return return_angle * ramp


########################################################################################
# Main loop
########################################################################################




########################################################################################
# race.py SECTION CONTRACT ADAPTER
# cone()/cone_reset()/cone_ready()/cone_state -- rc is injected by race.py.
# Logic below is cone_slalom_tm4.py verbatim; only the drive call became a
# return value and a done condition was added.
########################################################################################

CONE_DONE_TIME = 3.0     # s with no cone in view at all -> the slalom is over
CONE_READY_AREA = MIN_CONTOUR_AREA   # a cone this big ends the transit leg

cone_state = "Slalom"
_ever_saw_cone = False


def cone_reset():
    global last_hash, angle, tracking_color, last_passed_color
    global return_angle, time_since_loss, cone_state, _ever_saw_cone
    last_hash = None
    angle = 0.0
    tracking_color = None
    last_passed_color = None
    return_angle = 0.0
    time_since_loss = 0.0
    cone_state = "Slalom"
    _ever_saw_cone = False


def cone_ready(image=None):
    """A real cone in view ends the transit leg."""
    if image is None:               # dispatcher passes its own cheap frame
        image = rc.camera.get_color_image()
    if image is None:
        return False
    image = rc_utils.crop(image, (TOP_CROP_ROW, 0),
                          (rc.camera.get_height(), rc.camera.get_width()))
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    return find_cone_green(hsv) is not None or find_cone_red(hsv) is not None


def cone():
    """One frame of the slalom. Returns (speed, angle), or None when done."""
    global angle, tracking_color, last_passed_color, return_angle
    global time_since_loss, cone_state, _ever_saw_cone

    if cone_state == "Done":
        return None

    time_since_loss += rc.get_delta_time()

    image = rc.camera.get_color_image()
    if not is_new_frame(image):
        return (SPEED, angle)          # duplicate frame: hold the command

    image = rc_utils.crop(image, (TOP_CROP_ROW, 0),
                          (rc.camera.get_height(), rc.camera.get_width()))
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    green_contour = find_cone_green(hsv_image)
    red_contour = find_cone_red(hsv_image)
    contour_center_x, color = contour_center(green_contour, red_contour)

    if contour_center_x is not None:
        tracking_color = color
        time_since_loss = 0.0
        _ever_saw_cone = True
        angle = camera_return(contour_center_x, color)
    else:
        if tracking_color is not None:
            last_passed_color = tracking_color
            tracking_color = None
            return_angle = (RETURN_ANGLE_AFTER_GREEN
                            if last_passed_color == "green"
                            else RETURN_ANGLE_AFTER_RED)
        # the slalom is over once the cones have been gone a while
        if _ever_saw_cone and time_since_loss >= CONE_DONE_TIME:
            print("[cone] no cone for {:.1f}s -> done".format(time_since_loss))
            cone_state = "Done"
            return None
        angle = open_loop_return()

    angle = rc_utils.clamp(angle, -1.0, 1.0)
    return (SPEED, angle)
