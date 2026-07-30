import sys

sys.path.insert(1, "../../library")

import racecar_core
import racecar_utils as rc_utils


rc = racecar_core.create_racecar()

# Driving
DRIVE_SPEED = 0.5
MAX_SPEED = 1.0

# Green HSV range
GREEN_MIN = (40, 50, 50)
GREEN_MAX = (80, 255, 255)

MIN_CONTOUR_AREA = 200

status = "Stopped: no green block"




def start():
    global status

    rc.drive.stop()
    rc.drive.set_max_speed(MAX_SPEED)
    rc.set_update_slow_time(0.5)

    status = "Stopped: no green block"


def update():
    global status

    image = rc.camera.get_color_image()

    if image is None:
        rc.drive.stop()
        status = "Stopped: no camera image"
        return

    # Find green objects in the upper image.
    contours = rc_utils.find_contours(image, GREEN_MIN, GREEN_MAX)

    green_block = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

    if green_block is not None:
        rc.drive.set_speed_angle(DRIVE_SPEED, 0.0)

        area = rc_utils.get_contour_area(green_block)
        status = "GREEN block detected: moving | Area: " + str(int(area))

    else:
        # No green block, or the visible block is another color.
        rc.drive.stop()
        status = "Stopped: no green block"

def update_slow():
    print(status)

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()