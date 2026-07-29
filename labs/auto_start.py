import sys

sys.path.insert(1, "../../library")

from collections import deque
import racecar_core
import racecar_utils as rc_utils


rc = racecar_core.create_racecar()

# HSV range
RED_LOW = ((0, 120, 90), (8, 255, 255))
RED_HIGH = ((170, 120, 90), (179, 255, 255))
YELLOW = ((22, 100, 100), (35, 255, 255))
GREEN = ((45, 70, 70), (85, 255, 255))

global has_run

MIN_CONTOUR_AREA = 200

def detect_light_color(color_image):
    """Returns 'red', 'yellow', 'green', or None for a single frame,based on which color has the largest matching contour."""

    candidates = {}

    for name, ranges in [
        ("red", [RED_LOW, RED_HIGH]),
        ("yellow", [YELLOW]),
        ("green", [GREEN]),
    ]:
        best_area = 0
        for hsv_min, hsv_max in ranges:
            contours = rc_utils.find_contours(color_image, hsv_min, hsv_max)
            largest = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

            if largest is not None:
                best_area = max(best_area, rc_utils.get_contour_area(largest))

        if best_area > 0:
            candidates[name] = best_area

    if not candidates:
        return None

    return max(candidates, key=candidates.get)


def start():
    global has_run
    rc.drive.stop()
    rc.drive.set_max_speed(1.0)
    rc.set_update_slow_time(0.5)
    has_run = False


def update():
   global speed
   global has_run
   while not has_run:
      image = rc.camera.get_color_image()
      light = detect_light_color(image)
      if light == 'green':
         has_run = True
      else:
         rc.display.show_text("Waiting")
         speed = 0
         rc.drive.set_speed_angle(speed, 0)

   ## update loop code
   
 

def update_slow():
   pass


if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()