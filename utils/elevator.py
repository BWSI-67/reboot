import sys

sys.path.insert(1, "../../library")

import racecar_core
import racecar_utils as rc_utils


rc = racecar_core.create_racecar()

GREEN = ((40, 50, 50), (80, 255, 255), "GREEN")
BLUE = ((90, 50, 50), (120, 255, 255), "BLUE")

RED_LOW = ((0, 50, 50), (10, 255, 255), "RED")
RED_HIGH = ((170, 50, 50), (179, 255, 255), "RED")

COLORS = [GREEN, BLUE, RED_LOW, RED_HIGH]

status = "No AR marker detected"

def start():
    global status

    rc.drive.stop()
    rc.set_update_slow_time(0.5)
    status = "No AR marker detected"

    print("Elevator AR marker detector started")
    print("The car will not move")

def update():
    global status

    # Keep the car stopped during this detection test.
    rc.drive.stop()

    image = rc.camera.get_color_image()

    if image is None:
        status = "No camera image"
        return

    # Find AR markers and check the colored border around each marker.
    markers = rc_utils.get_ar_markers(image, COLORS)

    if len(markers) == 0:
        status = "No AR marker detected"
        return

    marker = markers[0]
    marker_id = marker.get_id()
    marker_color = marker.get_color()

    status = (
        "AR marker detected"
        + " | ID: "
        + str(marker_id)
        + " | Border color: "
        + str(marker_color)
    )

def update_slow():
    print(status)

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()