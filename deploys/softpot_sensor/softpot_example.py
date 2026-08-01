import sys
import cv2 as cv
from softpot import SoftPot

sys.path.insert(1, '../../library')
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar()
softpot = SoftPot()


def start():
    rc.drive.set_speed_angle(0, 0)
    print("SoftPot initialized")


def update():
    position = softpot.read_percent()
    voltage = softpot.read_voltage()
    
    if position is None:
        # Nobody is touching the sensor
        return
    
    print(position)
    rc.drive.set_speed_angle(0, 0)

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()