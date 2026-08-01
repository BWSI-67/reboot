MAX_SPEED, CRUISE = 1.0, 1.0   # ceiling, then the FRACTION of it to cruise at
kP, kD = 1.1, 0.02
# lidar left RANGE, right RANGE
RANGE = 110

angle = 0.0
speed = CRUISE
last_error = 0.0
a_kP = 0.5

import sys
import time

import numpy as np

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar(isSimulation=False)
time.sleep(2)

def findMaxW(scan, minA, maxA, window):
    length = scan.shape[0]

    originA = minA
    minA %= 360
    maxA %= 360

    start = int(minA * length / 360)
    end = int(maxA * length / 360)

    if start > end:
        sScan = np.append(scan[start:], scan[:end + 1])
    else:
        sScan = scan[start:end + 1]

    wSize = int(window * length / 360)
    sums = np.convolve(sScan, np.ones(wSize), "valid")
    center = int(sums.argmax()) + wSize // 2

    return originA + center * 360 / length

def start():
    rc.drive.stop()
    rc.set_update_slow_time(0.1)
    rc.drive.set_max_speed(MAX_SPEED)


def update():
    global speed, angle, last_error

    scan = rc.lidar.get_samples()
    front_dis = rc_utils.get_lidar_average_distance(scan, -10, 10)

    raw_gap = findMaxW(scan, -RANGE, RANGE, 30)

    error = raw_gap / RANGE * 2
    dt = rc.get_delta_time()
    d_error = (error - last_error) / dt if dt > 0 else 0.0
    last_error = error
    angle = float(np.clip(error * kP + d_error * kD, -1.0, 1.0))
    speed = CRUISE
    if front_dis < 30:
        speed = CRUISE * 0.5

    rc.drive.set_speed_angle(max(speed * (1 - abs(angle)) * a_kP, 0.3), rc_utils.clamp(angle + 0.04, -1.0, 1.0))
    print(raw_gap)

rc.set_start_update(start, update)
rc.go()