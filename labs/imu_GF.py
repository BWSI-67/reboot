MAX_SPEED = 1.0
kP, KD_YAW = 1.1, 0.3
TRIM = 0.04
RANGE = 110        # lidar search window: left RANGE .. right RANGE

angle = 0.0
speed = MAX_SPEED
yaw_rate = 0.0     # rad/s, + = LEFT (REP-103); stays 0 while the IMU is down

import sys, time
import numpy as np
sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

rc = racecar_core.create_racecar(isSimulation=False)
time.sleep(2)
# D term from the gyro (/attitude: bias-corrected, ~100 Hz, no lidar latency)
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

def onImu(msg):     # + = LEFT (REP-103)
    global yaw_rate
    yaw_rate = msg.angular_velocity.z

node = rclpy.create_node("gap_state")   # piggyback on racecar_core's executor
rclpy.get_global_executor().add_node(node)
node.create_subscription(Imu, "/attitude", onImu, qos_profile_sensor_data)

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
    global speed, angle
    scan = rc.lidar.get_samples()   # rc convention: index 0 = straight ahead
    if len(scan) == 0:
        return rc.drive.stop()     # no scan yet, right after launch

    raw_gap = findMaxW(scan, -RANGE, RANGE, 30)
    error = raw_gap / RANGE * 2
    # rc angle + = RIGHT but ROS yaw + = LEFT, so damping enters with a PLUS
    angle = float(np.clip(error * kP + KD_YAW * yaw_rate, -1.0, 1.0))
    speed = MAX_SPEED
    rc.drive.set_speed_angle(speed, rc_utils.clamp(angle + TRIM, -1.0, 1.0))
    print(f"gap {raw_gap:+6.1f} deg  yaw {np.degrees(yaw_rate):+6.1f} deg/s")

rc.set_start_update(start, update)
rc.go()
