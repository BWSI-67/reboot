"""Runs wf_lite on the car.

    CHECK=1 python3 race_deploy.py     sensor check, never drives (default)
    CHECK=0 VMAX=4.0 python3 ...       race
    GYRO_SIGN=-1 ...                   if the gyro reads inverted

The controller expects what the simulator fed it: a 720-sample cm scan, dt, and
{"gyro": [...], "speed": ...}. This file is the translation to real hardware.

Check first, on the ground: scan directions sane, gyro_y goes NEGATIVE when the
car is turned counter-clockwise, enc reads out when pushed. Then climb the
ladder VMAX = 1.5 -> 2.0 -> 2.5 -> 3.0 -> 4.0.
"""

import json
import os
import sys
import time

import numpy as np

try:
    import racecar_core
except ImportError:                    # laptop/sim: walk up to racecar-student/
    from pathlib import Path
    for _root in Path(__file__).resolve().parents:
        if (_root / "library").is_dir():
            sys.path.insert(0, str(_root / "library"))
            break
    import racecar_core

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wf_lite

rc = racecar_core.create_racecar()

CHECK_ONLY = os.environ.get("CHECK", "1") != "0"
GYRO_SIGN = float(os.environ.get("GYRO_SIGN", "1.0"))
VMAX = float(os.environ.get("VMAX", "1.5"))
PARAMS = os.environ.get("PARAMS", "../config/wf_lite_finalloop_params.json")

_here = os.path.dirname(os.path.abspath(__file__))
params = {k: v["default"] for k, v in wf_lite.PARAM_SPEC.items()}
_pfile = os.path.join(_here, PARAMS)
if os.path.exists(_pfile):
    params.update({k: v for k, v in json.load(open(_pfile)).items()
                   if k in params})
    print(f"params: {os.path.basename(_pfile)}")
else:
    print(f"params: {PARAMS} not found, using PARAM_SPEC defaults")
wf_lite.V_MAX = min(VMAX, wf_lite.V_MAX)      # only ever lower it
ctrl = wf_lite.create_controller(params, seed=0)

# ------------------------------------------------------------------ sensors
# rc.physics's /imu does not exist on this car, and /velocity is a message type
# that crashes take_message here. Both subscriptions piggyback on the racecar's
# own spinning node.
SPEED = []
GYRO = {"z": None, "t": 0.0}

try:
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Float32

    def _enc(m):
        SPEED.append(float(m.data))
        del SPEED[:-5]

    rc.physics.node.create_subscription(
        Float32, "/encoder/speed", _enc, qos_profile_sensor_data)
    rc.physics.node.create_subscription(
        Imu, "/attitude",
        lambda m: GYRO.update(z=float(m.angular_velocity.z), t=time.monotonic()),
        qos_profile_sensor_data)
except Exception as e:
    print(f"sensor taps unavailable ({e}) -- lidar only")


def make_imu():
    imu = {"gyro": None, "accel": None}
    if GYRO["z"] is not None and time.monotonic() - GYRO["t"] < 0.5:
        # /attitude is REP-103 (+z = CCW); the controller wants the sim's
        # opposite convention
        imu["gyro"] = [0.0, GYRO_SIGN * -GYRO["z"], 0.0]
    if len(SPEED) >= 3:
        # sensorless ESC: spikes and stuck zeros at low RPM. Outside the sane
        # window the twin beats the sensor, so omit the key entirely.
        vm = sorted(SPEED)[len(SPEED) // 2]
        if 0.30 < vm < 4.4:
            imu["speed"] = vm
    return imu


_RESAMPLE = {}


def to720(scan):
    """Real RPLidar emits ~1080/rev with angle_compensate; the contract is 720."""
    if scan is None or len(scan) == 720:
        return scan
    n = len(scan)
    if n not in _RESAMPLE:
        _RESAMPLE[n] = (np.round(np.arange(720) * (n / 720.0))
                        .astype(np.int64)) % n
    return np.asarray(scan, dtype=np.float32)[_RESAMPLE[n]]


_t = [0.0]
_frames = [0]


def start():
    rc.drive.stop()
    rc.drive.set_max_speed(1.0)
    ctrl.reset()
    mode = "CHECK (no driving)" if CHECK_ONLY else f"RACE v_max={wf_lite.V_MAX}"
    print(f"wf_lite [{mode}] GYRO_SIGN={GYRO_SIGN:+.0f}")


def update():
    dt = rc.get_delta_time()
    scan = to720(rc.lidar.get_samples())
    imu = make_imu()

    if CHECK_ONLY:
        rc.drive.stop()
        _t[0] += dt
        _frames[0] += 1
        if _t[0] >= 1.0:
            g = imu["gyro"]
            gtxt = f"{g[1]:+.2f}" if g else "N/A"
            fps = _frames[0] / _t[0]
            if scan is None or len(scan) != 720:
                n = "None" if scan is None else len(scan)
                print(f"[{fps:4.0f}fps] scan={n} | gyro_y={gtxt} | "
                      f"enc={imu.get('speed', 'N/A')}")
            else:
                print(f"[{fps:4.0f}fps] front={scan[0]:6.0f} "
                      f"right={scan[180]:6.0f} back={scan[360]:6.0f} "
                      f"left={scan[540]:6.0f}cm "
                      f"valid={(np.asarray(scan) > 0).mean() * 100:3.0f}% | "
                      f"gyro_y={gtxt} | enc={imu.get('speed', 'N/A')}")
            _t[0], _frames[0] = 0.0, 0
        return

    if scan is None or len(scan) != 720:
        rc.drive.stop()
        return

    speed, angle = ctrl.update(scan, dt, imu)
    rc.drive.set_speed_angle(max(min(speed, 1.0), -1.0),
                             max(min(angle, 1.0), -1.0))

    _t[0] += dt
    if _t[0] >= 1.0:
        _t[0] = 0.0
        if "speed" in imu:
            enc = f"{imu['speed']:.2f}"
        elif SPEED:                       # gated garbage shows as raw!
            enc = f"raw!{sorted(SPEED)[len(SPEED) // 2]:.2f}"
        else:
            enc = "N/A"
        print(f"v={ctrl.v:4.2f} blind={ctrl.blind:4.2f}m enc={enc} "
              f"cmd=({speed:+.2f},{angle:+.2f})")


rc.set_start_update(start, update)
rc.go()
