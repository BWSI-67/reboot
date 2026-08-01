"""wf_lite as the leg-0 cruise for race.py.

Wraps the evolve-contract controller exactly the way race_deploy.py does --
same param loading, same 720-bin scan resample, same /attitude + /encoder
sensor chain, same spd_scale contract -- so the car cruises to the green
line with the tuned finalloop racer instead of the naive max-gap PD (which
does not hold the corridor center).

Environment overrides keep the race_deploy muscle memory:
    VMAX=4.0    m/s cap (default 4.0, same as the race command line)

If anything here fails to load (file missing, controller error), the caller
falls back to the simple max-gap cruise -- never a crash.
"""

import importlib.util
import json
import os
import time

import numpy as np

WF_CTRL = "wf_lite.py"
WF_PARAMS = "wf_lite_finalloop_params.json"
WF_VMAX = float(os.environ.get("VMAX", "4.0"))   # race speed; the timed runs
                                                 # were all done at 4.0 m/s
GYRO_SIGN = float(os.environ.get("GYRO_SIGN", "1.0"))


class WfCruise:
    def __init__(self, rc):
        self.rc = rc
        self.ok = False
        self.spd_scale = 1.0
        self._rs = {}
        self._speed_hist = []
        self._gyro = {"z": None, "t": 0.0}
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            spec = importlib.util.spec_from_file_location(
                "wf_lite", os.path.join(here, WF_CTRL))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            params = {k: v["default"] for k, v in mod.PARAM_SPEC.items()}
            pfile = os.path.join(here, WF_PARAMS)
            if os.path.exists(pfile):
                tuned = json.load(open(pfile))
                params.update({k: v for k, v in tuned.items() if k in params})
            if "v_max" in params:
                params["v_max"] = WF_VMAX
            elif hasattr(mod, "V_MAX"):
                mod.V_MAX = min(WF_VMAX, float(mod.V_MAX))
            self.ctrl = mod.create_controller(params, seed=0)
            self.spd_scale = float(params.get("spd_scale", 1.0))
            self._subscribe()
            self.ok = True
            print("[race] leg-0 cruise: wf_lite  v_max={} m/s  ({})".format(
                params.get("v_max", getattr(mod, "V_MAX", "?")), WF_PARAMS))
        except Exception as e:
            print("[race] wf_lite unavailable ({}) -- leg 0 uses max-gap".format(e))

    # -- sensor chain, verbatim from race_deploy.py -----------------------
    def _subscribe(self):
        try:
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Imu
            from std_msgs.msg import Float32
            node = self.rc.physics.node
            node.create_subscription(Float32, "/encoder/speed",
                                     self._enc_cb, qos_profile_sensor_data)
            node.create_subscription(Imu, "/attitude",
                                     self._att_cb, qos_profile_sensor_data)
        except Exception:
            pass                     # lidar-only: wf_lite runs fine without

    def _enc_cb(self, m):
        self._speed_hist.append(float(m.data))
        if len(self._speed_hist) > 5:
            del self._speed_hist[0]

    def _att_cb(self, m):
        self._gyro["z"] = float(m.angular_velocity.z)
        self._gyro["t"] = time.monotonic()

    def _make_imu(self):
        g = None
        if (self._gyro["z"] is not None
                and time.monotonic() - self._gyro["t"] < 0.5):
            g = [0.0, GYRO_SIGN * (-self._gyro["z"]), 0.0]
        imu = {"gyro": g, "accel": None}
        h = self._speed_hist
        if len(h) >= 3:
            vm = sorted(h)[len(h) // 2]
            if 0.30 < vm < 4.4:
                imu["speed"] = vm
        return imu

    def _to720(self, scan):
        if scan is None:
            return None
        n = len(scan)
        if n == 720:
            return scan
        idx = self._rs.get(n)
        if idx is None:
            idx = (np.round(np.arange(720) * (n / 720.0)).astype(np.int64)) % n
            self._rs[n] = idx
        return np.asarray(scan, np.float32)[idx]

    def measured_gyro(self):
        """Yaw rate from /attitude, or None if the topic is stale/silent."""
        if (self._gyro["z"] is not None
                and time.monotonic() - self._gyro["t"] < 0.5):
            return self._gyro["z"]
        return None

    def measured_speed(self):
        """Median encoder speed in m/s, or None if the topic is silent."""
        h = self._speed_hist
        if len(h) >= 3:
            return sorted(h)[len(h) // 2]
        return None

    # -- per-frame --------------------------------------------------------
    def reset(self):
        if self.ok and hasattr(self.ctrl, "reset"):
            self.ctrl.reset()

    def update(self, dt):
        """(speed, angle) command, or None if the scan is unusable."""
        scan = self._to720(self.rc.lidar.get_samples())
        if scan is None or len(scan) != 720:
            return None
        speed, angle = self.ctrl.update(scan, dt, self._make_imu())
        return (max(min(speed, 1.0), -1.0), max(min(angle, 1.0), -1.0))
