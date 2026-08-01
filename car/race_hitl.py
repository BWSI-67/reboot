"""Human-in-the-loop racer: wf_lite drives, YOU override, overrides REPLAY.

Same contract as race_deploy.py (CTRL / PARAMS / VMAX / CHECK), plus three
modes that share one file:

  TEACH  (default)  wf_lite drives. Touch the right trigger or the left stick
                    and you INSTANTLY take over; let go and it hands back.
                    Every overridden frame is recorded against ODOMETRY
                    (encoder-integrated metres), not wall time, so a replay
                    lands in the same PLACE even at a different speed.
                    On exit (Ctrl-C / BACK) the run is written to
                    ~/race_takeovers/tk_<stamp>.json

  REPLAY=<file>     wf_lite drives, but inside every recorded takeover window
                    the recorded (speed, angle) is played back automatically.
                    You can still grab the controls at any moment: a live
                    human always outranks the recording, and the new
                    intervention is recorded on top (so each lap improves).

  Safety: in REPLAY the played-back speed is capped by wf_lite's own braking
  law (min of the two). A recording made on a clear track can therefore never
  drive you into something that is there today.

Usage on the car:
    cd ~/jupyter_ws/team1/labs/racing
    # lap 1 -- teach: drive the bits wf_lite gets wrong
    PARAMS=wf_lite_finalloop_params.json CTRL=wf_lite.py CHECK=0 VMAX=4.0 \
        python3 race_hitl.py
    # race -- replay those fixes automatically
    PARAMS=wf_lite_finalloop_params.json CTRL=wf_lite.py CHECK=0 VMAX=4.0 \
        REPLAY=~/race_takeovers/tk_20260801_120000.json python3 race_hitl.py

Controls:  RIGHT TRIGGER = throttle   LEFT STICK X = steer
           B (hold) = emergency stop      BACK = quit + save
"""

import importlib.util
import json
import os
import sys
import time

import numpy as np

try:
    import racecar_core
except ImportError:
    from pathlib import Path
    for _root in Path(__file__).resolve().parents:
        if (_root / "library").is_dir():
            sys.path.insert(0, str(_root / "library"))
            if (_root / "sim2d").is_dir():
                sys.path.insert(0, str(_root / "sim2d"))
            break
    import racecar_core

rc = racecar_core.create_racecar()

CHECK_ONLY = os.environ.get("CHECK", "1") != "0"
GYRO_SIGN = float(os.environ.get("GYRO_SIGN", "1.0"))
V_MAX_OVERRIDE = float(os.environ.get("VMAX", "1.5"))
REPLAY_FILE = os.path.expanduser(os.environ.get("REPLAY", ""))
OUT_DIR = os.path.expanduser("~/race_takeovers")

# --- how a human is detected. Deliberately twitchy: a deliberate nudge must
# take effect on the SAME frame, so there is no fighting the controller.
TRIG_TH = 0.06        # right trigger past this = you are driving
STICK_TH = 0.12       # left stick |x| past this = you are steering
HOLD_S = 0.45         # stay in manual this long after you let go (a stick
                      # crossing zero mid-correction must not flap to auto)
REPLAY_WIN_M = 0.25   # replay a stored sample when within this distance of it
STEER_GAIN = 1.0      # scale of your stick -> steering command

# ------------------------------------------------------------ load controller
_here = os.path.dirname(os.path.abspath(__file__))
_ctrl_file = os.environ.get("CTRL", "wf_lite.py")
_stem = os.path.splitext(os.path.basename(_ctrl_file))[0]
_spec = importlib.util.spec_from_file_location(
    _stem, os.path.join(_here, _ctrl_file))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_params = {k: v["default"] for k, v in _mod.PARAM_SPEC.items()}
_pfile = os.path.join(_here, os.environ.get("PARAMS", _stem + "_params.json"))
if os.path.exists(_pfile):
    _params.update({k: v for k, v in json.load(open(_pfile)).items()
                    if k in _params})
    print(f"race_hitl: loaded tuned params from {os.path.basename(_pfile)}")

if "v_max" in _params:
    _params["v_max"] = float(V_MAX_OVERRIDE)
elif hasattr(_mod, "V_MAX"):
    _mod.V_MAX = min(float(V_MAX_OVERRIDE), float(_mod.V_MAX))
ctrl = _mod.create_controller(_params, seed=0)

# ------------------------------------------------------------ sensor adapter
SPEED = {"hist": []}
GYRO = {"z": None, "t": 0.0}


def _enc_cb(m):
    h = SPEED["hist"]
    h.append(float(m.data))
    if len(h) > 5:
        del h[0]


try:
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu
    from std_msgs.msg import Float32
    rc.physics.node.create_subscription(
        Float32, "/encoder/speed", _enc_cb, qos_profile_sensor_data)
    rc.physics.node.create_subscription(
        Imu, "/attitude",
        lambda m: GYRO.update(z=float(m.angular_velocity.z),
                              t=time.monotonic()), qos_profile_sensor_data)
except Exception as e:
    print(f"race_hitl: sensor taps unavailable ({e})")


def make_imu():
    g = None
    if GYRO["z"] is not None and time.monotonic() - GYRO["t"] < 0.5:
        g = [0.0, GYRO_SIGN * (-GYRO["z"]), 0.0]
    imu = {"gyro": g, "accel": None}
    h = SPEED["hist"]
    if len(h) >= 3:
        vm = sorted(h)[len(h) // 2]
        if 0.30 < vm < 4.4:
            imu["speed"] = vm
    return imu


def measured_speed():
    h = SPEED["hist"]
    if len(h) >= 3:
        vm = sorted(h)[len(h) // 2]
        if 0.0 <= vm < 5.0:
            return vm
    return None


# ------------------------------------------------------------- replay table
REPLAY = []            # [(s_m, speed, angle), ...] sorted by s
if REPLAY_FILE and os.path.exists(REPLAY_FILE):
    _d = json.load(open(REPLAY_FILE))
    REPLAY = [(float(r["s"]), float(r["speed"]), float(r["angle"]))
              for r in _d.get("samples", [])]
    REPLAY.sort(key=lambda r: r[0])
    print(f"race_hitl: REPLAY {len(REPLAY)} samples from "
          f"{os.path.basename(REPLAY_FILE)} "
          f"({REPLAY[0][0]:.1f}-{REPLAY[-1][0]:.1f} m)" if REPLAY else
          "race_hitl: replay file has no samples")
elif REPLAY_FILE:
    print(f"race_hitl: REPLAY file not found: {REPLAY_FILE}")

_R_S = np.array([r[0] for r in REPLAY]) if REPLAY else None


def replay_at(s):
    """Recorded (speed, angle) if this distance is inside a takeover window."""
    if _R_S is None or _R_S.size == 0:
        return None
    i = int(np.searchsorted(_R_S, s))
    best, bd = None, 1e9
    for j in (i - 1, i):
        if 0 <= j < len(REPLAY):
            d = abs(REPLAY[j][0] - s)
            if d < bd:
                best, bd = REPLAY[j], d
    if best is not None and bd <= REPLAY_WIN_M:
        return best[1], best[2]
    return None


# ------------------------------------------------------------------- state
S = {"dist": 0.0, "manual_t": -9.0, "rec": [], "mode": "auto",
     "n_take": 0, "t0": None, "last_mode": "auto"}
_t_print = [0.0]
_frames = [0]


def human_input():
    """(speed, angle, active) from the pad. Active means YOU are driving."""
    try:
        thr = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
        brk = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
        x, _y = rc.controller.get_joystick(rc.controller.Joystick.LEFT)
    except Exception:
        return 0.0, 0.0, False
    active = thr > TRIG_TH or brk > TRIG_TH or abs(x) > STICK_TH
    return float(thr - brk), float(np.clip(STEER_GAIN * x, -1.0, 1.0)), active


def save():
    if not S["rec"]:
        print("race_hitl: no takeovers recorded, nothing saved")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, time.strftime("tk_%Y%m%d_%H%M%S.json"))
    json.dump({"samples": S["rec"], "dist_m": S["dist"],
               "takeovers": S["n_take"],
               "params": os.path.basename(_pfile), "vmax": V_MAX_OVERRIDE},
              open(path, "w"))
    print(f"\nrace_hitl: SAVED {len(S['rec'])} samples / {S['n_take']} "
          f"takeovers over {S['dist']:.1f} m -> {path}")
    print(f"  race it back with:  REPLAY={path} <same env> python3 race_hitl.py")


def start():
    rc.drive.stop()
    rc.drive.set_max_speed(1.0)
    if hasattr(ctrl, "reset"):
        ctrl.reset()
    S.update(dist=0.0, manual_t=-9.0, rec=[], mode="auto", n_take=0, t0=None)
    mode = "STATIC CHECK (no driving)" if CHECK_ONLY else (
        "REPLAY+HITL" if REPLAY else "TEACH (drive to override)")
    print(f"race_hitl: {_mod.ALGO_FAMILY}  [{mode}]  VMAX={V_MAX_OVERRIDE}")
    print("  RIGHT TRIGGER = throttle | LEFT STICK = steer | "
          "release = back to auto | BACK = quit+save")


_RS = {}


def to720(scan):
    if scan is None:
        return None
    n = len(scan)
    if n == 720:
        return scan
    idx = _RS.get(n)
    if idx is None:
        idx = (np.round(np.arange(720) * (n / 720.0)).astype(np.int64)) % n
        _RS[n] = idx
    return np.asarray(scan, dtype=np.float32)[idx]


def update():
    dt = rc.get_delta_time()
    scan = to720(rc.lidar.get_samples())
    imu = make_imu()

    # odometry: distance is the replay index (a place, not a moment)
    v = measured_speed()
    if v is None:
        v = abs(getattr(ctrl, "v", 0.0))
    S["dist"] += v * dt

    if CHECK_ONLY:
        rc.drive.stop()
        _t_print[0] += dt
        _frames[0] += 1
        if _t_print[0] >= 1.0:
            fps = _frames[0] / _t_print[0]
            _t_print[0] = 0.0
            _frames[0] = 0
            g = imu.get("gyro")
            hs, ha, hact = human_input()
            print(f"[{fps:4.0f}fps] scan="
                  f"{'None' if scan is None else len(scan)} | "
                  f"gyro_y={'%+.2f' % g[1] if g else 'N/A'} | "
                  f"enc={imu.get('speed', 'N/A')} | "
                  f"pad=({hs:+.2f},{ha:+.2f}) {'MANUAL' if hact else 'idle'}")
        return

    try:
        if rc.controller.is_down(rc.controller.Button.B):
            rc.drive.stop()
            return
    except Exception:
        pass

    if scan is None or len(scan) != 720:
        rc.drive.stop()
        return

    a_speed, a_angle = ctrl.update(scan, dt, imu)      # wf_lite always runs
    now = time.monotonic()
    h_speed, h_angle, h_active = human_input()

    if h_active:
        S["manual_t"] = now
    manual = (now - S["manual_t"]) < HOLD_S

    if manual:
        speed, angle, src = h_speed, h_angle, "MANUAL"
        # record against distance so the fix replays at the same PLACE
        S["rec"].append({"s": round(S["dist"], 3),
                         "speed": round(float(speed), 3),
                         "angle": round(float(angle), 3)})
    else:
        rp = replay_at(S["dist"])
        if rp is not None:
            # replay, but never faster than wf_lite's braking law allows
            speed = min(rp[0], a_speed) if rp[0] > 0 else rp[0]
            angle, src = rp[1], "REPLAY"
        else:
            speed, angle, src = a_speed, a_angle, "auto"

    if src != S["last_mode"]:
        if src == "MANUAL":
            S["n_take"] += 1
        print(f"[{S['dist']:6.1f}m] {S['last_mode']} -> {src}")
        S["last_mode"] = src

    rc.drive.set_speed_angle(max(min(speed, 1.0), -1.0),
                             max(min(angle, 1.0), -1.0))

    _t_print[0] += dt
    if _t_print[0] >= 1.0:
        _t_print[0] = 0.0
        blind = getattr(ctrl, "blind", 0.0)
        print(f"s={S['dist']:6.1f}m {src:6s} cmd=({speed:+.2f},{angle:+.2f}) "
              f"auto=({a_speed:+.2f},{a_angle:+.2f}) "
              f"enc={imu.get('speed', 0.0) if 'speed' in imu else 0.0:.2f} "
              f"blind={blind:.1f} rec={len(S['rec'])}")

    try:
        if rc.controller.is_down(rc.controller.Button.BACK):
            rc.drive.stop()
            save()
            os._exit(0)
    except Exception:
        pass


try:
    rc.set_start_update(start, update)
    rc.go()
finally:
    if not CHECK_ONLY:
        save()
