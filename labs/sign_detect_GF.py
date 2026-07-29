MAX_SPEED, CRUISE = 1.0, 0.7
kP, kD, TRIM = 1.8, 0.4, 0.03
STOP_HOLD_S, YIELD_SLOW_S, YIELD_FACTOR = 2.0, 3.0, 0.4
COOLDOWN_S = 4.0
VOTE_N, VOTE_K = 5, 3
TRIGGER_H, CONF = 0.22, 0.5   # box height / frame height = close enough to act

angle = last_error = 0.0
state = "CRUISE"
state_until = cooldown_until = 0.0

import sys, time
import numpy as np
sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
from sign_support import CLASS_NAMES, SignWatcher, findMaxW   # same dir on car

GO_AROUND, STOP, YIELD = 3, 7, 8
ACTION = {STOP: ("STOP_WAIT", STOP_HOLD_S),
          YIELD: ("YIELD_SLOW", YIELD_SLOW_S),
          GO_AROUND: ("GO", 0.5)}
rc = racecar_core.create_racecar(isSimulation=False)
time.sleep(2)
watcher = SignWatcher("best_v5_edgetpu.tflite", (GO_AROUND, STOP, YIELD), CONF,
                      TRIGGER_H, VOTE_N, core=3, nice=10)   # cores 0-2 are busy

def start():
    rc.drive.stop()
    rc.set_update_slow_time(0.1)
    rc.drive.set_max_speed(MAX_SPEED)

def update():   # 60 Hz, but the camera only refreshes every other tick
    global angle, last_error, state, state_until, cooldown_until
    now = time.time()
    watcher.poll(rc.camera.get_color_image())

    if state != "CRUISE" and now >= state_until:
        state = "CRUISE"
    if state == "CRUISE" and now >= cooldown_until:
        sign = watcher.winner(VOTE_K)       # a class with K of N votes, or None
        if sign in ACTION:
            state, hold = ACTION[sign]
            state_until = now + hold
            cooldown_until = state_until + COOLDOWN_S
            watcher.clear()
            print(f"[sign] {CLASS_NAMES[sign]} -> {state}")

    scan = rc.lidar.get_samples()
    raw_gap = findMaxW(scan, -110, 110, 30)
    error = raw_gap / 220
    dt = rc.get_delta_time()
    d_error = (error - last_error) / dt if dt > 0 else 0.0
    last_error = error
    angle = float(np.clip(error * kP + d_error * kD, -1.0, 1.0))
    drive = CRUISE
    if state == "STOP_WAIT":
        drive = 0.0
    elif state == "YIELD_SLOW":
        drive *= YIELD_FACTOR
    rc.drive.set_speed_angle(drive, rc_utils.clamp(angle + TRIM, -1.0, 1.0))

def update_slow():
    print(f"{state:<10} {watcher.summary()}")

rc.set_start_update(start, update, update_slow)
rc.go()
