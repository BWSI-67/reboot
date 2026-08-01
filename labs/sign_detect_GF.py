MAX_SPEED, CRUISE = 0.5, 0.7
kP, kD, TRIM = 1.8, 0.4, 0.03 # Trim lessens drift
STOP_HOLD_S, YIELD_SLOW_S, YIELD_FACTOR = 2.0, 3.0, 0.4   # Stop hold time, yield slow time, yield factor of slowing
CLEAR_S, MAX_HOLD_S = 0.4, 8.0   # resume 0.4 after the sign leaves view, follow sign for a max 8 seconds
BRAKE, BRAKE_S = 0.4, 0.3        # braking and how long to brake
COOLDOWN_S, VOTE_N, NEED = 4.0, 9, 2.0      # car parks in front of never leaves
TRIGGER_H, CONF = 0.22, 0.35  # full-weight distance; conf floor (see NEED)
LIGHT_N, LIGHT_K, LIGHT_PX, RED_MAX_S = 7, 5, 150, 20.0   # wall light strips
angle = last_error = hold_until = hard_until = cool_until = 0.0
state, active = "CRUISE", None
import sys, time
import numpy as np
sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils
from sign_support import CLASS_NAMES, SignWatcher, findMaxW   # same dir on car
GO_AROUND, STOP, YIELD = 3, 7, 8
ACTION = {STOP: ("STOP_WAIT", STOP_HOLD_S), GO_AROUND: ("GO", 0.5),
          YIELD: ("YIELD_SLOW", YIELD_SLOW_S)}
rc = racecar_core.create_racecar(isSimulation=False)
time.sleep(2)
watcher = SignWatcher("best_v5_edgetpu.tflite", (GO_AROUND, STOP, YIELD), CONF,
                      TRIGGER_H, VOTE_N, 3, 10,   # pinned off the busy cores
                      LIGHT_N, LIGHT_K, LIGHT_PX, RED_MAX_S)
def start():
    rc.drive.stop()
    rc.set_update_slow_time(0.1)
    rc.drive.set_max_speed(MAX_SPEED)

def update():   # 60 Hz, but the camera only refreshes every other tick
    global angle, last_error, state, active, hold_until, hard_until, cool_until
    now = time.time()
    watcher.poll(rc.camera.get_color_image(), now)
    if state != "CRUISE":
        if watcher.count(active):          # still in view: keep the hold alive
            hold_until = max(hold_until, now + CLEAR_S)
        if now >= hold_until or now >= hard_until:
            state, cool_until = "CRUISE", now + COOLDOWN_S
    elif now >= cool_until:
        sign = watcher.winner(NEED)     # enough accumulated evidence, or None
        if sign in ACTION:
            state, hold = ACTION[sign]
            active, hold_until, hard_until = sign, now + hold, now + MAX_HOLD_S
            watcher.clear()
            print(f"[sign] {CLASS_NAMES[sign]} -> {state}")
    scan = rc.lidar.get_samples()
    if len(scan) == 0:
        return rc.drive.stop()      # lidar has not published its first sweep
    raw_gap = findMaxW(scan, -110, 110, 30)
    error = raw_gap / 220
    dt = rc.get_delta_time()
    d_error = (error - last_error) / dt if dt > 0 else 0.0
    last_error = error
    angle = float(np.clip(error * kP + d_error * kD, -1.0, 1.0))
    drive = CRUISE                 # sign and strip both cut in, strictest wins
    if state == "YIELD_SLOW":
        drive *= YIELD_FACTOR
    if state == "STOP_WAIT":       # speed 0 is ESC neutral: brake, then hold
        drive = -BRAKE if hard_until - now > MAX_HOLD_S - BRAKE_S else 0.0
    if watcher.red_since:          # red strip outranks everything
        drive = -BRAKE if now - watcher.red_since < BRAKE_S else 0.0
    rc.drive.set_speed_angle(drive, rc_utils.clamp(angle + TRIM, -1.0, 1.0))
def update_slow():
    print(f"{state:<10} {watcher.summary()}")

rc.set_start_update(start, update, update_slow)
rc.go()
