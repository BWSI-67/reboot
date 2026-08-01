########################################################################################
# Imports
########################################################################################
import sys
import math
from pathlib import Path
from typing import List, Optional, Sequence

sys.path.insert(1, str(Path(__file__).resolve().parents[3] / "utils"))
import paths  # noqa: F401 - adds library/utils/config to sys.path

import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# LIDAR side-sampling angle (degrees, using the 0=forward/clockwise
# convention) for the left/right readings below. Normal mode looks mostly
# forward; turn mode looks farther to each side to choose the roomier exit.
NORMAL_SIDE_ANGLE = 20
TURN_SIDE_ANGLE   = 50
SIDE_WINDOW       = 30   # fixed averaging window (degrees), same in both modes

# Tunable parameters
speed = 0.0                        # Current commanded speed (updated every frame)
angle = 0.0                        # Current commanded steering angle (updated every frame)
lastSteer = 0.0                    # Smoothed steering output from the previous frame
angleOffset = 0.02                 # Constant steering trim added every frame (corrects drift/bias)
MAX_STEER = 1.0                    # Clamp on the steering command magnitude
speed_angle_scale = 1.0            # Extra multiplier on speed based on how sharp the current angle is
FRONT_TURN = 150.0                 # Front distance (cm) below which turning_mode engages
FRONT_SLOW = 200.0                 # Front distance (cm) below which the car slows down
frontDist = 999.0                  # Latest measured front distance (cm)
leftDist = 0.0                     # Latest measured left-side distance (cm)
rightDist = 0.0                    # Latest measured right-side distance (cm)
WHEELBASE = 17.78                  # Measured car wheelbase (cm) used in the pure-pursuit curvature formula
MAX_STEER_RAD = 0.5                # Approximate steering-rack limit (radians) used to normalize steer_rad to [-1, 1]
LOOKAHEAD_MIN = 220.0              # Minimum pure-pursuit lookahead distance (cm)
LOOKAHEAD_SCALE = 330.0            # Multiplier scaling lookahead distance with current speed
turning_mode = False               # True while committed to a hard, full-lock turn
lidar_valid = False                # True when the latest LIDAR scan had at least one valid return
STEER_SMOOTHING = 0.3              # Exponential smoothing factor on steer output (0 = none, closer to 1 = smoother/slower)
FRONT_TURN_EXIT_SCALE = 1.3        # Multiplier on FRONT_TURN for the "exit turning_mode" hysteresis threshold
DIST_SMOOTHING_FRAMES = 4          # Number of frames averaged to smooth left/right distance readings
left_dist_hist: List[float] = []   # Rolling history of left-side distance readings for smoothing
right_dist_hist: List[float] = []  # Rolling history of right-side distance readings for smoothing
FRONT_OPEN = 400.0                 # Front distance (cm) at/above which the road is considered fully open
LOOKAHEAD_TURN_MULT = 0.6          # Lookahead multiplier used when close to a wall/turn (shorter, tighter)
LOOKAHEAD_STRAIGHT_MULT = 1.4      # Lookahead multiplier used on an open straight (longer, smoother)
turn_direction = 0                 # Committed turn direction (+1/-1) for the duration of a hard turn
TURN_DEADBAND = 15.0               # Left/right distance difference (cm) below which turn direction falls back to the previous lean
TURN_SPEED = 0.25                  # Speed used while turning_mode is active
NORMAL_SPEED = 0.6                 # Base speed used during normal pure-pursuit wall following
SLOW_SPEED = 0.3                   # Speed used when approaching a wall (frontDist < FRONT_SLOW)
MAX_SPEED = 1.0                    # Speed cap sent to rc.drive.set_max_speed() at start


def _scan_has_return(scan: Optional[Sequence[float]]) -> bool:
    if scan is None or len(scan) == 0:
        return False
    for sample in scan:
        try:
            value = float(sample)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value) and value > 0.0:
            return True
    return False

########################################################################################
# Functions
########################################################################################

def start():
    global speed, angle, lastSteer
    global frontDist, leftDist, rightDist
    global turning_mode, lidar_valid
    global left_dist_hist, right_dist_hist
    global turn_direction

    frontDist = 999.0
    leftDist = 0.0
    rightDist = 0.0
    turning_mode = False

    speed = 0.0
    angle = 0.0
    lastSteer = 0.0
    lidar_valid = False

    left_dist_hist = []
    right_dist_hist = []
    turn_direction = 0

    speed_angle_scale = 1.0

    rc.drive.set_max_speed(MAX_SPEED)
    rc.drive.stop()

    print("test started")

def wall_follow_angle(scan: Optional[Sequence[float]] = None):
    global lastSteer, frontDist, leftDist, rightDist, turning_mode, lidar_valid
    global left_dist_hist, right_dist_hist, turn_direction
    if scan is None:
        scan = rc.lidar.get_samples()
    if not _scan_has_return(scan):
        lidar_valid = False
        frontDist = 0.0
        leftDist = 0.0
        rightDist = 0.0
        return 0.0

    lidar_valid = True

    was_turning = turning_mode

    front_dist = rc_utils.get_lidar_average_distance(scan, 0, 20)
    frontDist = front_dist if front_dist > 0 else 999.0

    # Decide sampling mode from the front reading before pulling the side
    # distances, so the angle we look at below matches the situation we're
    # actually in this frame.
    #
    # Hysteresis: use a wider "exit" threshold than "entry" threshold so
    # frontDist hovering right around FRONT_TURN doesn't cause turning_mode
    # to flicker on/off every frame.
    front_turn_exit = FRONT_TURN * FRONT_TURN_EXIT_SCALE
    if turning_mode:
        turning_mode = frontDist < front_turn_exit
    else:
        turning_mode = frontDist < FRONT_TURN

    side_angle = TURN_SIDE_ANGLE if turning_mode else NORMAL_SIDE_ANGLE

    right_dist = rc_utils.get_lidar_average_distance(scan, side_angle, SIDE_WINDOW)
    left_dist  = rc_utils.get_lidar_average_distance(scan, 360 - side_angle, SIDE_WINDOW)

    # Smooth the raw side distances over a short window to reduce the
    # effect of a single noisy lidar scan on the steering decision.
    if DIST_SMOOTHING_FRAMES > 1:
        left_dist_hist.append(left_dist)
        right_dist_hist.append(right_dist)
        left_dist_hist = left_dist_hist[-DIST_SMOOTHING_FRAMES:]
        right_dist_hist = right_dist_hist[-DIST_SMOOTHING_FRAMES:]
        # Ignore zero (invalid) readings when averaging, but keep at least
        # the latest raw value if everything in the window was invalid.
        valid_left = [d for d in left_dist_hist if d > 0]
        valid_right = [d for d in right_dist_hist if d > 0]
        left_dist = sum(valid_left) / len(valid_left) if valid_left else left_dist
        right_dist = sum(valid_right) / len(valid_right) if valid_right else right_dist

    leftDist = left_dist
    rightDist = right_dist

    # Wall dead ahead - no amount of arc-fitting helps here, just spin toward
    # a committed direction, same as before. Still passed through the
    # smoothing filter below so we don't snap instantly to full lock.
    if turning_mode:
        if not was_turning:
            # Just entered turning_mode this frame - decide and commit to a
            # direction for the rest of the turn.
            diff = right_dist - left_dist
            if abs(diff) < TURN_DEADBAND:
                # Too close to call from lidar alone (near-symmetric
                # hairpin) - keep whichever way we were already leaning
                # instead of risking a noise-driven coin flip.
                turn_direction = 1 if lastSteer >= 0 else -1
            else:
                turn_direction = 1 if diff > 0 else -1
        raw_steer = turn_direction * MAX_STEER
        lastSteer = STEER_SMOOTHING * lastSteer + (1 - STEER_SMOOTHING) * raw_steer
        return lastSteer

    if right_dist == 0 and left_dist == 0:
        return lastSteer
    if right_dist == 0:
        right_dist = left_dist
    if left_dist == 0:
        left_dist = right_dist

    # ── Pure Pursuit steering ──────────────────────────────────────────────
    # Lateral offset from the corridor centerline. Positive = car sits closer
    # to the right wall (needs to move left), matching the sign convention
    # of the original (left - right) error term.
    lateral_offset = (left_dist - right_dist) / 2.0

    # Faster driving = look further ahead, slower driving = look closer
    base_lookahead = max(LOOKAHEAD_MIN, LOOKAHEAD_SCALE * max(speed, 0.2))

    # Scale that by how open the road ahead is: frontDist near FRONT_TURN
    # (approaching a wall/turn) -> shrink toward LOOKAHEAD_TURN_MULT;
    # frontDist at/above FRONT_OPEN (open straight) -> grow toward
    # LOOKAHEAD_STRAIGHT_MULT. Linearly interpolated in between.
    span = max(FRONT_OPEN - FRONT_TURN, 1e-6)  # avoid divide-by-zero if misconfigured
    openness = rc_utils.clamp((frontDist - FRONT_TURN) / span, 0.0, 1.0)
    curve_mult = LOOKAHEAD_TURN_MULT + openness * (LOOKAHEAD_STRAIGHT_MULT - LOOKAHEAD_TURN_MULT)

    lookahead = max(LOOKAHEAD_MIN, base_lookahead * curve_mult)

    # Curvature calculations (brings car to lookahead point) w/ pure pursuit formula: curvature = 2x / L^2.
    curvature = (2.0 * lateral_offset) / (lookahead ** 2)

    # Curvature to angle calculations
    steer_rad = math.atan(WHEELBASE * curvature)
    steer = -(steer_rad / MAX_STEER_RAD)   # negated to match original sign convention

    raw_steer = rc_utils.clamp(steer, -MAX_STEER, MAX_STEER)
    # Exponential smoothing on the final steer command - this is the main
    # fix for oscillation caused by frame-to-frame noise in the lidar-derived
    # steering angle.
    lastSteer = STEER_SMOOTHING * lastSteer + (1 - STEER_SMOOTHING) * raw_steer
    return lastSteer


def update():
    global speed, angle

    scan = rc.lidar.get_samples()
    angle = wall_follow_angle(scan)

    if not lidar_valid:
        speed = 0.0
        angle = 0.0
        rc.drive.stop()
    elif turning_mode:
        speed = TURN_SPEED
    elif frontDist < FRONT_SLOW:
        speed = SLOW_SPEED
    else:
        speed = NORMAL_SPEED
        speed *= (1.2 - abs(angle)) * speed_angle_scale
        speed = rc_utils.clamp(speed, 0.0, 1.0)

    if lidar_valid:
        angle = rc_utils.clamp(angle + angleOffset, -1.0, 1.0)
        rc.drive.set_speed_angle(speed, angle)

########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()