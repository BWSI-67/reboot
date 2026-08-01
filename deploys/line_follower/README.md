# LINE_SLOPE

Dual-band blue-line follower that steers on both where the line is
(offset) and which way it's heading (slope), plus its HSV tuning tool.

# line_slope.py
    The follower. Samples the line in two horizontal bands (near/far), steers on the near band's offset from center and the near-to-far slope, so it starts turning into a corner before the offset alone would show it.

# line_hsv.py
    Tuner for the same two bands. Shows whether each band actually sees the line, tints what the current threshold accepts, and can print the tape's real HSV so you can copy the numbers in.


RUNNING IT

    racecar sim line_slope.py

or on the physical car, after connecting to its Wi-Fi:

    python3 line_slope.py

To tune first:

    python3 line_hsv.py                 # terminal only
    python3 line_hsv.py --http          # browser: http://10.42.0.1:8002
    python3 line_hsv.py --probe         # print real H/S/V to copy in
    python3 line_hsv.py --mask          # stream the raw mask instead

HSV_MIN, HSV_MAX, NEAR_TOP, FAR_TOP, and BAND must be kept identical
between the two files -- line_hsv.py exists to tune line_slope.py's
numbers, not its own.


# TUNING line_slope.py

Oscillates / overshoots turns
    -> kP down

Doesn't turn into corners early enough
    -> kS up

Reacts to noise near dead-center
    -> SLOPE_DEADZONE up

Drifts to one side on straights
    -> TRIM

Line lost too often
    -> widen HSV_MIN/HSV_MAX (using line_hsv.py --probe), or lower
       MIN_AREA

Far band never contributes (slope always 0)
    -> FAR_TOP is aimed wrong -- check with line_hsv.py first, not by
       adjusting kS

Too fast into turns
    -> SLOW up, or lower MAX_SPEED

Hunts left/right while searching for a lost line
    -> LOST_SPEED down, or check that last_side is actually flipping
       (SLOPE_DEADZONE too high suppresses it)