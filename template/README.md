# Files

# general_gap_follow.py
    LiDAR widest-gap follower. Scans a +/-110 degree window in front of
    the car, steers toward the widest open gap.

# general_line_follow.py
    Blue-line camera follower. Crops/downsamples the frame, finds the
    largest blue contour, steers to keep it centered.


# RUNNING IT

    racecar sim gap_follow.py     # or line_follow.py

or on the physical car, after connecting to its Wi-Fi:

    python3 gap_follow.py         # or line_follow.py


# TUNING

Oscillates / overshoots turns
    -> kP down

Reacts too slowly to a closing gap/line
    -> kP up, or kD up

Drifts to one side
    -> TRIM (line) or the +0.04 bias in gap_follow.py

Line lost too often
    -> widen HSV_MIN/HSV_MAX, or lower MIN_AREA

Too fast into turns
    -> SLOW up (line), or lower CRUISE (either)

line_follow.py prints LOST and slows to a crawl when it can't find the line rather than stopping outright. It keeps the last steering angle so losing the line doesn't kill the run.