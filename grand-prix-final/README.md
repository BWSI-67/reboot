# Grand Prix — final algorithm

Everything the car actually raced with, and nothing else. Three files.

| File | What it is |
|---|---|
| `race_deploy.py` | Entry point. Loads a controller + params from the environment, adapts the car's sensors to the controller contract, drives. |
| `wf_lite.py` | The controller: lidar-only racer (model twin + scan carry-forward + ets steering). |
| `wf_lite_finalloop_params.json` | The tuned parameters raced. |

## Run

```bash
cd ~/jupyter_ws/team1/labs/racing
PARAMS=wf_lite_finalloop_params.json CTRL=wf_lite.py CHECK=0 VMAX=4.0 python3 race_deploy.py
```

`CHECK=1` (the default) is a static sensor check that never drives — use it to
confirm the lidar, gyro and encoder are alive before a run. Climb the speed
ladder `VMAX = 1.5 → 2.0 → 2.5 → 3.0 → 4.0` on a new track.

START launches, BACK reclaims manual control.

## Raced parameters

```
window_deg      85.93    steering search window, each side
ray_width_deg   28.65    smoothing width of the openness average
weight_range_m   2.99    how much a left/right openness difference steers
kp              0.032    openness blend -> steering angle
half_width      0.343    swath half-width for the braking law
a_brake         12.0     how late to brake
a_lat           16.0     how fast to corner
wall_rep         0.4     1/d centring pull off the outside sweeps
```

Two module constants in `wf_lite.py` matter as much as the JSON and are not
tunable knobs:

- `SLEW = 0.20` — per-tick steering rate limit. At 0.13 the car was late both
  turning in AND straightening out; the slow un-turn read as "it turns too
  much".
- `OPEN_M = 3.0` / `OPEN_DAMP = 0.35` — in open floor (nothing within 3 m
  ahead) the steering is damped toward straight. Openness-following has no
  gradient to track out there: every heading looks equally good, so the car
  turns, re-evaluates, turns again, and circles in place at the anti-stall
  creep speed. Damping makes it cross the open area and re-acquire a wall.
  A real corner always has a near wall ahead, so cornering is untouched.

## Notes

- Lidar is the only required sensor. Gyro (`/attitude`) and wheel speed
  (`/encoder/speed`) are used as corrections when present and sane, and
  silently skipped otherwise — a dead IMU degrades cornering, it does not
  stop the car.
- `spd_scale` / `u_floor` exist in `PARAM_SPEC` but are deliberately absent
  from the raced JSON, so both take their defaults (1.0 / 0.0 — full speed,
  no throttle floor).
- The camera is never touched. `racecar_core` subscribes to it lazily and
  then decodes every 640x480 frame at 30 Hz inside the same executor that
  spins the control loop, which costs loop rate for the rest of the run —
  and `wf_lite`'s model twin steers badly on a jittery loop.
