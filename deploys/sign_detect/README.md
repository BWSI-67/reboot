# sign_detect

Support files for `labs/sign_detect_GF.py`.

| File | What it does |
|---|---|
| `sign_detector.py` | Coral Edge TPU inference (pure `tflite_runtime`, no torch) |
| `sign_support.py` | `findMaxW` (widest-gap direction) + `SignWatcher` (new-frame filter, K-of-N voting) |
| `best_v5_edgetpu.tflite` | Model weights, 4.2 MB |
| `show_detections.py` | Draws the boxes so you can check the model. Does not drive. |

## Deploy

These files and the lab script must land in the **same directory** — the script
finds them with `from sign_support import ...` and the relative path
`"best_v5_edgetpu.tflite"`.

```bash
scp deploys/sign_detect/*.py deploys/sign_detect/*.tflite labs/sign_detect_GF.py \
    racecar@10.42.0.1:/home/racecar/jupyter_ws/team1/labs/worksheet-6/

ssh racecar@10.42.0.1
cd ~/jupyter_ws/team1/labs/worksheet-6 && python3 sign_detect_GF.py
```

Hold the right bumper while it runs — that is the drive mux deadman.

## How a sign fires

1. `SignWatcher.poll()` runs inference on **new camera frames only** and appends
   one vote: the closest sign in that frame, or `None`.
2. A sign only counts as "close" when its box is at least `TRIGGER_H` (0.22) of
   the frame height. A stop sign across the room is detected but never votes.
3. `winner(VOTE_K)` returns a class once it holds 3 of the last 5 votes. With
   K > N/2 only one class can ever win, so there is no tie to break.
4. `ACTION[sign]` maps it to a state and a hold time. After the hold expires the
   car returns to `CRUISE`, and `COOLDOWN_S` keeps the same sign from firing
   again while the car drives past it.

The new-frame check matters more than it looks: the camera publishes at ~30 Hz
while `update()` runs at 60 Hz, so `racecar_core` hands out every frame roughly
twice. A repeat frame would cast a second vote from a single observation, which
halves the K-of-N filter's ability to reject a one-frame false positive. The
fingerprint sums ~200 sampled pixels — two orders of magnitude cheaper than
hashing the frame.

## Notes

- `sign_detector.py` uses `int | None`, so it needs Python 3.10+. The car runs
  3.12; the Mac's `racecar-venv` is 3.9 and cannot import it. Car only.
- Inference is pinned to core 3 at nice 10. Do not move it to cores 0-2 — those
  run the control loop and the ROS nodes.
- Only `GO_AROUND (3)`, `STOP (7)`, `YIELD (8)` are acted on. The model detects
  the other six classes; the lab ignores them.
- Zeros in the lidar scan mean "no echo", not "far". `findMaxW` leaves them as
  zeros on purpose — substituting a large value makes the gap search chase
  dropout clusters instead of open space.
