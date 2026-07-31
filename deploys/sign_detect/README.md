# sign_detect

Traffic-sign detection (Coral Edge TPU) and wall light-strip detection, bolted
onto a widest-gap follower. Drop this folder on a RACECAR Neo and run it.

## Files

Everything here has to stay in **one directory**: the code imports by module
name and opens the model by relative path.

| File | Required | What it is |
|---|---|---|
| `sign_detect_GF.py` | yes | The entry point — the script you run |
| `sign_support.py` | yes | `findMaxW` gap follower + `SignWatcher` (frame de-dup, weighted voting, red-light latch) |
| `sign_detector.py` | yes | Edge TPU inference, pure `tflite_runtime` |
| `ground_lights.py` | yes | Wall light strips, HSV on the CPU. Also runs standalone as its own tuner |
| `best_v5_edgetpu.tflite` | yes | Model weights, 4.2 MB, 9 classes |
| `selfcheck.py` | no | Run this first. Names every blocker and prints the fix |
| `show_detections.py` | no | Live view of what the detector sees |

## Running it

```bash
scp -r sign_detect racecar@<your-car>:~/jupyter_ws/*/labs/

# on the car
source /opt/ros/jazzy/setup.bash          # a plain ssh shell has no ROS
cd ~/jupyter_ws/*/labs/sign_detect
python3 selfcheck.py                      # fix anything it flags
python3 sign_detect_GF.py                 # then hold the RIGHT BUMPER
```

The right bumper is the drive mux deadman. Without it the car ignores every
command and the lab looks broken.

The script must sit exactly two levels under the directory that holds
`library/`, because it reaches the rc API with `sys.path.insert(0,
"../../library")`. `~/jupyter_ws/<anything>/labs/sign_detect/` satisfies that.

## Two things that will bite you on a different car

**The Coral takes one process at a time.** On a stock car the driver stack's
`edgetpu_node` already owns it, and every run then dies with

```
ValueError: Failed to load delegate from libedgetpu.so.1
```

which says nothing about the real cause. Relaunch teleop with
`edgetpu_enable:=false`, or stop whatever holds `/dev/apex_0`. `selfcheck.py`
tells you which process it is.

**The light-strip HSV is our venue's, not yours.** LED hue depends on the strip
and on exposure. Re-measure before you trust it:

```bash
python3 ground_lights.py --racecar --http     # browser: http://<your-car>:8001
```

Hit pixels are tinted in their own colour. You want the strip fully covered and
the floor untouched. `--probe` prints the real H/S/V of the brightest pixels in
the region of interest, so you can copy the numbers into `COLOR_BANDS`.

## How a sign fires

1. `SignWatcher.poll()` runs inference on **new camera frames only** — the
   camera publishes at ~25-30 Hz while `update()` runs at 60 Hz, so every frame
   is handed out twice and a repeat would vote twice from one observation.
2. Each frame contributes, per class, `confidence * min(1, height / TRIGGER_H)`.
   A sign further away still accumulates, just slower, instead of being thrown
   away by a hard height cut — and a single weak or tiny false positive
   contributes ~0.2 and ages out of the window long before anything fires.
3. `winner(NEED)` returns a class once its sum over the 9-frame window reaches
   `NEED = 2.0` **and** it leads the runner-up by 1.5x.
4. `ACTION[sign]` maps it to a state and a minimum hold. The car resumes
   `CLEAR_S` after the sign leaves view, but never later than `MAX_HOLD_S` —
   without that cap a sign the car parks in front of never leaves, and the car
   waits forever.

Signs and light strips are read from the same frame, in that order: the sign
boxes are painted out of the light region first, or a red `DO_NOT_ENTER` at the
trackside reads as a red strip. That is also why the detector is opened to all
nine classes even though only `GO_AROUND`, `STOP` and `YIELD` trigger anything.

Speed arbitration is strictest-wins: a red strip outranks the signs, and both
outrank cruising.

## Tuning, in the order worth doing it

Everything you would touch is in the first 8 lines of `sign_detect_GF.py`.

| Symptom | Knob |
|---|---|
| Reacts too late / too early to a sign | `TRIGGER_H` (0.22 = box height as a fraction of frame height) |
| Fires on things that are not signs | `NEED` up, or `CONF` up |
| Misses real signs | `NEED` down, or `VOTE_N` up |
| Stops for too long | `MAX_HOLD_S`, and check whether the sign stays in frame |
| Lurches backward at a stop | `BRAKE` to 0.0 (speed 0 is ESC neutral, i.e. coast, so a short reverse is what actually brakes) |
| Light strips never / always trigger | `LIGHT_PX`, after re-measuring HSV |
