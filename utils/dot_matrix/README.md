# Racecar Dot Matrix Animation Player

Plays a video/GIF as an animation on the racecar's 24x8 dot matrix display,
following the [RACECAR Neo Bad Apple guide](https://mitracecarneo.github.io/racecar-neo-library/guides/bad-apple.html).

## How it works

The dot matrix display is **binary only** — each of the 192 pixels is either
on (lit orange) or off. There's no grayscale or color to work with, so the
whole pipeline is about reducing a normal video/GIF down to a sequence of
24x8 black-and-white frames.

Two scripts handle this:

| Script | Runs on | Purpose |
|---|---|---|
| `pixelate_frames.py` | Your laptop | Converts a folder of extracted frames into 24x8 binary PNGs |
| `play_animation.py` | The racecar | Loads the processed frames and plays them on the display |

## Requirements

- Python 3
- OpenCV: `pip install opencv-python --break-system-packages`

## Workflow

### 1. Get your source frames

Extract frames from a video or GIF (Photopea's "Export Layers", Ezgif,
`ffmpeg`, etc.) into a single folder, e.g. `image_sequence/`.

### 2. Pixelate them (on your laptop)

Edit the constants at the top of `pixelate_frames.py`:

```python
INPUT_PATH = './image_sequence'   # your extracted frames
OUTPUT_PATH = './processed'       # where the 24x8 output goes
USE_MASK = False                  # True if you need to isolate a color/subject
```

Run it:

```bash
python3 pixelate_frames.py
```

This writes `frame_00.png`, `frame_01.png`, ... to `OUTPUT_PATH`, each
resized to 24x8 and thresholded to pure black/white.

### 3. Copy the processed frames to the racecar

Either `scp` them over from your laptop:

```bash
scp -r ./processed/ racecar@192.168.1.<team number>:~/jupyter_ws/team1/labs/utility/
```

or upload a zipped version through Jupyter's file browser and unzip it in a
cell/terminal on the racecar:

```bash
unzip -o processed.zip -d processed
```

`processed/` should end up in the same directory as `play_animation.py`.

### 4. Play it (on the racecar)

Set the playback rate in `play_animation.py` to match your source (e.g. a
GIF with a 100ms per-frame delay is ~10 fps):

```python
FRAMES_PER_SECOND = 10
```

Then run:

```bash
python3 play_animation.py
```

You should see `Loaded N frames.` printed (N > 0). Press **START** to begin
playback, **BACK** to stop.

## Troubleshooting

**`Loaded 0 frames.` / `ZeroDivisionError`**
The `processed/` folder isn't where `play_animation.py` is looking. Usually
means the `scp`/upload step didn't actually land the files next to the
script — check you're running `scp` from your laptop (not from inside an
SSH session on the racecar), and that the destination path is correct.

**Almost everything on the display is lit orange**
The display's on/off polarity may be inverted from what the script assumes
(0 = off, 255 = on). Test it directly:

```python
rc.display.set_matrix(np.zeros((8, 24), dtype=np.uint8))
```

If an all-zero array lights up the whole display, the polarity is flipped.
Fix it in `pixelate_frames.py` by swapping `cv2.THRESH_BINARY` for
`cv2.THRESH_BINARY_INV`.

**The image is unrecognizable at 24x8**
Photo-realistic sources compress very badly at this resolution (192 total
pixels). High-contrast, simple silhouettes work best. Try:
- Cropping tightly on the subject before pixelating
- Setting `USE_MASK = True` and tuning `HSV_TARGET`/`HSV_TOLERANCE` to
  isolate just the subject instead of a straight grayscale threshold
- Matching your source's aspect ratio to the display's 3:1 ratio before
  resizing, to avoid stretching
