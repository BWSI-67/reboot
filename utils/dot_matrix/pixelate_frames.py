"""
Preprocesses a folder of video/GIF frames into 8x24 black-and-white frames
ready to be played on the racecar's dot matrix display.

Usage:
    1. Set INPUT_PATH to the folder containing your raw extracted frames
       (from Photopea, Ezgif, ffmpeg, etc).
    2. Set OUTPUT_PATH to where you want the processed frames to go.
    3. If your frames are already black & white (like Bad Apple), leave
       USE_MASK = False. Otherwise set USE_MASK = True and tune the HSV
       target/tolerance below to isolate the color you want to appear as
       "on" pixels (see the guide's masking section for how to find these
       values with a test script).
    4. Run: python pixelate_frames.py
"""

import os
import re
import cv2
import numpy as np

INPUT_PATH = './image_sequence'
OUTPUT_PATH = './processed'

# Dot matrix display is 8 rows x 24 columns -> 3:1 aspect ratio
DISPLAY_WIDTH = 24
DISPLAY_HEIGHT = 8

# Set to True if your source frames have color/grayscale and need to be
# masked down to pure black & white first.
USE_MASK = False
HSV_TARGET = np.array([35, 20, 225])  # only used if USE_MASK = True
HSV_TOLERANCE = 10


def natural_sort_key(filename):
    """Sorts 'frame2.jpg' before 'frame10.jpg' instead of alphabetically."""
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', filename)]


def main():
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    files = sorted(os.listdir(INPUT_PATH), key=natural_sort_key)
    if not files:
        print(f'No files found in {INPUT_PATH}')
        return
    num_digits = len(str(len(files)))

    for i, f in enumerate(files):
        img = cv2.imread(f'{INPUT_PATH}/{f}')
        if img is None:
            print(f'Skipping unreadable file: {f}')
            continue

        if USE_MASK:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, HSV_TARGET - HSV_TOLERANCE, HSV_TARGET + HSV_TOLERANCE)
            img = mask
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to the display's resolution. NOTE: cv2.resize takes (width, height).
        out = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # Threshold to pure binary (black/white) now, so play_animation.py doesn't
        # need to redo it every frame during playback.
        _, out = cv2.threshold(out, 127, 255, cv2.THRESH_BINARY)

        # Zero-padded sequential naming makes loading trivial and fast later.
        out_name = f'frame_{i:0{num_digits}d}.png'
        cv2.imwrite(f'{OUTPUT_PATH}/{out_name}', out)

        if i % 500 == 0:
            print(f'Processed {i}/{len(files)} frames...')

    print(f'Done. {len(files)} frames written to {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
