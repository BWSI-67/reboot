"""
Plays a preprocessed frameset (from pixelate_frames.py) on the racecar's
dot matrix display.

Usage:
    1. scp the processed frames folder to the racecar, e.g.:
       scp -r ./processed/ racecar@192.168.1.<team number>:~/jupyter_ws/student/labs
    2. Place this script alongside that folder on the racecar (adjust
       FRAMES_PATH below if they're not in the same directory), and make
       sure the sys.path.insert line correctly points at the library folder.
    3. Run the script, then press START to begin playback. Press BACK to stop.
"""

import glob
import sys
import re

import cv2

# If this file is nested inside a folder in the labs folder, the relative path
# should be '../../library' instead. Adjust to match your folder structure.
sys.path.insert(0, '../../library')
import racecar_core

rc = racecar_core.create_racecar()

FRAMES_PATH = './processed'
FRAMES_PER_SECOND = 30


def natural_sort_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', path)]


# Load all frames on startup. Loading dynamically via glob means you never
# have to hardcode a frame count or naming convention here.
frame_paths = sorted(glob.glob(f'{FRAMES_PATH}/*.png'), key=natural_sort_key)
frames = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in frame_paths]
TOTAL_FRAMES = len(frames)
print(f'Loaded {TOTAL_FRAMES} frames.')

time = 0


# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    global time
    time = 0


# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed
def update():
    global time
    time += rc.get_delta_time()

    # Play FRAMES_PER_SECOND frames per second, looping at the end of the animation.
    curr_frame = int(time * FRAMES_PER_SECOND) % TOTAL_FRAMES
    rc.display.set_matrix(frames[curr_frame])


# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. Useful for debug prints without cluttering every frame.
def update_slow():
    pass


if __name__ == '__main__':
    rc.set_start_update(start, update, update_slow)
    rc.go()
