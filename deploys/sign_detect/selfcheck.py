#!/usr/bin/env python3
"""Run this FIRST on a new car. It reports why sign detection will not work.

    python3 selfcheck.py

Every failure it prints has a fix printed next to it. It touches no hardware
beyond opening the TPU once and reading two ROS topics, and it never drives.

The checks are ordered by how often each one is the actual problem, learned
from bringing this up on our own car:

  1. TPU already owned by the driver stack   -- by far the most common
  2. ROS environment not sourced
  3. Files split across directories
  4. Python too old for sign_detector.py
"""

import glob
import os
import subprocess
import sys

OK, BAD, WARN = "  OK   ", "  FAIL ", "  WARN "
NEEDED = ["sign_detector.py", "sign_support.py", "ground_lights.py",
          "best_v5_edgetpu.tflite"]
problems = []


def report(status, label, detail="", fix=""):
    print(f"{status}{label}" + (f"  {detail}" if detail else ""))
    if status is BAD:
        problems.append((label, fix))
        if fix:
            print(f"         fix: {fix}")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:
        return ""


print("=" * 68)
print("sign_detect self-check")
print("=" * 68)

# -- 1. files ---------------------------------------------------------------
here = os.path.dirname(os.path.abspath(__file__)) or "."
missing = [f for f in NEEDED if not os.path.exists(os.path.join(here, f))]
if missing:
    report(BAD, "files", f"missing {missing}",
           "every file must sit in the SAME directory as the lab script: "
           "the code imports by module name and opens the model by relative path")
else:
    report(OK, "files", f"all {len(NEEDED)} present in {here}")

lab = os.path.join(here, "sign_detect_GF.py")
report(OK if os.path.exists(lab) else WARN, "entry point",
       "sign_detect_GF.py" if os.path.exists(lab)
       else "sign_detect_GF.py not here -- that is the script you actually run")

# -- 2. python --------------------------------------------------------------
v = sys.version_info
report(OK if v >= (3, 10) else BAD, "python", f"{v.major}.{v.minor}",
       "sign_detector.py uses `int | None`, which needs 3.10+")

# -- 3. Coral TPU present and free -----------------------------------------
apex = glob.glob("/dev/apex_*")
report(OK if apex else BAD, "coral device", apex[0] if apex else "no /dev/apex_*",
       "no PCIe Coral on this car -- sign detection cannot run")

if apex:
    groups = sh("id -nG")
    report(OK if "apex" in groups.split() else BAD, "apex group",
           groups or "?", "sudo usermod -aG apex $USER, then log out and back in")

    holder = sh(f"fuser -v {apex[0]} 2>&1 | tail -1")
    busy = holder and "python" in holder
    report(BAD if busy else OK, "tpu free",
           holder if busy else "not held by another process",
           "the Coral takes ONE process at a time. Stop whatever is using it, "
           "and if it is the driver's edgetpu_node, relaunch teleop with "
           "edgetpu_enable:=false -- otherwise every run fails with "
           "'Failed to load delegate from libedgetpu.so.1'")

    if not busy:
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate
            model = os.path.join(here, "best_v5_edgetpu.tflite")
            interp = Interpreter(model_path=model,
                                 experimental_delegates=[
                                     load_delegate("libedgetpu.so.1")])
            interp.allocate_tensors()
            shape = interp.get_input_details()[0]["shape"]
            report(OK, "tpu inference", f"model loads, input {list(shape)}")
        except ImportError as e:
            report(BAD, "tflite_runtime", str(e),
                   "pip3 install tflite_runtime")
        except Exception as e:
            report(BAD, "tpu inference", str(e).splitlines()[0],
                   "delegate failed even though the device looks free -- "
                   "check libedgetpu.so.1 is installed (ldconfig -p | grep edgetpu)")

# -- 4. ROS + the two topics the lab consumes -------------------------------
try:
    import rclpy  # noqa: F401
    report(OK, "rclpy", "importable")
    for topic in ("/camera/color", "/scan"):
        hz = sh(f"timeout 6 ros2 topic hz {topic} --window 20 2>/dev/null"
                f" | grep -o 'average rate: [0-9.]*' | head -1")
        report(OK if hz else BAD, f"topic {topic}", hz or "silent",
               "the driver stack is not publishing it; check racecar-teleop")
except ImportError:
    report(BAD, "rclpy", "not importable",
           "source /opt/ros/jazzy/setup.bash   (a plain ssh shell does not "
           "have the ROS environment, and the lab will die on import)")

print("=" * 68)
if problems:
    print(f"{len(problems)} problem(s) to fix before running the lab:")
    for label, _ in problems:
        print(f"  - {label}")
else:
    print("all clear. Run:  python3 sign_detect_GF.py   (hold the right bumper)")
    print("The bumper is the drive mux deadman -- without it the car ignores")
    print("every command and the lab looks broken.")
print("=" * 68)
