#!/bin/bash
# One command to race:   bash ~/race/go.sh   [script] [args...]
#
# ALWAYS restarts the IMU stack (~/imu.sh) first, then runs race.py once
# /attitude is actually publishing. Rationale: the stock teleop launch runs
# imu_fusion_enable:=false, and with /attitude silent wf_lite quietly falls
# back to lidar-only and corners worse. imu.sh also re-captures the gyro bias,
# which is exactly what you want at the start of every run.
#
#   bash ~/race/go.sh                          # IMU reset, then race
#   bash ~/race/go.sh test_elevator.py --hold  # any script in ~/race
#   SKIP_LAUNCH=1 bash ~/race/go.sh            # stack is fine, just race
#
# !! KEEP THE CAR STILL ON THE GROUND for the first ~3 s: imu.sh captures the
#    gyro bias then, and nothing publishes until the bias locks.

# NOTE: no `set -u` -- ROS's setup.bash reads unbound vars and dies under it
source /opt/ros/jazzy/setup.bash
cd "$(dirname "$0")"
mkdir -p logs

SCRIPT="${1:-race.py}"
[ $# -gt 0 ] && shift

# /attitude is BEST_EFFORT; `ros2 topic echo` subscribes RELIABLE and would
# never match it (that mismatch made this script wait out its whole timeout
# while the topic was happily publishing at 210 Hz). `topic hz` matches --
# same idiom imu.sh itself uses.
attitude_alive() {
    timeout 4 ros2 topic hz /attitude --window 5 2>/dev/null \
        | grep -q "average rate"
}

if [ "${SKIP_LAUNCH:-0}" != "1" ]; then
    if attitude_alive; then
        echo "[go] /attitude already live -- stack is healthy, not restarting"
        echo "[go] (force a fresh gyro-bias capture with:  bash ~/imu.sh)"
        SKIP_LAUNCH=1
    fi
fi

if [ "${SKIP_LAUNCH:-0}" != "1" ]; then
    echo "============================================================"
    echo ">>> RESTARTING IMU STACK (~/imu.sh)"
    echo ">>> KEEP THE CAR STILL ON THE GROUND -- gyro bias capture"
    echo "============================================================"
    IMU_LOG="logs/imu_$(date +%Y%m%d_%H%M%S).log"
    # imu.sh ends in `exec ros2 launch`, so it never returns: background it
    # setsid: put imu.sh in its OWN session. With a plain `nohup ... &` it
    # stays in this shell's process group, so Ctrl-C on race.py killed the
    # IMU stack too -- and every single test run then paid a 30 s restart.
    setsid bash ~/imu.sh > "$IMU_LOG" 2>&1 < /dev/null &
    IMU_PID=$!
    echo "[go] imu.sh pid $IMU_PID, log $IMU_LOG"

    ATT_OK=0
    for i in $(seq 1 45); do        # poll every 1 s so a healthy stack is caught fast
        sleep 1
        if attitude_alive; then
            echo "[go] /attitude live after ${i} s"
            ATT_OK=1
            break
        fi
        if [ $((i % 10)) -eq 0 ]; then
            echo "[go] still waiting for /attitude ... (${i} s)"
            tail -2 "$IMU_LOG" 2>/dev/null | sed 's/^/[imu] /'
        fi
    done

    # trust the loop's verdict: probing again here raced with itself and
    # printed "live" and "still silent" back to back
    if [ "$ATT_OK" != "1" ]; then
        echo "[go] WARNING: /attitude still silent."
        echo "[go]   wf_lite will run LIDAR-ONLY (worse cornering)."
        echo "[go]   check $IMU_LOG -- a 'sudo systemctl restart racecar-teleop'"
        echo "[go]   inside imu.sh may be waiting for a password."
    fi
fi

ENC=$(timeout 3 ros2 topic hz /encoder/speed --window 5 2>/dev/null | grep -o "average rate:.*" | head -1)
echo "[go] sensors: attitude=$([ "${ATT_OK:-1}" = "1" ] && echo LIVE || echo SILENT)  encoder=${ENC:-SILENT}"
echo "[go] launching $SCRIPT"
exec python3 "$SCRIPT" "$@"
