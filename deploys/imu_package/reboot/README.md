# STATE_ESTIMATION

Attitude, velocity, and pose estimation from the RealSense IMU and wheel encoder (two Extended Kalman Filters behind three ROS 2 nodes). This is the one part of REBOOT that doesn't follow the "67 lines or fewer" rule. 


# FILES

Everything here has to stay in one package: the nodes import the filters
and math helpers by relative module (.attitude_ekf, .ekf, .math_utils,
.gyro_bias), so this only runs installed as a ROS 2 package, not copied
around loose like the gap-follower labs.

preprocessor_node.py  (required)
    Rotates the raw IMU into the body frame, removes the gyro bias, republishes as /imu/primary. Nothing else works without this running first.

math_utils.py  
    Shared angle/quaternion helpers: wrap_angle, quat_from_rpy,
    rpy_from_quat, validate_rotation. No ROS import.

gyro_bias.py 
    GyroBiasEstimator: averages a stationary startup window to find and lock the gyro bias. No ROS import.

attitude_ekf.py 
    The 3-DOF [roll, pitch, yaw] filter. Gyro predicts, accelerometer's gravity corrects. No ROS import, pytest-able standalone.

attitude_node.py  
    Entry point. runs attitude_ekf.py on /imu/primary, publishes
    /attitude and /attitude/rpy.

ekf.py  
    The 4-DOF [x, y, yaw, forward_speed] Ackermann filter. No ROS import.

localizer_node.py 
    Entry point. Runs ekf.py on /imu/primary + /encoder/speed,
    publishes /velocity and /pose_estimate.

state_view.py  (optional)
    Live terminal dashboard of every topic above. Run this first when something looks wrong.


# RUNNING IT

    colcon build --packages-select reboot
    source install/setup.bash

    # hold the car still (nothing downstream publishes until this locks a bias)
    ros2 run reboot imu_preprocessor

    # in two more terminals, once "gyro bias locked" prints above
    ros2 run reboot attitude_node
    ros2 run reboot localizer_node

    # and to watch it work
    ros2 run reboot state_view

imu_preprocessor publishes nothing for the first ~2 seconds while it
captures a stationary gyro bias. Don't move the car until it logs
"gyro bias locked": moving it during capture just restarts the window.

state_view refreshes 10x a second and tags every topic with how fresh it is. A row that goes STALE is almost always the actual bug -- check that node's inputs before doubting the filter itself.


# TWO THINGS THAT WILL BITE YOU ON A DIFFERENT CAR

The yaw gyro reads ~8% high on this platform. imu_preprocessor corrects
it with gyro_scale_z = 0.9186, fit against scan-matched ground truth on
this specific car. Re-verify on a different one: spin exactly 2 full
turns, read the resulting yaw, then set

    gyro_scale_z *= 720 / measured_deg

The encoder can go stale without looking broken. It only publishes while the motor is actually driving, so hand-pushing the car or a dead encoder, both show up the same way: /velocity falls back to 0 (or the commanded speed, if use_cmd_speed_fallback is on). 

Use_cmd_speed_fallback defaults to false on purpose: /drive keeps streaming even with the deadman released, so fusing it while the car is stationary fakes motion into the pose. Only flip it on if the encoder is confirmed dead.


# HOW THE PIPELINE FIRES

1. imu_preprocessor is the only node that touches the raw sensor. It
   rotates the optical-frame IMU into the body frame, decimates to
   ~100 Hz, and -- after the one-time bias capture -- subtracts the
   locked gyro bias from every sample before publishing /imu/primary.

2. attitude_node predicts roll/pitch/yaw by integrating the gyro every
   /imu/primary message, then corrects roll/pitch with the
   accelerometer's gravity direction whenever |accel| is close enough to
   g to trust it (outside that band -- braking, cornering, bumps -- the
   correction is skipped and the gyro is trusted alone). Yaw has no absolute correction by default, so its uncertainty grows without bound. the bias capture is what keeps its real-world drift to a few degrees per minute anyway.

3. localizer_node predicts x/y/yaw with an Ackermann (bicycle) motion
   model, using a gyro yaw rate that's first clamped to whatever the physical steering geometry could actually produce (an outlier gate against gyro glitches, not a hard limit). Speed is never integrated from acceleration -- it's a directly measured quantity, fused every cycle from the wheel encoder (or, only if enabled, the commanded speed) -- so it can never run away the way a double-integrated accelerometer would.

4. Both filters publish independently; state_view just listens to
   everything and shows it on one screen.


# TUNING, IN THE ORDER WORTH DOING IT

Everything you'd touch is a declared ROS parameter on the relevant node.

Yaw drifts steadily even at rest
    -> gyro_scale_z (imu_preprocessor) -- re-run the 2-turn calibration on this car

Nothing publishes on /imu/primary
    -> bias_capture_sec / bias_capture_still_std (imu_preprocessor) -- car probably wasn't held still long enough

/velocity reads 0 while clearly moving
    -> Encoder is stale -- check state_view's ENCODER row. Only enable
       use_cmd_speed_fallback (localizer_node) if it's actually dead

Pose jumps when you pick the car up by hand
    -> YAW_BOUND_MARGIN in localizer_node.py -- the steering-bound clamp
       is an outlier gate, not a hard limit, and a violent hand-turn can still get partially clipped. Re-anchor via /initialpose or the /pose_estimate/reset service afterward

Along-track position drifts vs. a taped straight
    -> encoder_scale (localizer_node) -- wheel typically over-reads a few percent

Yaw won't hold better than a few deg/min
    -> use_mag (attitude_node / imu_preprocessor) -- off by default;
       only worth enabling with a real hard/soft-iron magnetometer gyro alone