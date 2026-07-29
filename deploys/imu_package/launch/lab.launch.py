"""Launch the Trial 2D state-estimation stack.

Publishes /imu/primary, /attitude, /attitude/rpy, /velocity, /pose_estimate
and the map->base_link TF. Driving is done by the rc-API labs (examples/,
car/), which consume these topics — this launch file starts estimators only.

    ros2 launch reboot lab.launch.py
    ros2 run reboot state_view        # live dashboard

Keep the car still for ~2 s after launch so the gyro-bias capture can lock.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory("reboot")
    default_config = os.path.join(package_share, "config", "lab.yaml")
    config = LaunchConfiguration("config")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            Node(
                package="reboot",
                executable="imu_preprocessor",
                name="imu_preprocessor",
                output="screen",
                parameters=[config],
            ),
            Node(
                package="reboot",
                executable="attitude_node",
                name="attitude_node",
                output="screen",
                parameters=[config],
            ),
            Node(
                package="reboot",
                executable="localizer_node",
                name="localizer_node",
                output="screen",
                parameters=[config],
            ),
        ]
    )
