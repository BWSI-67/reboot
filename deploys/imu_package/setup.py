from glob import glob

from setuptools import find_packages, setup


package_name = "reboot"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="BWSI-67 phoenix-reboot team",
    maintainer_email="oak.wjr@gmail.com",
    description="Kalman-filtered attitude, velocity, pose, and wall following for Trial 2D.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "imu_preprocessor = reboot.preprocessor_node:main",
            "attitude_node = reboot.attitude_node:main",
            "localizer_node = reboot.localizer_node:main",
            "state_view = reboot.state_view:main",
        ],
    },
)
