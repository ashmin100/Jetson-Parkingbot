import os
from setuptools import find_packages, setup

package_name = "parkingbot"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [os.path.join("resource", package_name)]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), ["launch/parkingbot.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TODO Maintainer",
    maintainer_email="todo@example.com",
    description="Gesture-controlled TurtleBot3 using jetson-inference poseNet and RealSense depth.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "parkingbot_node = parkingbot.vision_node:main",
        ],
    },
)
