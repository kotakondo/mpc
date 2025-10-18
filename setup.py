from setuptools import setup, find_packages
from glob import glob
import os

package_name = "mpc"


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        # ament package index
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        # package.xml
        (os.path.join("share", package_name), ["package.xml"]),
        # install ALL configs in config/
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        # install ALL launch files in launch/
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Lucas",
    maintainer_email="yixuany@mit.edu",
    description="Model Predictive Controller in ROS2",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "mpc_node = mpc.mpc_node:main",
        ],
    },
)
