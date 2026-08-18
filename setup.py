from glob import glob
import os

from setuptools import find_packages, setup

package_name = "robot_parameter_identification"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"),
         glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "profiles"),
         glob(package_name + "/profiles/*.yaml")),
        (os.path.join("share", package_name, "profiles", "templates"),
         glob(package_name + "/profiles/templates/*.yaml")),
    ],
    package_data={
        package_name: [
            "profiles/*.yaml",
            "profiles/templates/*.yaml",
            "dashboard/static/*.html",
            "dashboard/static/*.css",
            "dashboard/static/*.js",
            "dashboard/static/vendor/*.js",
            "dashboard/static/vendor/addons/controls/*.js",
            "dashboard/static/vendor/addons/loaders/*.js",
        ],
    },
    include_package_data=True,
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Yizhong Zhang",
    maintainer_email="yizhongzhang1989@users.noreply.github.com",
    description="Robot-agnostic dynamic parameter identification with a "
                "web dashboard.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "dashboard = robot_parameter_identification.dashboard.node:main",
        ],
    },
)
