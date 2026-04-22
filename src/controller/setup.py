from glob import glob

from setuptools import find_packages, setup


package_name = "controller"


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="axel",
    maintainer_email="axel@example.com",
    description="Minimal Python ROS 2 controller scaffold for Isaac Sim store shelf.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "controller = controller.controller:main",
        ],
    },
)
