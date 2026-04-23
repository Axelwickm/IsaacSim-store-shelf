from glob import glob

from setuptools import find_packages, setup


package_name = "vision"


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "torch",
        "torchvision",
    ],
    zip_safe=True,
    maintainer="axel",
    maintainer_email="axel@example.com",
    description="Vision training package for the Isaac Sim store shelf project.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "train_vision = vision.train_vision:main",
            "view_dataset = vision.view_dataset:main",
        ],
    },
)
