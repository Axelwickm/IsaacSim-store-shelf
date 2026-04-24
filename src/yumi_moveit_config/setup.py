from glob import glob
from pathlib import Path
from setuptools import find_packages, setup


package_name = "yumi_moveit_config"


def data_glob(pattern: str) -> list[str]:
    return [path for path in glob(pattern) if Path(path).is_file()]


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", data_glob("config/*")),
        (f"share/{package_name}/launch", data_glob("launch/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="axel",
    maintainer_email="axel@example.com",
    description="MoveIt configuration package for ABB YuMi in Isaac Sim store shelf.",
    license="BSD-2-Clause",
)
