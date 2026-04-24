from glob import glob
from pathlib import Path
from setuptools import find_packages, setup


package_name = "yumi_description"


def data_glob(pattern: str) -> list[str]:
    return [path for path in glob(pattern) if Path(path).is_file()]


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "LICENSE"]),
        (f"share/{package_name}/urdf", data_glob("urdf/*")),
        (f"share/{package_name}/urdf/Gazebo", data_glob("urdf/Gazebo/*")),
        (f"share/{package_name}/urdf/Grippers", data_glob("urdf/Grippers/*")),
        (f"share/{package_name}/urdf/Util", data_glob("urdf/Util/*")),
        (f"share/{package_name}/meshes", data_glob("meshes/*.stl")),
        (f"share/{package_name}/meshes/coarse", data_glob("meshes/coarse/*")),
        (f"share/{package_name}/meshes/gripper", data_glob("meshes/gripper/*")),
        (f"share/{package_name}/meshes/gripper/coarse", data_glob("meshes/gripper/coarse/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="axel",
    maintainer_email="axel@example.com",
    description="Canonical YuMi robot description package for MoveIt and Isaac Sim.",
    license="BSD-2-Clause",
    entry_points={
        "console_scripts": [
            "export_isaacsim_urdf = yumi_description.export_isaacsim_urdf:main",
        ],
    },
)
