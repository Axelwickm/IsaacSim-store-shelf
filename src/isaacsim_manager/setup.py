from setuptools import find_packages, setup


package_name = "isaacsim_manager"


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="axel",
    maintainer_email="axel@example.com",
    description="ROS 2 manager node for coordinating a running Isaac Sim process.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "manager = isaacsim_manager.manager:main",
        ],
    },
)
