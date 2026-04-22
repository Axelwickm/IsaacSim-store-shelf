FROM ros:humble-ros-base

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    ros-humble-rclpy \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
