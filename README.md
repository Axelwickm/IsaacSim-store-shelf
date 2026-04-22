# IsaacSim Store Shelf

Minimal ROS 2 Jazzy Python workspace scaffold for the proper Isaac Sim + ROS 2 version of this project.


## What This Starts With

- A single Python ROS 2 package: `controller`
- One executable script: `controller.py`
- Docker Compose wired to a `ros:jazzy-ros-base`-derived image

## Run

```bash
docker compose up --build
```

The container will:

1. Source ROS 2 Jazzy
2. Build the workspace with `colcon`
3. Run `ros2 launch controller sim.launch.py`

Right now the controller proves the ROS environment is there by importing and initializing `rclpy`, then logging the selected launch parameters.

Pass simulation launch options as ROS launch arguments:

```bash
docker compose run --rm ros2 bash -lc "source /opt/ros/jazzy/setup.bash && colcon build --symlink-install && source install/setup.bash && ros2 launch controller sim.launch.py headless:=true scene:=store_shelf.usd"
```

## Isaac Sim GUI

The compose file also includes an `isaacsim` service using the latest stable NVIDIA Isaac Sim container image: `nvcr.io/nvidia/isaac-sim:5.1.0`.

The Isaac Sim service starts only the manager. Publish a command to `isaacsim_manager/control` to start, stop, or restart Isaac Sim.

Headed mode needs local X11 access on Linux:

```bash
xhost +local:
```

Before the first Isaac Sim run, create the writable bind-mount directories and hand them to the container user NVIDIA documents for Isaac Sim 5.1.0:

```bash
mkdir -p .docker/isaacsim/cache/main .docker/isaacsim/cache/computecache .docker/isaacsim/logs .docker/isaacsim/config .docker/isaacsim/data .docker/isaacsim/pkg
sudo chown -R 1234:1234 .docker/isaacsim
```

Start headed:

```bash
docker compose up isaacsim
docker compose run --rm ros2 bash -lc "source /opt/ros/jazzy/setup.bash && ros2 topic pub --once /isaacsim_manager/control std_msgs/msg/String '{data: start}'"
```

Start headless:

```bash
docker compose up isaacsim
docker compose run --rm ros2 bash -lc "source /opt/ros/jazzy/setup.bash && ros2 topic pub --once /isaacsim_manager/control std_msgs/msg/String '{data: start_headless}'"
```

This service keeps its cache, config, logs, and app data in repo-local bind mounts under `.docker/isaacsim/`.

If Isaac Sim starts but fails with Vulkan initialization errors inside Docker, this compose service also sets `NVIDIA_DRIVER_CAPABILITIES=all` so the container gets the graphics and display driver libraries needed for Vulkan/X11, not just compute access.

On hybrid Intel/NVIDIA laptops, the compose file also forces PRIME render offload to the NVIDIA GPU with `__NV_PRIME_RENDER_OFFLOAD=1` and `__VK_LAYER_NV_optimus=NVIDIA_only`, which helps Vulkan apps avoid selecting the integrated GPU first.

## Credits and Licenses

### Supermarket Potato Chips Shelf Asset

- Author: Rendevr
- License: CC Attribution
- Source: https://sketchfab.com/3d-models/supermarket-potato-chips-shelf-asset-4e4ccc3074f0474bbfa23611c46a4029

## Troubleshooting

If Isaac Sim in Docker fails with Vulkan or GPU initialization errors such as:

- `vkCreateInstance failed`
- `ERROR_INCOMPATIBLE_DRIVER`
- `Failed to create any GPU devices`

check the host `nvidia-container-toolkit` version first.

An older toolkit can break Vulkan inside containers even when `nvidia-smi` works. Updating to the latest upstream NVIDIA container toolkit release fixed this setup.
