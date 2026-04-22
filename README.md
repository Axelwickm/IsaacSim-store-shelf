# IsaacSim Store Shelf

Minimal ROS 2 Humble Python workspace scaffold for the proper Isaac Sim + ROS 2 version of this project.


## What This Starts With

- A single Python ROS 2 package: `store_shelf_controller`
- One executable script: `controller.py`
- Docker Compose wired to a `ros:humble-ros-base`-derived image

## Run

```bash
docker compose up --build
```

The container will:

1. Source ROS 2 Humble
2. Build the workspace with `colcon`
3. Run `ros2 run store_shelf_controller controller`

Right now the controller only proves the ROS environment is there by importing and initializing `rclpy`, then printing/logging a hello-world message.

## Isaac Sim GUI

The compose file also includes an `isaacsim` service using the latest stable NVIDIA Isaac Sim container image: `nvcr.io/nvidia/isaac-sim:5.1.0`.

Use `ISAACSIM_MODE` to choose how Isaac Sim starts:

- `headed`: launches the windowed app with `./runapp.sh`
- `headless`: launches the streaming build with `./runheadless.sh -v`

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
```

Start headless:

```bash
ISAACSIM_MODE=headless docker compose up isaacsim
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
