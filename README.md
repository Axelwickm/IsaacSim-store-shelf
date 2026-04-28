# THIS IS WIP and not yet fully functioning

# IsaacSim Store Shelf

Minimal ROS 2 Jazzy Python workspace scaffold for the proper Isaac Sim + ROS 2 version of this project.

NOTE: Only Nvidia GPU supported


## What This Starts With

- A single Python ROS 2 package: `controller`
- A Python ROS 2 motion package: `motion`
- One executable script: `controller.py`
- Docker Compose wired to a `ros:jazzy-ros-base`-derived image
- MoveIt 2 installed in the ROS 2 container image
- NVIDIA Isaac ROS cuMotion built from source in the ROS 2 container image

## Run

```bash
COMPOSE_PARALLEL_LIMIT=1 docker compose build ros2 isaacsim
docker compose up ros2 isaacsim
```

The ROS container will:

1. Source ROS 2 Jazzy
2. Build the workspace with `colcon`
3. Run `ros2 launch controller sim.launch.py`

Right now the controller proves the ROS environment is there by importing and initializing `rclpy`, then logging the selected launch parameters.

Pass simulation launch options as ROS launch arguments:

```bash
docker compose run --rm ros2 bash -lc "source /opt/ros/jazzy/setup.bash && colcon build --symlink-install && source install/setup.bash && ros2 launch controller sim.launch.py headless:=true scene:=store_shelf.usd"
```

The `ros2` compose service is GPU-enabled. cuMotion and CUDA vision inference
need the host NVIDIA runtime to be available inside the container.

## Motion Planning

The workspace now includes a `motion` package with a planner scaffold node:

```bash
docker compose run --rm ros2 bash -lc "source /opt/ros/jazzy/setup.bash && colcon build --symlink-install && source install/setup.bash && ros2 launch motion motion.launch.py"
```

The image installs:

- `ros-jazzy-moveit`

The image also clones and builds `isaac_ros_cumotion` from source in `/opt/isaac_ros_cumotion_ws`, because the matching Jazzy Debian package is not currently available from NVIDIA's apt repository for this base image.

For shell access to the GPU-enabled ROS 2 service:

```bash
docker compose run --rm ros2 bash
```

The `motion` node currently validates that MoveIt and cuMotion are present in the environment, then listens for `geometry_msgs/msg/PoseStamped` goals on `/motion/target_pose`. It is the package where the vision-to-planning bridge and MoveIt request wiring should live next.

## Store Demo With cuMotion

Build the ROS workspace inside the running `ros2` container after code changes:

```bash
colcon build --symlink-install
source install/setup.bash
```

Start the full store demo with cuMotion:

```bash
ros2 launch controller store_demo.launch.py \
  motion_pipeline_id:=isaac_ros_cumotion \
  motion_planner_id:=cuMotion
```

This launches:

- Isaac Sim manager control node
- robot state publisher
- ros2_control topic hardware and joint state broadcaster
- NVIDIA cuMotion planner launch
- MoveGroup using the `isaac_ros_cumotion` planning pipeline
- vision inference
- motion planner
- Isaac Sim direct `FollowJointTrajectory` executor

The YuMi cuMotion robot model is:

```text
/workspace/usd/robot/yumi_isaacsim.urdf
/workspace/usd/robot/yumi_isaacsim.xrdf
```

If cuMotion reports `INVALID_START_STATE_SELF_COLLISION`, inspect the cuMotion
sphere approximation before changing ignore pairs or sphere radii.

## cuMotion Sphere Visualization

Launch the store demo with RViz enabled:

```bash
ros2 launch controller store_demo.launch.py \
  motion_pipeline_id:=isaac_ros_cumotion \
  motion_planner_id:=cuMotion \
  use_moveit_rviz:=true
```

RViz opens with `src/yumi_moveit_config/rviz/cumotion_debug.rviz`, which shows:

- YuMi robot model
- `/cumotion/robot_spheres`
- `/viz_all_spheres/planner_attach_object`
- `/curobo/voxels`

The `/cumotion/robot_spheres` topic is published by:

```text
motion/cumotion_sphere_publisher
```

It loads the same XRDF/URDF as cuMotion, evaluates cuRobo link spheres at the
current `/joint_states`, and publishes a `visualization_msgs/MarkerArray` in
the `yumi_body` frame. Use this to compare cuMotion's robot collision model
against the visible robot before editing the XRDF.

If RViz is already running and no spheres appear, verify:

```bash
ros2 topic echo --once /cumotion/robot_spheres
ros2 topic hz /joint_states
```

## Robot Description

The canonical YuMi source now lives in the ROS 2 package `src/yumi_description`.

- Use the Xacro tree in `src/yumi_description/urdf` as the source of truth.
- Treat `usd/robot/yumi_isaacsim.urdf` as a generated Isaac Sim artifact.
- The checked-in YuMi USD is no longer the default robot source.
- Isaac Sim imports the generated URDF directly into the live stage at runtime.
- If you change the YuMi Xacro or meshes, trigger regeneration of the Isaac-ready URDF with:

```bash
docker compose run --rm ros2 bash -lc "source /opt/ros/jazzy/setup.bash && colcon build --symlink-install && source install/setup.bash && ros2 run yumi_description export_isaacsim_urdf"
```

Inside an already sourced ROS 2 container, the equivalent direct command is:

```bash
ros2 run yumi_description export_isaacsim_urdf
```

To choose the output path explicitly:

```bash
ros2 run yumi_description export_isaacsim_urdf -- --output /workspace/usd/robot/yumi_isaacsim.urdf
```

To validate/export the MoveIt-side `ros2_control` block as part of the URDF:

```bash
ros2 run yumi_description export_isaacsim_urdf -- --with-ros2-control --output /tmp/yumi_ros2_control.urdf
```

The Isaac Sim joint bridge publishes raw simulated joint states on
`/isaac_joint_states`. The ros2_control topic hardware consumes that topic, then
`joint_state_broadcaster` publishes the normalized MoveIt-facing `/joint_states`
topic.

This keeps MoveIt and Isaac Sim derived from the same YuMi description instead of maintaining separate robot definitions by hand.

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

### ABB YuMi Robot Assets

- Source lineage: Orebro University `yumi` repository, previously vendored in `../PickerDemo/third_party/yumi`
- Canonical local robot package: `src/yumi_description`
- License: BSD-2-Clause-style text in `src/yumi_description/LICENSE`

## Troubleshooting

If Isaac Sim in Docker fails with Vulkan or GPU initialization errors such as:

- `vkCreateInstance failed`
- `ERROR_INCOMPATIBLE_DRIVER`
- `Failed to create any GPU devices`

check the host `nvidia-container-toolkit` version first.

An older toolkit can break Vulkan inside containers even when `nvidia-smi` works. Updating to the latest upstream NVIDIA container toolkit release fixed this setup.
