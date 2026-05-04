import json
import os
import random
import time
import uuid
from pathlib import Path

import rclpy
from std_msgs.msg import String

from .scene_construction import (
    DEFAULT_COLLECT_VISION_SCENE,
    VISION_CAMERA_LINK_PRIM_NAME,
    VISION_CAMERA_OPTICAL_FRAME_PRIM_NAME,
    add_robot_owned_camera,
    apply_capture_semantics,
    build_item_position_pools,
    construct_scene,
    find_prim_named,
    randomize_item_position_pools,
    randomize_robot_arm_joints,
    randomize_store_demo_robot_arm_joints,
    set_prim_local_translation,
)
from .vision_panel import DEFAULT_DEBUG_IMAGE_TOPIC, IsaacSimVisionPanel


DEFAULT_REPLICATOR_OUTPUT_DIR = "/workspace/collect_vision_data_output"
DEFAULT_CAPTURES_PER_RUN = 10
DEFAULT_CAPTURE_INTERVAL_SECONDS = 0.35
DEFAULT_CAPTURE_WARMUP_SECONDS = 1.0
DEFAULT_CART_LEFT_OFFSET_METERS = 0.0
DEFAULT_CART_RIGHT_OFFSET_METERS = 2.0
DEFAULT_CART_SLIDE_SPEED_METERS_PER_SECOND = 0.25
DEFAULT_STORE_DEMO_CART_MOTION_ENABLED = False
DEFAULT_STORE_DEMO_RESET_INTERVAL_SECONDS = 120.0
DEFAULT_STORE_DEMO_IDLE_RESET_SECONDS = 6.0
DEFAULT_GROUND_TRUTH_ITEMS_TOPIC = "/vision/ground_truth_items"
DEFAULT_COORDINATOR_STATE_TOPIC = "/motion/coordinator_state"
DEFAULT_MOTION_RESET_TOPIC = "/motion/reset"
DEFAULT_GROUND_TRUTH_ITEMS_PUBLISH_PERIOD_SECONDS = 0.2
SIMULATION_STEP_SECONDS = 1.0 / 60.0

_replicator_capture_enabled = False
_timeline_autoplay_enabled = False
_vision_panel = None
_ground_truth_items_node = None
_ground_truth_items_publisher = None
_motion_reset_publisher = None
_coordinator_state_subscription = None
_latest_coordinator_state = None
_ground_truth_items_publish_elapsed = 0.0
_trajectory_executors = []
_flow_state = None


def _play_timeline_if_needed() -> bool:
    if not _timeline_autoplay_enabled:
        return False

    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        return False

    timeline.play()
    return True


def _timeline_is_playing() -> bool:
    import omni.timeline

    return omni.timeline.get_timeline_interface().is_playing()


def _cart_translation(
    cart_base_translation: tuple[float, float, float],
    x_offset: float,
) -> tuple[float, float, float]:
    return (
        cart_base_translation[0] + x_offset,
        cart_base_translation[1],
        cart_base_translation[2],
    )


def _set_cart_x_offset(cart_prim, cart_base_translation, x_offset: float) -> None:
    set_prim_local_translation(cart_prim, _cart_translation(cart_base_translation, x_offset))


def _random_cart_x_offset(flow_state: dict) -> float:
    return random.uniform(
        flow_state["cart_left_offset"],
        flow_state["cart_right_offset"],
    )


def _schedule_next_store_demo_reset(flow_state: dict) -> None:
    flow_state["next_reset_monotonic"] = (
        time.monotonic() + flow_state["reset_interval_seconds"]
    )


def _prim_world_center(prim) -> list[float] | None:
    from pxr import Usd, UsdGeom

    try:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            ["default", "render"],
            useExtentsHint=True,
        )
        center = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox().GetMidpoint()
        return [float(center[0]), float(center[1]), float(center[2])]
    except Exception:
        try:
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            translation = matrix.ExtractTranslation()
            return [float(translation[0]), float(translation[1]), float(translation[2])]
        except Exception:
            return None


def _publish_ground_truth_items(flow_state: dict) -> None:
    global _ground_truth_items_publish_elapsed
    if _ground_truth_items_publisher is None:
        return
    _ground_truth_items_publish_elapsed += SIMULATION_STEP_SECONDS
    if _ground_truth_items_publish_elapsed < DEFAULT_GROUND_TRUTH_ITEMS_PUBLISH_PERIOD_SECONDS:
        return
    _ground_truth_items_publish_elapsed = 0.0

    items = []
    for pool in flow_state.get("item_position_pools", []):
        for item_prim in pool:
            if not item_prim.IsValid() or not item_prim.IsActive():
                continue
            center_xyz = _prim_world_center(item_prim)
            if center_xyz is None:
                continue
            items.append(
                {
                    "name": item_prim.GetName(),
                    "path": item_prim.GetPath().pathString,
                    "center_xyz": center_xyz,
                }
            )

    message = String()
    message.data = json.dumps(
        {
            "frame_id": "world",
            "items": items,
        },
        sort_keys=True,
    )
    _ground_truth_items_publisher.publish(message)


def _handle_coordinator_state(message: String) -> None:
    global _latest_coordinator_state
    try:
        payload = json.loads(message.data)
    except json.JSONDecodeError:
        return
    _latest_coordinator_state = payload if isinstance(payload, dict) else None


def _controller_is_idle() -> bool:
    if not isinstance(_latest_coordinator_state, dict):
        return False
    active_step = _latest_coordinator_state.get("active_step")
    assigned_target = _latest_coordinator_state.get("assigned_target")
    return not isinstance(active_step, dict) and not isinstance(assigned_target, dict)


def _publish_motion_reset(reason: str) -> None:
    if _motion_reset_publisher is None:
        return
    message = String()
    message.data = json.dumps(
        {
            "reason": reason,
            "stamp_ns": int(time.time_ns()),
        },
        sort_keys=True,
    )
    _motion_reset_publisher.publish(message)


def _randomize_items(item_pools: list[list]) -> list[str]:
    shuffled_groups = randomize_item_position_pools(item_pools)
    print(
        "[store_shelf] Randomized item scene: "
        + ("item_groups=" + ", ".join(shuffled_groups) if shuffled_groups else "none"),
        flush=True,
    )
    return shuffled_groups


def _setup_replicator_capture(stage, robot_prim_path: str) -> str:
    global _replicator_capture_enabled

    import omni.replicator.core as rep

    output_dir = os.environ.get(
        "ISAACSIM_REPLICATOR_OUTPUT_DIR",
        DEFAULT_REPLICATOR_OUTPUT_DIR,
    ).strip()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Robot prim is not valid: {robot_prim_path}")

    apply_capture_semantics(stage, robot_prim_path)

    camera_path = add_robot_owned_camera(stage, robot_prim_path)
    render_product = rep.create.render_product(camera_path, (1024, 1024))

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=output_dir,
        rgb=True,
        instance_segmentation=True,
        distance_to_camera=True,
    )
    writer.attach([render_product])
    _replicator_capture_enabled = True

    print(
        "[store_shelf] Configured Replicator writer "
        f"(rgb, instance_segmentation, distance_to_camera) at {output_dir}",
        flush=True,
    )
    print(
        f"[store_shelf] Collect vision camera owned by robot at {camera_path}",
        flush=True,
    )
    return output_dir


def _matrix_to_rows(matrix) -> list[list[float]]:
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _prim_transform_metadata(stage, prim_path: str) -> dict:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return {"path": prim_path, "valid": False}

    local_transform = UsdGeom.Xformable(prim).GetLocalTransformation()
    world_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    return {
        "path": prim_path,
        "valid": True,
        "local_to_parent": _matrix_to_rows(local_transform),
        "local_to_world": _matrix_to_rows(world_transform),
    }


def _write_capture_metadata(flow_state: dict, cart_x_offset: float) -> None:
    stage = flow_state["stage"]
    frame_id = f"{flow_state['capture_frame_index']:04d}"
    metadata_path = Path(flow_state["output_dir"]) / f"metadata_{frame_id}.json"
    robot_prim_path = flow_state["robot_prim_path"]
    camera_link_prim = find_prim_named(stage, VISION_CAMERA_LINK_PRIM_NAME)
    camera_prim = find_prim_named(stage, VISION_CAMERA_OPTICAL_FRAME_PRIM_NAME)
    camera_link_path = (
        camera_link_prim.GetPath().pathString if camera_link_prim is not None else ""
    )
    camera_path = camera_prim.GetPath().pathString if camera_prim is not None else ""
    payload = {
        "run_id": flow_state["run_id"],
        "run_index": flow_state["run_index"],
        "capture_index_in_run": flow_state["capture_index_in_run"],
        "capture_frame_index": flow_state["capture_frame_index"],
        "cart_x_offset": cart_x_offset,
        "captures_per_run": flow_state["captures_per_run"],
        "arm_joint_positions": flow_state.get("arm_joint_positions", {}),
        "transforms": {
            "cart": _prim_transform_metadata(
                stage,
                flow_state["cart_prim"].GetPath().pathString,
            ),
            "robot": _prim_transform_metadata(stage, robot_prim_path),
            "camera_link": _prim_transform_metadata(stage, camera_link_path),
            "camera_optical_frame": _prim_transform_metadata(stage, camera_path),
        },
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _start_collection_run(flow_state: dict) -> None:
    flow_state["run_id"] = uuid.uuid4().hex
    flow_state["capture_index_in_run"] = 0
    flow_state["run_index"] += 1
    _randomize_items(flow_state["item_position_pools"])
    _set_cart_x_offset(
        flow_state["cart_prim"],
        flow_state["cart_base_translation"],
        flow_state["cart_left_offset"],
    )
    capture_offsets = [
        random.uniform(flow_state["cart_left_offset"], flow_state["cart_right_offset"])
        for _ in range(flow_state["captures_per_run"])
    ]
    capture_offsets.sort()
    flow_state["capture_offsets"] = capture_offsets
    print(
        "[store_shelf] Started collection run "
        f"{flow_state['run_index']} id={flow_state['run_id']} "
        f"captures={flow_state['captures_per_run']} "
        f"cart_offsets={[round(offset, 3) for offset in capture_offsets]}",
        flush=True,
    )
    _prepare_collection_capture_pose(flow_state)


def _prepare_collection_capture_pose(flow_state: dict) -> None:
    if flow_state["capture_index_in_run"] >= flow_state["captures_per_run"]:
        return

    cart_x_offset = flow_state["capture_offsets"][flow_state["capture_index_in_run"]]
    flow_state["arm_joint_positions"] = randomize_robot_arm_joints(
        flow_state["stage"],
        flow_state["robot_prim_path"],
    )
    flow_state["cart_x_offset"] = cart_x_offset
    _set_cart_x_offset(
        flow_state["cart_prim"],
        flow_state["cart_base_translation"],
        cart_x_offset,
    )
    flow_state["time_until_capture"] = flow_state["capture_interval_seconds"]


def _reset_store_demo_flow(flow_state: dict) -> None:
    _publish_motion_reset(str(flow_state.get("reset_reason", "store_demo_reset")))
    _randomize_items(flow_state["item_position_pools"])
    flow_state["arm_joint_positions"] = randomize_store_demo_robot_arm_joints(
        flow_state["stage"],
        flow_state["robot_prim_path"],
    )
    flow_state["cart_x_offset"] = _random_cart_x_offset(flow_state)
    _set_cart_x_offset(
        flow_state["cart_prim"],
        flow_state["cart_base_translation"],
        flow_state["cart_x_offset"],
    )
    print(
        "[store_shelf] Reset store demo flow "
        f"cart_offset={flow_state['cart_x_offset']:.3f}m "
        f"arm_joints={len(flow_state['arm_joint_positions'])}",
        flush=True,
    )
    flow_state["controller_idle_started_monotonic"] = None
    _schedule_next_store_demo_reset(flow_state)


def _update_collection_flow(flow_state: dict) -> None:
    global _replicator_capture_enabled

    if not _replicator_capture_enabled:
        return

    if flow_state["warmup_seconds_remaining"] > 0.0:
        flow_state["warmup_seconds_remaining"] -= SIMULATION_STEP_SECONDS
        if flow_state["warmup_seconds_remaining"] > 0.0:
            return
        _start_collection_run(flow_state)
        return

    flow_state["time_until_capture"] -= SIMULATION_STEP_SECONDS
    if flow_state["time_until_capture"] > 0.0:
        return

    if flow_state["capture_index_in_run"] >= flow_state["captures_per_run"]:
        _start_collection_run(flow_state)
        return

    cart_x_offset = flow_state["cart_x_offset"]
    _write_capture_metadata(flow_state, cart_x_offset)

    import omni.replicator.core as rep

    rep.orchestrator.step(rt_subframes=1)
    print(
        "[store_shelf] Captured training image "
        f"frame={flow_state['capture_frame_index']:04d} "
        f"run_id={flow_state['run_id']} "
        f"capture={flow_state['capture_index_in_run'] + 1}/"
        f"{flow_state['captures_per_run']} cart_x_offset={cart_x_offset:.3f}m",
        flush=True,
    )
    flow_state["capture_frame_index"] += 1
    flow_state["capture_index_in_run"] += 1
    _prepare_collection_capture_pose(flow_state)


def _update_store_demo_flow(flow_state: dict) -> None:
    global _trajectory_executors

    if _timeline_autoplay_enabled and not _timeline_is_playing():
        return

    if flow_state["cart_motion_enabled"]:
        x_offset = min(
            flow_state["cart_x_offset"]
            + flow_state["cart_slide_speed"] * SIMULATION_STEP_SECONDS,
            flow_state["cart_right_offset"],
        )
        flow_state["cart_x_offset"] = x_offset
        _set_cart_x_offset(
            flow_state["cart_prim"],
            flow_state["cart_base_translation"],
            x_offset,
        )

    if _controller_is_idle():
        if flow_state["controller_idle_started_monotonic"] is None:
            flow_state["controller_idle_started_monotonic"] = time.monotonic()
    else:
        flow_state["controller_idle_started_monotonic"] = None

    idle_started_monotonic = flow_state["controller_idle_started_monotonic"]
    if (
        idle_started_monotonic is not None
        and time.monotonic() - idle_started_monotonic >= flow_state["idle_reset_seconds"]
    ):
        print(
            "[store_shelf] Store demo idle reset firing "
            f"idle_elapsed={time.monotonic() - idle_started_monotonic:.3f}s",
            flush=True,
        )
        flow_state["reset_reason"] = "idle"
        _reset_store_demo_flow(flow_state)
        return

    if time.monotonic() < flow_state["next_reset_monotonic"]:
        return
    if any(executor.has_active_trajectory() for executor in _trajectory_executors):
        return
    print("[store_shelf] Store demo timed reset firing", flush=True)
    flow_state["reset_reason"] = "timer"
    _reset_store_demo_flow(flow_state)


def update_simulation_app(simulation_app) -> None:
    simulation_app.update()
    if _vision_panel is not None:
        _vision_panel.update()
    if _ground_truth_items_node is not None:
        rclpy.spin_once(_ground_truth_items_node, timeout_sec=0.0)
    for executor in _trajectory_executors:
        executor.update(SIMULATION_STEP_SECONDS)
    if _flow_state is None:
        return
    _publish_ground_truth_items(_flow_state)
    if _flow_state["mode"] == "collect":
        _update_collection_flow(_flow_state)
    elif _flow_state["mode"] == "store_demo":
        _update_store_demo_flow(_flow_state)


def _setup_store_shelf_scene(
    simulation_app,
    configuration: str,
    capture_enabled: bool,
) -> str:
    global _replicator_capture_enabled
    global _timeline_autoplay_enabled
    global _vision_panel
    global _ground_truth_items_node
    global _ground_truth_items_publisher
    global _motion_reset_publisher
    global _coordinator_state_subscription
    global _latest_coordinator_state
    global _ground_truth_items_publish_elapsed
    global _trajectory_executors
    global _flow_state

    _replicator_capture_enabled = False
    _timeline_autoplay_enabled = False
    _flow_state = None
    if _vision_panel is not None:
        _vision_panel.close()
        _vision_panel = None
    if _ground_truth_items_node is not None:
        _ground_truth_items_node.destroy_node()
        _ground_truth_items_node = None
        _ground_truth_items_publisher = None
        _motion_reset_publisher = None
        _coordinator_state_subscription = None
        _latest_coordinator_state = None
        _ground_truth_items_publish_elapsed = 0.0
    for executor in _trajectory_executors:
        executor.close()
    _trajectory_executors = []

    scene = construct_scene(configuration)
    _timeline_autoplay_enabled = (
        scene["ros2_joint_bridge"] is not None or configuration == "collect_vision_data"
    )
    output_dir = None
    if capture_enabled:
        output_dir = _setup_replicator_capture(scene["stage"], scene["robot_prim_path"])
    item_position_pools = build_item_position_pools(scene["stage"])
    cart_left_offset = float(
        os.environ.get(
            "ISAACSIM_CART_LEFT_OFFSET",
            str(DEFAULT_CART_LEFT_OFFSET_METERS),
        ).strip()
    )
    cart_right_offset = float(
        os.environ.get(
            "ISAACSIM_CART_RIGHT_OFFSET",
            str(DEFAULT_CART_RIGHT_OFFSET_METERS),
        ).strip()
    )
    captures_per_run = int(
        os.environ.get(
            "ISAACSIM_CAPTURES_PER_RUN",
            str(DEFAULT_CAPTURES_PER_RUN),
        ).strip()
    )
    capture_interval_seconds = float(
        os.environ.get(
            "ISAACSIM_CAPTURE_INTERVAL_SECONDS",
            str(DEFAULT_CAPTURE_INTERVAL_SECONDS),
        ).strip()
    )
    capture_warmup_seconds = float(
        os.environ.get(
            "ISAACSIM_CAPTURE_WARMUP_SECONDS",
            str(DEFAULT_CAPTURE_WARMUP_SECONDS),
        ).strip()
    )
    cart_slide_speed = float(
        os.environ.get(
            "ISAACSIM_CART_SLIDE_SPEED",
            str(DEFAULT_CART_SLIDE_SPEED_METERS_PER_SECOND),
        ).strip()
    )
    store_demo_cart_motion_enabled = (
        os.environ.get(
            "ISAACSIM_STORE_DEMO_CART_MOTION_ENABLED",
            str(DEFAULT_STORE_DEMO_CART_MOTION_ENABLED).lower(),
        ).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    store_demo_reset_interval_seconds = float(
        os.environ.get(
            "ISAACSIM_STORE_DEMO_RESET_INTERVAL_SECONDS",
            str(DEFAULT_STORE_DEMO_RESET_INTERVAL_SECONDS),
        ).strip()
    )
    store_demo_idle_reset_seconds = float(
        os.environ.get(
            "ISAACSIM_STORE_DEMO_IDLE_RESET_SECONDS",
            str(DEFAULT_STORE_DEMO_IDLE_RESET_SECONDS),
        ).strip()
    )

    _set_cart_x_offset(scene["cart_prim"], scene["cart_base_translation"], cart_left_offset)
    if configuration == "collect_vision_data":
        _flow_state = {
            "mode": "collect",
            "stage": scene["stage"],
            "output_dir": output_dir,
            "robot_prim_path": scene["robot_prim_path"],
            "cart_prim": scene["cart_prim"],
            "cart_base_translation": scene["cart_base_translation"],
            "item_position_pools": item_position_pools,
            "cart_left_offset": cart_left_offset,
            "cart_right_offset": cart_right_offset,
            "captures_per_run": captures_per_run,
            "capture_interval_seconds": capture_interval_seconds,
            "capture_frame_index": 0,
            "capture_index_in_run": 0,
            "run_index": 0,
            "run_id": "",
            "capture_offsets": [],
            "time_until_capture": 0.0,
            "warmup_seconds_remaining": capture_warmup_seconds,
        }
        print(
            "[store_shelf] Collection warmup before first capture "
            f"{capture_warmup_seconds:.3f}s simulation time",
            flush=True,
        )
    elif configuration == "store_demo":
        _flow_state = {
            "mode": "store_demo",
            "stage": scene["stage"],
            "robot_prim_path": scene["robot_prim_path"],
            "cart_prim": scene["cart_prim"],
            "cart_base_translation": scene["cart_base_translation"],
            "item_position_pools": item_position_pools,
            "cart_x_offset": cart_left_offset,
            "cart_left_offset": cart_left_offset,
            "cart_right_offset": cart_right_offset,
            "cart_slide_speed": cart_slide_speed,
            "cart_motion_enabled": store_demo_cart_motion_enabled,
            "reset_interval_seconds": store_demo_reset_interval_seconds,
            "idle_reset_seconds": store_demo_idle_reset_seconds,
            "controller_idle_started_monotonic": None,
            "next_reset_monotonic": time.monotonic() + store_demo_reset_interval_seconds,
            "reset_reason": "startup",
        }
        _reset_store_demo_flow(_flow_state)
        print(
            "[store_shelf] Store demo timed reset enabled "
            f"cart_offset_range=({cart_left_offset:.3f}, {cart_right_offset:.3f})m "
            f"cart_motion_enabled={store_demo_cart_motion_enabled} "
            f"cart_slide_speed={cart_slide_speed:.3f}m/s "
            f"reset_interval={store_demo_reset_interval_seconds:.3f}s "
            f"idle_reset={store_demo_idle_reset_seconds:.3f}s",
            flush=True,
        )
    else:
        print("[store_shelf] Static mode: scene flow disabled", flush=True)

    print(
        f"[store_shelf] {configuration} referenced robot {scene['robot_path']} at "
        f"{scene['robot_prim_path']}",
        flush=True,
    )
    if scene["ros2_joint_bridge"] is not None:
        print(
            "[store_shelf] ROS 2 joint bridge topics: "
            f"joint_states={scene['ros2_joint_bridge']['joint_state_topic']}, "
            f"joint_command={scene['ros2_joint_bridge']['joint_command_topic']}, "
            f"articulation_root={scene['ros2_joint_bridge']['articulation_root_path']}",
            flush=True,
        )
    if scene["ros2_camera_bridge"] is not None:
        print(
            "[store_shelf] ROS 2 camera bridge topics: "
            f"image={scene['ros2_camera_bridge']['camera_topic']}, "
            f"camera_info={scene['ros2_camera_bridge']['camera_info_topic']}, "
            f"frame_id={scene['ros2_camera_bridge']['camera_frame_id']}",
            flush=True,
        )
    if configuration == "store_demo":
        if not rclpy.ok():
            rclpy.init(args=None)
        ground_truth_items_topic = os.environ.get(
            "ISAACSIM_GROUND_TRUTH_ITEMS_TOPIC",
            DEFAULT_GROUND_TRUTH_ITEMS_TOPIC,
        ).strip()
        _ground_truth_items_node = rclpy.create_node("isaacsim_ground_truth_items")
        _ground_truth_items_publisher = _ground_truth_items_node.create_publisher(
            String,
            ground_truth_items_topic,
            10,
        )
        motion_reset_topic = os.environ.get(
            "ISAACSIM_MOTION_RESET_TOPIC",
            DEFAULT_MOTION_RESET_TOPIC,
        ).strip()
        _motion_reset_publisher = _ground_truth_items_node.create_publisher(
            String,
            motion_reset_topic,
            10,
        )
        coordinator_state_topic = os.environ.get(
            "ISAACSIM_COORDINATOR_STATE_TOPIC",
            DEFAULT_COORDINATOR_STATE_TOPIC,
        ).strip()
        _coordinator_state_subscription = _ground_truth_items_node.create_subscription(
            String,
            coordinator_state_topic,
            _handle_coordinator_state,
            10,
        )
        print(
            "[store_shelf] Publishing ground-truth item centers "
            f"topic={ground_truth_items_topic}; "
            f"monitoring coordinator state topic={coordinator_state_topic}; "
            f"publishing motion reset topic={motion_reset_topic}",
            flush=True,
        )
        from .trajectory_executor import IsaacSimTrajectoryExecutor

        from .trajectory_executor import LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS

        _trajectory_executors = [
            IsaacSimTrajectoryExecutor(
                articulation_root_path=scene["ros2_joint_bridge"]["articulation_root_path"],
                action_name="left_arm_controller/follow_joint_trajectory",
                joint_names=LEFT_ARM_JOINTS,
            ),
            IsaacSimTrajectoryExecutor(
                articulation_root_path=scene["ros2_joint_bridge"]["articulation_root_path"],
                action_name="right_arm_controller/follow_joint_trajectory",
                joint_names=RIGHT_ARM_JOINTS,
            ),
        ]
    if _timeline_autoplay_enabled:
        if _play_timeline_if_needed():
            print(
                "[store_shelf] Started Isaac Sim timeline for store_demo",
                flush=True,
            )
        print(
            "[store_shelf] Isaac Sim timeline autoplay enabled for store_demo",
            flush=True,
        )
    if (
        configuration == "store_demo"
        and os.environ.get("ISAACSIM_RUNNER_MODE", "headed").strip().lower()
        != "headless"
        and os.environ.get("ISAACSIM_VISION_PANEL_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"}
    ):
        debug_image_topic = os.environ.get(
            "ISAACSIM_VISION_DEBUG_IMAGE_TOPIC",
            DEFAULT_DEBUG_IMAGE_TOPIC,
        ).strip()
        if not rclpy.ok():
            rclpy.init(args=None)
        _vision_panel = IsaacSimVisionPanel(topic_name=debug_image_topic)
        print(
            f"[store_shelf] Opened Isaac Sim vision panel for {debug_image_topic}",
            flush=True,
        )
    update_simulation_app(simulation_app)
    return (
        f"Loaded {configuration} scene {scene['scene_path']} with robot at "
        f"{scene['robot_prim_path']}; item_pools={len(item_position_pools)}; "
        f"replicator_output={output_dir or 'disabled'}"
    )


def collect_vision_data(simulation_app) -> str:
    return _setup_store_shelf_scene(
        simulation_app,
        configuration="collect_vision_data",
        capture_enabled=True,
    )


def static(simulation_app) -> str:
    return _setup_store_shelf_scene(
        simulation_app,
        configuration="static",
        capture_enabled=True,
    )


def store_demo(simulation_app) -> str:
    return _setup_store_shelf_scene(
        simulation_app,
        configuration="store_demo",
        capture_enabled=False,
    )
