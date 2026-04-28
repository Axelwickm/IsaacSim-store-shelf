import json
import os
import random
import uuid
from pathlib import Path

import rclpy

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
    set_robot_arm_joints_for_planning,
    set_prim_local_translation,
)
from .vision_panel import DEFAULT_DEBUG_IMAGE_TOPIC, IsaacSimVisionPanel
from .target_marker_visualizer import (
    DEFAULT_CLEAR_LATCHED_TARGET_TOPIC,
    DEFAULT_DEBUG_DRAW_CROSSHAIR_SIZE_METERS,
    DEFAULT_DEBUG_DRAW_RADIUS_PIXELS,
    DEFAULT_LATCHED_TARGET_POINT_TOPIC,
    DEFAULT_MARKER_RADIUS_METERS,
    IsaacSimTargetMarkerVisualizer,
)


DEFAULT_REPLICATOR_OUTPUT_DIR = "/workspace/collect_vision_data_output"
DEFAULT_CAPTURES_PER_RUN = 10
DEFAULT_CAPTURE_INTERVAL_SECONDS = 0.35
DEFAULT_CAPTURE_WARMUP_SECONDS = 1.0
DEFAULT_CART_LEFT_OFFSET_METERS = 0.0
DEFAULT_CART_RIGHT_OFFSET_METERS = 2.0
DEFAULT_CART_SLIDE_SPEED_METERS_PER_SECOND = 0.25
DEFAULT_STORE_DEMO_CART_MOTION_ENABLED = False
DEFAULT_STORE_DEMO_RESET_INTERVAL_SECONDS = 60.0
SIMULATION_STEP_SECONDS = 1.0 / 60.0

_replicator_capture_enabled = False
_timeline_autoplay_enabled = False
_vision_panel = None
_target_marker_visualizer = None
_trajectory_executor = None
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
    _randomize_items(flow_state["item_position_pools"])
    set_robot_arm_joints_for_planning(flow_state["stage"], flow_state["robot_prim_path"])
    flow_state["cart_x_offset"] = flow_state["cart_left_offset"]
    _set_cart_x_offset(
        flow_state["cart_prim"],
        flow_state["cart_base_translation"],
        flow_state["cart_x_offset"],
    )


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
    global _trajectory_executor

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

    flow_state["reset_time_remaining"] -= SIMULATION_STEP_SECONDS
    if flow_state["reset_time_remaining"] > 0.0:
        return
    if (
        _trajectory_executor is not None
        and _trajectory_executor.has_active_trajectory()
    ):
        flow_state["reset_time_remaining"] = 1.0
        return
    print("[store_shelf] Store demo timed reset firing", flush=True)
    _reset_store_demo_flow(flow_state)
    flow_state["reset_time_remaining"] = flow_state["reset_interval_seconds"]


def update_simulation_app(simulation_app) -> None:
    simulation_app.update()
    if _vision_panel is not None:
        _vision_panel.update()
    if _target_marker_visualizer is not None:
        _target_marker_visualizer.update()
    if _trajectory_executor is not None:
        _trajectory_executor.update(SIMULATION_STEP_SECONDS)
    if _flow_state is None:
        return
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
    global _target_marker_visualizer
    global _trajectory_executor
    global _flow_state

    _replicator_capture_enabled = False
    _timeline_autoplay_enabled = False
    _flow_state = None
    if _vision_panel is not None:
        _vision_panel.close()
        _vision_panel = None
    if _target_marker_visualizer is not None:
        _target_marker_visualizer.close()
        _target_marker_visualizer = None
    if _trajectory_executor is not None:
        _trajectory_executor.close()
        _trajectory_executor = None

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
            "reset_time_remaining": store_demo_reset_interval_seconds,
        }
        _reset_store_demo_flow(_flow_state)
        print(
            "[store_shelf] Store demo timed reset enabled "
            f"cart_offset={cart_left_offset:.3f}m "
            f"cart_motion_enabled={store_demo_cart_motion_enabled} "
            f"cart_right_offset={cart_right_offset:.3f}m "
            f"cart_slide_speed={cart_slide_speed:.3f}m/s "
            f"reset_interval={store_demo_reset_interval_seconds:.3f}s",
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
        from .trajectory_executor import IsaacSimTrajectoryExecutor

        _trajectory_executor = IsaacSimTrajectoryExecutor(
            articulation_root_path=scene["ros2_joint_bridge"]["articulation_root_path"],
        )
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
    if (
        configuration == "store_demo"
        and os.environ.get("ISAACSIM_TARGET_MARKERS_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"}
    ):
        target_topic = os.environ.get(
            "ISAACSIM_LATCHED_TARGET_POINT_TOPIC",
            DEFAULT_LATCHED_TARGET_POINT_TOPIC,
        ).strip()
        clear_topic = os.environ.get(
            "ISAACSIM_CLEAR_LATCHED_TARGET_TOPIC",
            DEFAULT_CLEAR_LATCHED_TARGET_TOPIC,
        ).strip()
        marker_radius = float(
            os.environ.get(
                "ISAACSIM_TARGET_MARKER_RADIUS",
                str(DEFAULT_MARKER_RADIUS_METERS),
            ).strip()
        )
        debug_draw_radius = float(
            os.environ.get(
                "ISAACSIM_TARGET_MARKER_DEBUG_DRAW_RADIUS",
                str(DEFAULT_DEBUG_DRAW_RADIUS_PIXELS),
            ).strip()
        )
        debug_draw_crosshair_size = float(
            os.environ.get(
                "ISAACSIM_TARGET_MARKER_CROSSHAIR_SIZE",
                str(DEFAULT_DEBUG_DRAW_CROSSHAIR_SIZE_METERS),
            ).strip()
        )
        max_markers = int(
            os.environ.get("ISAACSIM_MAX_TARGET_MARKERS", "20").strip()
        )
        if not rclpy.ok():
            rclpy.init(args=None)
        _target_marker_visualizer = IsaacSimTargetMarkerVisualizer(
            target_topic_name=target_topic,
            clear_topic_name=clear_topic,
            marker_radius_meters=marker_radius,
            debug_draw_radius_pixels=debug_draw_radius,
            debug_draw_crosshair_size_meters=debug_draw_crosshair_size,
            max_markers=max_markers,
        )
        print(
            "[store_shelf] Enabled Isaac Sim latched target markers "
            f"target_topic={target_topic} clear_topic={clear_topic} "
            f"radius={marker_radius:.3f}m debug_draw_radius={debug_draw_radius:.1f}px "
            f"crosshair={debug_draw_crosshair_size:.3f}m max_markers={max_markers}",
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
