import os
import random
from pathlib import Path

from .scene_construction import (
    DEFAULT_COLLECT_VISION_SCENE,
    add_robot_owned_camera,
    apply_capture_semantics,
    build_item_position_pools,
    construct_scene,
    randomize_cart_x,
    randomize_item_position_pools,
)


DEFAULT_REPLICATOR_OUTPUT_DIR = "/workspace/collect_vision_data_output"
MAX_CAPTURE_SETTLE_TIME_SECONDS = 1.5
SIMULATION_STEP_SECONDS = 1.0 / 60.0

_prepare_next_capture = None
_replicator_capture_enabled = False
_capture_settle_time_remaining = 0.0


def _randomize_scene(item_pools: list[list], cart_prim, cart_base_translation) -> None:
    shuffled_groups = randomize_item_position_pools(item_pools)
    cart_x_offset = None
    if cart_prim is not None:
        cart_x_offset = randomize_cart_x(cart_prim, cart_base_translation)

    details = []
    if shuffled_groups:
        details.append("item_groups=" + ", ".join(shuffled_groups))
    if cart_x_offset is not None:
        details.append(f"cart_x_offset={cart_x_offset:.3f}m")
    print("[store_shelf] Randomized scene: " + "; ".join(details), flush=True)


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


def update_simulation_app(simulation_app) -> None:
    global _capture_settle_time_remaining

    simulation_app.update()
    if not _replicator_capture_enabled:
        return

    import omni.replicator.core as rep

    _capture_settle_time_remaining -= SIMULATION_STEP_SECONDS
    if _capture_settle_time_remaining > 0.0:
        return

    rep.orchestrator.step(rt_subframes=1)
    if _prepare_next_capture is not None:
        _prepare_next_capture()
    _capture_settle_time_remaining = random.uniform(0.0, MAX_CAPTURE_SETTLE_TIME_SECONDS)


def _setup_store_shelf_scene(simulation_app, configuration: str, randomize: bool) -> str:
    global _capture_settle_time_remaining, _prepare_next_capture, _replicator_capture_enabled

    _prepare_next_capture = None
    _replicator_capture_enabled = False
    _capture_settle_time_remaining = 0.0

    scene = construct_scene(configuration)
    output_dir = _setup_replicator_capture(scene["stage"], scene["robot_prim_path"])
    item_position_pools = []
    cart_x_offset = None

    if randomize:
        item_position_pools = build_item_position_pools(scene["stage"])
        _randomize_scene(
            item_position_pools,
            scene["cart_prim"],
            scene["cart_base_translation"],
        )
        _prepare_next_capture = lambda: _randomize_scene(
            item_position_pools,
            scene["cart_prim"],
            scene["cart_base_translation"],
        )
        _capture_settle_time_remaining = random.uniform(
            0.0, MAX_CAPTURE_SETTLE_TIME_SECONDS
        )
        cart_x_offset = None
        print("[store_shelf] Registered capture-cycle scene randomizer", flush=True)
    else:
        print("[store_shelf] Static mode: randomization disabled", flush=True)

    print(
        f"[store_shelf] {configuration} referenced robot {scene['robot_path']} at "
        f"{scene['robot_prim_path']}",
        flush=True,
    )
    if cart_x_offset is not None:
        print(f"[store_shelf] Initial cart x offset: {cart_x_offset:.3f}m", flush=True)

    update_simulation_app(simulation_app)
    return (
        f"Loaded {configuration} scene {scene['scene_path']} with robot at "
        f"{scene['robot_prim_path']}; item_pools={len(item_position_pools)}; "
        f"replicator_output={output_dir}"
    )


def collect_vision_data(simulation_app) -> str:
    return _setup_store_shelf_scene(
        simulation_app,
        configuration="collect_vision_data",
        randomize=True,
    )


def static(simulation_app) -> str:
    return _setup_store_shelf_scene(
        simulation_app,
        configuration="static",
        randomize=False,
    )
