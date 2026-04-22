import os
from pathlib import Path


DEFAULT_COLLECT_VISION_SCENE = "/workspace/usd/store_scene.usdc"
DEFAULT_ROBOT_PATH = "/workspace/usd/robot/yumi.usd"
ROBOT_PRIM_NAME = "YuMi"


def _find_prim_named(stage, name: str):
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def _add_robot_at_cart_origin(stage, robot_path: str) -> str:
    from pxr import UsdGeom

    robot_file = Path(robot_path)
    if not robot_file.exists():
        raise FileNotFoundError(f"Robot USD does not exist: {robot_file}")

    cart_prim = _find_prim_named(stage, "Cart")
    if cart_prim is None:
        raise RuntimeError("Could not find Cart prim in loaded scene")

    robot_prim_path = cart_prim.GetPath().AppendChild(ROBOT_PRIM_NAME)
    robot_prim = stage.DefinePrim(robot_prim_path, "Xform")
    robot_prim.GetReferences().ClearReferences()
    robot_prim.GetReferences().AddReference(str(robot_file))

    robot_xform = UsdGeom.Xformable(robot_prim)
    robot_xform.ClearXformOpOrder()
    robot_xform.AddRotateZOp().Set(180.0)
    robot_xform.AddTranslateOp().Set((-0.40, 0.0, 0.0))
    return robot_prim_path.pathString


def _open_scene(scene_path: str):
    import omni.usd

    scene_file = Path(scene_path)
    if not scene_file.exists():
        raise FileNotFoundError(f"Scene USD does not exist: {scene_file}")

    if not omni.usd.get_context().open_stage(str(scene_file)):
        raise RuntimeError(f"Failed to open Isaac Sim scene: {scene_file}")

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Scene opened but no USD stage is available")
    return stage


def collect_vision_data(simulation_app) -> str:
    scene_path = os.environ.get(
        "ISAACSIM_COLLECT_VISION_SCENE",
        DEFAULT_COLLECT_VISION_SCENE,
    ).strip()
    robot_path = os.environ.get("ISAACSIM_ROBOT_PATH", DEFAULT_ROBOT_PATH).strip()

    print(f"[store_shelf] collect_vision_data opening scene: {scene_path}", flush=True)
    stage = _open_scene(scene_path)
    robot_prim_path = _add_robot_at_cart_origin(stage, robot_path)
    print(
        f"[store_shelf] collect_vision_data referenced robot {robot_path} "
        f"at {robot_prim_path}",
        flush=True,
    )
    simulation_app.update()
    return f"Loaded collect_vision_data scene {scene_path} with robot at {robot_prim_path}"
