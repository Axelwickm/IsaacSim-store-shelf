import os
import random
from pathlib import Path


DEFAULT_COLLECT_VISION_SCENE = "/workspace/usd/store_scene.usdc"
DEFAULT_ROBOT_PATH = "/workspace/usd/robot/yumi.usd"
ROBOT_PRIM_NAME = "YuMi"
ITEMS_PRIM_NAME = "all_items"
CART_PRIM_NAME = "Cart"
SHELF_PRIM_NAME = "Shelf"
VISION_CAMERA_PRIM_NAME = "VisionCamera"
ITEM_MASS_KG = 0.2


def find_prim_named(stage, name: str):
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def prim_local_translation(prim) -> tuple[float, float, float]:
    from pxr import UsdGeom

    local_transform = UsdGeom.Xformable(prim).GetLocalTransformation()
    translation = local_transform.ExtractTranslation()
    return float(translation[0]), float(translation[1]), float(translation[2])


def set_prim_local_translation(prim, translation: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(prim)
    translate_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
            break
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(*translation))


def active_children(prim) -> list:
    return [child for child in prim.GetChildren() if child.IsActive()]


def open_scene(scene_path: str):
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


def add_robot_at_cart_origin(stage, robot_path: str) -> str:
    from pxr import UsdGeom

    robot_file = Path(robot_path)
    if not robot_file.exists():
        raise FileNotFoundError(f"Robot USD does not exist: {robot_file}")

    cart_prim = find_prim_named(stage, CART_PRIM_NAME)
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


def add_robot_owned_camera(stage, robot_prim_path: str) -> str:
    from pxr import Gf, UsdGeom

    camera_path = f"{robot_prim_path}/{VISION_CAMERA_PRIM_NAME}"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr().Set(14.0)
    camera.CreateFocusDistanceAttr().Set(1.0)

    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    camera_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.8))
    camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(70.0, 0.0, -90.0))
    return camera_path


def configure_scene_physics(stage) -> None:
    from pxr import PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    physics_scene = None
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            physics_scene = prim
            break
    if physics_scene is None:
        physics_scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene")).GetPrim()
        physics_api = UsdPhysics.Scene(physics_scene)
        physics_api.CreateGravityDirectionAttr().Set((0.0, 0.0, -1.0))
        physics_api.CreateGravityMagnitudeAttr().Set(9.81)

    items_prim = find_prim_named(stage, ITEMS_PRIM_NAME)
    if items_prim is not None:
        for group_prim in active_children(items_prim):
            if not group_prim.GetName().startswith("item_group_"):
                continue
            for item_prim in active_children(group_prim):
                UsdPhysics.RigidBodyAPI.Apply(item_prim)
                UsdPhysics.MassAPI.Apply(item_prim).CreateMassAttr().Set(ITEM_MASS_KG)
                for prim in Usd.PrimRange(item_prim):
                    if prim.IsA(UsdGeom.Mesh):
                        UsdPhysics.CollisionAPI.Apply(prim)
                        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
                            "convexHull"
                        )
                        PhysxSchema.PhysxCollisionAPI.Apply(prim)

    shelf_prim = find_prim_named(stage, SHELF_PRIM_NAME)
    if shelf_prim is not None:
        for prim in Usd.PrimRange(shelf_prim):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim)
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
                    "none"
                )
                PhysxSchema.PhysxCollisionAPI.Apply(prim)

    print("[store_shelf] Configured item and shelf physics", flush=True)


def build_item_position_pools(stage) -> list[list]:
    items_prim = find_prim_named(stage, ITEMS_PRIM_NAME)
    if items_prim is None:
        print(
            f"[store_shelf] No {ITEMS_PRIM_NAME!r} prim found; "
            "skipping item position randomization",
            flush=True,
        )
        return []

    pools = []
    for group_prim in active_children(items_prim):
        if not group_prim.GetName().startswith("item_group_"):
            continue
        items = active_children(group_prim)
        if len(items) < 2:
            print(
                f"[store_shelf] Item group {group_prim.GetPath()} has "
                f"{len(items)} active child; skipping",
                flush=True,
            )
            continue
        pools.append(items)

    print(
        f"[store_shelf] Discovered {len(pools)} item position pools under "
        f"{items_prim.GetPath()}",
        flush=True,
    )
    for pool in pools:
        print(
            "[store_shelf] Item pool: "
            + ", ".join(item.GetPath().pathString for item in pool),
            flush=True,
        )
    return pools


def randomize_item_position_pools(pools: list[list]) -> list[str]:
    shuffled_groups = []
    for pool in pools:
        translations = [prim_local_translation(item) for item in pool]
        random.shuffle(translations)
        for item, translation in zip(pool, translations):
            set_prim_local_translation(item, translation)
        shuffled_groups.append(pool[0].GetParent().GetPath().pathString)
    return shuffled_groups


def randomize_cart_x(cart_prim, base_translation: tuple[float, float, float]) -> float:
    x_offset = random.uniform(0.0, 2.0)
    set_prim_local_translation(
        cart_prim,
        (base_translation[0] + x_offset, base_translation[1], base_translation[2]),
    )
    return x_offset


def set_semantic_label(prim, label: str) -> None:
    try:
        from pxr import Semantics
    except ImportError:
        print(
            "[store_shelf] USD Semantics schema is unavailable; "
            "instance labels may be unnamed",
            flush=True,
        )
        return

    semantic_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    semantic_api.CreateSemanticTypeAttr().Set("class")
    semantic_api.CreateSemanticDataAttr().Set(label)


def apply_capture_semantics(stage, robot_prim_path: str) -> None:
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if robot_prim.IsValid():
        set_semantic_label(robot_prim, "robot")

    cart_prim = find_prim_named(stage, CART_PRIM_NAME)
    if cart_prim is not None:
        set_semantic_label(cart_prim, "cart")

    items_prim = find_prim_named(stage, ITEMS_PRIM_NAME)
    if items_prim is None:
        return

    for group_prim in active_children(items_prim):
        group_label = group_prim.GetName()
        for item_prim in active_children(group_prim):
            set_semantic_label(item_prim, group_label)


def construct_scene(configuration: str) -> dict:
    scene_path = os.environ.get(
        "ISAACSIM_COLLECT_VISION_SCENE",
        DEFAULT_COLLECT_VISION_SCENE,
    ).strip()
    robot_path = os.environ.get("ISAACSIM_ROBOT_PATH", DEFAULT_ROBOT_PATH).strip()

    print(f"[store_shelf] {configuration} opening scene: {scene_path}", flush=True)
    stage = open_scene(scene_path)
    configure_scene_physics(stage)

    cart_prim = find_prim_named(stage, CART_PRIM_NAME)
    if cart_prim is None:
        raise RuntimeError("Could not find Cart prim in loaded scene")

    robot_prim_path = add_robot_at_cart_origin(stage, robot_path)
    cart_base_translation = prim_local_translation(cart_prim)

    return {
        "stage": stage,
        "scene_path": scene_path,
        "robot_path": robot_path,
        "robot_prim_path": robot_prim_path,
        "cart_prim": cart_prim,
        "cart_base_translation": cart_base_translation,
    }
