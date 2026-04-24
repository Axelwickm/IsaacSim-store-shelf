import os
import random
import shutil
from pathlib import Path


DEFAULT_COLLECT_VISION_SCENE = "/workspace/usd/store_scene.usdc"
DEFAULT_ROBOT_PATH = "/workspace/usd/robot/yumi_isaacsim.urdf"
ROBOT_PRIM_NAME = "YuMi"
ITEMS_PRIM_NAME = "all_items"
CART_PRIM_NAME = "Cart"
SHELF_PRIM_NAME = "Shelf"
VISION_CAMERA_PRIM_NAME = "VisionCamera"
ITEM_MASS_KG = 0.2
ROS2_JOINT_GRAPH_PATH = "/ActionGraph/ROS2JointBridge"


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

    runtime_root = Path("/tmp/isaacsim_runtime")
    runtime_root.mkdir(parents=True, exist_ok=True)

    # Preserve the scene's directory layout so relative USD asset paths
    # (for example ./textures/*) still resolve after opening the runtime copy.
    source_scene_dir = scene_file.parent
    runtime_scene_dir = runtime_root / source_scene_dir.name
    shutil.copytree(source_scene_dir, runtime_scene_dir, dirs_exist_ok=True)
    runtime_scene_file = runtime_scene_dir / scene_file.name

    if not omni.usd.get_context().open_stage(str(runtime_scene_file)):
        raise RuntimeError(
            f"Failed to open Isaac Sim scene: {scene_file} "
            f"(runtime copy {runtime_scene_file})"
        )

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Scene opened but no USD stage is available")
    return stage


def _import_robot_urdf(urdf_path: Path, dest_path: str) -> tuple[str, str]:
    import omni.kit.commands
    import omni.usd
    from isaacsim.asset.importer.urdf import _urdf
    from pxr import Sdf

    import_config = _urdf.ImportConfig()
    import_config.set_fix_base(True)
    import_config.set_make_default_prim(True)
    import_config.set_create_physics_scene(False)
    import_config.set_self_collision(True)
    import_config.set_parse_mimic(True)
    import_config.set_merge_fixed_joints(False)
    import_config.set_convex_decomp(False)
    import_config.set_collision_from_visuals(False)
    import_config.set_import_inertia_tensor(True)
    import_config.set_density(0.0)
    import_config.set_distance_scale(1.0)

    result, robot_model = omni.kit.commands.execute(
        "URDFParseFile",
        urdf_path=str(urdf_path),
        import_config=import_config,
    )
    if not result:
        raise RuntimeError(f"Failed to parse robot URDF: {urdf_path}")

    result, imported_prim_path = omni.kit.commands.execute(
        "URDFImportRobot",
        urdf_robot=robot_model,
        import_config=import_config,
    )
    if not result:
        raise RuntimeError(f"Failed to import robot URDF: {urdf_path}")

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("URDF import succeeded but no USD stage is available")

    imported_prim_path = str(imported_prim_path or "")
    if not imported_prim_path:
        raise RuntimeError("URDF import did not return a robot prim path")

    imported_prim = stage.GetPrimAtPath(imported_prim_path)
    if not imported_prim.IsValid():
        raise RuntimeError(
            f"URDF import returned invalid robot prim path: {imported_prim_path}"
        )

    if imported_prim_path != dest_path:
        result = omni.kit.commands.execute(
            "MovePrimCommand",
            path_from=imported_prim_path,
            path_to=dest_path,
            keep_world_transform=True,
        )
        move_succeeded = result[0] if isinstance(result, tuple) else bool(result)
        if not move_succeeded:
            raise RuntimeError(
                f"Failed to move imported robot prim from {imported_prim_path} to {dest_path}"
            )

    final_prim = stage.GetPrimAtPath(dest_path)
    if not final_prim.IsValid():
        final_prim = stage.GetPrimAtPath(imported_prim_path)
    if not final_prim.IsValid():
        raise RuntimeError(
            f"Robot prim is missing after URDF import: {imported_prim_path}"
        )

    return str(final_prim.GetPath()), str(imported_prim_path)


def add_robot_at_cart_origin(stage, robot_path: str) -> str:
    import omni.kit.app
    from pxr import UsdGeom

    robot_file = Path(robot_path)
    if not robot_file.exists():
        raise FileNotFoundError(f"Robot asset does not exist: {robot_file}")

    cart_prim = find_prim_named(stage, CART_PRIM_NAME)
    if cart_prim is None:
        raise RuntimeError("Could not find Cart prim in loaded scene")

    robot_prim_path = cart_prim.GetPath().AppendChild(ROBOT_PRIM_NAME)
    if robot_file.suffix.lower() != ".urdf":
        raise RuntimeError(
            f"Robot asset must be a URDF for direct import, got: {robot_file}"
        )

    imported_path, articulation_hint = _import_robot_urdf(
        robot_file, dest_path=robot_prim_path.pathString
    )
    print(
        f"[store_shelf] Directly imported URDF into scene at {imported_path} "
        f"(articulation_hint={articulation_hint})",
        flush=True,
    )
    robot_prim = stage.GetPrimAtPath(imported_path)
    if not robot_prim.IsValid():
        raise RuntimeError(
            f"URDF import reported success but robot prim is missing: {imported_path}"
        )

    robot_xform = UsdGeom.Xformable(robot_prim)
    robot_xform.ClearXformOpOrder()
    robot_xform.AddRotateZOp().Set(180.0)
    robot_xform.AddTranslateOp().Set((-0.40, 0.0, 0.0))
    # Let USD compose referenced robot contents before downstream traversal.
    omni.kit.app.get_app().update()
    return imported_path


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


def find_articulation_root_prim_path(stage, robot_prim_path: str) -> str:
    import omni.kit.app
    from pxr import Sdf, Usd, UsdPhysics

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Robot prim is not valid: {robot_prim_path}")

    # Imported URDF contents may not be traversable until a frame has advanced.
    omni.kit.app.get_app().update()

    articulation_paths = []
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_paths.append(prim.GetPath().pathString)

    if articulation_paths:
        print(
            "[store_shelf] Articulation root candidates: "
            + ", ".join(sorted(articulation_paths)),
            flush=True,
        )

    if not articulation_paths:
        fallback = f"{robot_prim_path}/root_joint"
        fallback_prim = stage.GetPrimAtPath(fallback)
        if fallback_prim.IsValid():
            print(
                f"[store_shelf] Falling back to inferred articulation root {fallback}",
                flush=True,
            )
            return fallback
        raise RuntimeError(
            f"Could not find articulation root under imported robot prim: {robot_prim_path}"
        )

    articulation_paths.sort(key=len)
    articulation_root = articulation_paths[0]
    articulation_root_path = Sdf.Path(articulation_root)
    parent_path = articulation_root_path.GetParentPath()
    if (
        not parent_path.isEmpty
        and articulation_root_path.name == parent_path.name
        and stage.GetPrimAtPath(parent_path).IsValid()
    ):
        normalized = parent_path.pathString
        print(
            f"[store_shelf] Normalizing duplicated articulation root path "
            f"{articulation_root} -> {normalized}",
            flush=True,
        )
        articulation_root = normalized
    print(
        f"[store_shelf] Resolved articulation root for {robot_prim_path} -> {articulation_root}",
        flush=True,
    )
    return articulation_root


def configure_ros2_joint_bridge(stage, robot_prim_path: str) -> dict[str, str]:
    import omni.graph.core as og

    joint_state_topic = os.environ.get("ISAACSIM_JOINT_STATE_TOPIC", "/joint_states").strip()
    joint_command_topic = os.environ.get("ISAACSIM_JOINT_COMMAND_TOPIC", "/joint_command").strip()
    node_namespace = os.environ.get("ISAACSIM_ROS2_NODE_NAMESPACE", "").strip()
    print(
        f"[store_shelf] Resolving articulation root under {robot_prim_path} for ROS 2 joint bridge",
        flush=True,
    )
    articulation_root_path = find_articulation_root_prim_path(stage, robot_prim_path)

    og.Controller.edit(
        {"graph_path": ROS2_JOINT_GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("SubscribeJointState.outputs:execOut", "ArticulationController.inputs:execIn"),
                (
                    "SubscribeJointState.outputs:jointNames",
                    "ArticulationController.inputs:jointNames",
                ),
                (
                    "SubscribeJointState.outputs:positionCommand",
                    "ArticulationController.inputs:positionCommand",
                ),
                (
                    "SubscribeJointState.outputs:velocityCommand",
                    "ArticulationController.inputs:velocityCommand",
                ),
                (
                    "SubscribeJointState.outputs:effortCommand",
                    "ArticulationController.inputs:effortCommand",
                ),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishJointState.inputs:targetPrim", [articulation_root_path]),
                ("PublishJointState.inputs:topicName", joint_state_topic),
                ("SubscribeJointState.inputs:topicName", joint_command_topic),
                ("ArticulationController.inputs:robotPath", articulation_root_path),
                ("PublishJointState.inputs:nodeNamespace", node_namespace),
                ("SubscribeJointState.inputs:nodeNamespace", node_namespace),
            ],
        },
    )

    print(
        "[store_shelf] Configured Isaac Sim ROS 2 joint bridge "
        f"(joint_states={joint_state_topic}, joint_command={joint_command_topic}, "
        f"articulation_root={articulation_root_path})",
        flush=True,
    )
    return {
        "joint_state_topic": joint_state_topic,
        "joint_command_topic": joint_command_topic,
        "node_namespace": node_namespace,
        "articulation_root_path": articulation_root_path,
    }


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
    ros2_joint_bridge = None
    if configuration == "store_demo":
        ros2_joint_bridge = configure_ros2_joint_bridge(stage, robot_prim_path)
    cart_base_translation = prim_local_translation(cart_prim)

    return {
        "stage": stage,
        "scene_path": scene_path,
        "robot_path": robot_path,
        "robot_prim_path": robot_prim_path,
        "ros2_joint_bridge": ros2_joint_bridge,
        "cart_prim": cart_prim,
        "cart_base_translation": cart_base_translation,
    }
