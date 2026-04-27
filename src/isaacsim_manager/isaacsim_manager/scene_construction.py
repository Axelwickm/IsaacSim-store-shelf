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
COLLIDERS_PRIM_NAME = "colliders"
DEFAULT_CART_Y = -1.0
DEFAULT_ROBOT_RELATIVE_X = 0.1
VISION_CAMERA_LINK_PRIM_NAME = "vision_camera_link"
VISION_CAMERA_OPTICAL_FRAME_PRIM_NAME = "vision_camera_optical_frame"
DROP_OFF_TARGETS_PRIM_NAME = "DropoffTargets"
RIGHT_ARM_DROPOFF_PRIM_NAME = "right_arm_dropoff"
LEFT_ARM_DROPOFF_PRIM_NAME = "left_arm_dropoff"
RIGHT_ARM_DROPOFF_POSITION_YUMI_BODY = (0.25, -0.14, 0.0)
LEFT_ARM_DROPOFF_POSITION_YUMI_BODY = (0.25, 0.14, 0.0)
ITEM_MASS_KG = 0.2
ROS2_JOINT_GRAPH_PATH = "/ActionGraph/ROS2JointBridge"
ROS2_CAMERA_GRAPH_PATH = "/ActionGraph/ROS2CameraBridge"
ROS2_DYNAMIC_TF_GRAPH_PATH = "/ActionGraph/ROS2DynamicTF"
ROS2_STATIC_TF_GRAPH_PATH = "/ActionGraph/ROS2StaticTF"
DEFAULT_CAMERA_TOPIC = "/camera/image_raw"
DEFAULT_CAMERA_INFO_TOPIC = "/camera/camera_info"
DEFAULT_CAMERA_FRAME_ID = VISION_CAMERA_OPTICAL_FRAME_PRIM_NAME
DEFAULT_CAMERA_RESOLUTION = 512
ROBOT_PHYSICAL_ROOT_LINK_NAME = "yumi_body"
YUMI_ARM_JOINT_LIMITS_RAD = {
    # Match MoveIt's slightly tighter configured limits so randomized Isaac
    # starts are always accepted by CheckStartStateBounds.
    "yumi_joint_1_r": (-2.84, 2.84),
    "yumi_joint_2_r": (-2.4, 0.65),
    "yumi_joint_7_r": (-2.84, 2.84),
    "yumi_joint_3_r": (-2.0, 1.29),
    "yumi_joint_4_r": (-4.9, 4.9),
    "yumi_joint_5_r": (-1.43, 2.3),
    "yumi_joint_6_r": (-3.89, 3.89),
    "yumi_joint_1_l": (-2.84, 2.84),
    "yumi_joint_2_l": (-2.4, 0.65),
    "yumi_joint_7_l": (-2.84, 2.84),
    "yumi_joint_3_l": (-2.0, 1.29),
    "yumi_joint_4_l": (-4.9, 4.9),
    "yumi_joint_5_l": (-1.43, 2.3),
    "yumi_joint_6_l": (-3.89, 3.89),
}
YUMI_STORE_DEMO_READY_JOINT_POSITIONS_RAD = {
    # Keep the left arm parked away from the right arm, while the right arm starts
    # roughly shelf-facing. These values are mirrored in config/yumi.srdf.
    "yumi_joint_1_l": 2.4,
    "yumi_joint_2_l": -0.6,
    "yumi_joint_7_l": 0.2,
    "yumi_joint_3_l": 0.2,
    "yumi_joint_4_l": -1.2,
    "yumi_joint_5_l": 0.3,
    "yumi_joint_6_l": 0.0,
    "yumi_joint_1_r": -1.2,
    "yumi_joint_2_r": -0.8,
    "yumi_joint_7_r": -0.4,
    "yumi_joint_3_r": -0.6,
    "yumi_joint_4_r": 1.2,
    "yumi_joint_5_r": 0.8,
    "yumi_joint_6_r": 0.0,
}


def find_prim_named(stage, name: str):
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    return None


def find_prim_named_case_insensitive(stage, name: str):
    target = name.casefold()
    for prim in stage.Traverse():
        if prim.GetName().casefold() == target:
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


def apply_static_box_collider(prim) -> None:
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    collision_api = UsdPhysics.CollisionAPI.Apply(prim)
    collision_api.CreateCollisionEnabledAttr().Set(True)
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
            "boundingCube"
        )
    PhysxSchema.PhysxCollisionAPI.Apply(prim)

    try:
        from omni.physx.scripts import utils as physx_utils

        physx_utils.setCollider(prim, approximationShape="boundingCube")
    except Exception:
        pass

    UsdGeom.Imageable(prim).MakeInvisible()


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
            keep_world_transform=False,
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
    robot_xform.AddTranslateOp().Set((DEFAULT_ROBOT_RELATIVE_X, 0.0, 0.0))
    # Let USD compose referenced robot contents before downstream traversal.
    omni.kit.app.get_app().update()
    return imported_path


def get_robot_physical_root_prim(stage, robot_prim_path: str):
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        raise RuntimeError(f"Robot prim is not valid: {robot_prim_path}")

    robot_root_prim_path = f"{robot_prim_path}/{ROBOT_PHYSICAL_ROOT_LINK_NAME}"
    robot_root_prim = stage.GetPrimAtPath(robot_root_prim_path)
    if not robot_root_prim.IsValid():
        raise RuntimeError(
            "Robot physical root prim is missing. Isaac Sim must expose "
            f"{robot_root_prim_path}"
        )
    return robot_root_prim


def add_robot_owned_camera(stage, robot_prim_path: str) -> str:
    from pxr import Gf, UsdGeom

    robot_mount_prim = get_robot_physical_root_prim(stage, robot_prim_path)

    camera_link_path = (
        f"{robot_mount_prim.GetPath().pathString}/{VISION_CAMERA_LINK_PRIM_NAME}"
    )
    camera_link = UsdGeom.Xform.Define(stage, camera_link_path)
    camera_link_xform = UsdGeom.Xformable(camera_link.GetPrim())
    camera_link_xform.ClearXformOpOrder()
    camera_link_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.8))
    camera_link_xform.AddRotateXYZOp().Set(Gf.Vec3f(-180.0, -30.0, -180.0))

    camera_path = f"{camera_link_path}/{VISION_CAMERA_OPTICAL_FRAME_PRIM_NAME}"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr().Set(14.0)
    camera.CreateFocusDistanceAttr().Set(1.0)
    camera.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000000.0))

    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, -90.0))
    print(
        "[store_shelf] Mounted vision camera under "
        f"{robot_mount_prim.GetPath().pathString}",
        flush=True,
    )
    return camera_path


def add_dropoff_target_markers(stage, robot_prim_path: str) -> dict[str, str]:
    from pxr import Gf, Sdf, UsdGeom

    robot_mount_prim = get_robot_physical_root_prim(stage, robot_prim_path)
    root_path = f"{robot_mount_prim.GetPath().pathString}/{DROP_OFF_TARGETS_PRIM_NAME}"
    root = UsdGeom.Xform.Define(stage, root_path)
    root_xform = UsdGeom.Xformable(root.GetPrim())
    root_xform.ClearXformOpOrder()
    root_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

    markers = {
        "right": (
            RIGHT_ARM_DROPOFF_PRIM_NAME,
            RIGHT_ARM_DROPOFF_POSITION_YUMI_BODY,
            Gf.Vec3f(0.1, 0.45, 1.0),
        ),
        "left": (
            LEFT_ARM_DROPOFF_PRIM_NAME,
            LEFT_ARM_DROPOFF_POSITION_YUMI_BODY,
            Gf.Vec3f(1.0, 0.55, 0.1),
        ),
    }
    marker_paths = {}
    for side, (prim_name, position, color) in markers.items():
        marker_path = f"{root_path}/{prim_name}"
        marker = UsdGeom.Cylinder.Define(stage, Sdf.Path(marker_path))
        marker.CreateRadiusAttr().Set(0.08)
        marker.CreateHeightAttr().Set(0.015)
        marker.CreateDisplayColorAttr().Set([color])
        marker.CreateDisplayOpacityAttr().Set([0.55])
        marker_xform = UsdGeom.Xformable(marker.GetPrim())
        marker_xform.ClearXformOpOrder()
        marker_xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        marker_paths[side] = marker_path

    print(
        "[store_shelf] Added dropoff target markers in "
        f"{ROBOT_PHYSICAL_ROOT_LINK_NAME}: "
        f"right={RIGHT_ARM_DROPOFF_POSITION_YUMI_BODY}, "
        f"left={LEFT_ARM_DROPOFF_POSITION_YUMI_BODY}",
        flush=True,
    )
    return marker_paths


def configure_ros2_camera_bridge(
    stage,
    camera_prim_path: str,
    cart_prim_path: str,
    robot_prim_path: str,
) -> dict[str, str]:
    import omni.graph.core as og
    import omni.replicator.core as rep
    import usdrt.Sdf

    node_namespace = os.environ.get("ISAACSIM_ROS2_NODE_NAMESPACE", "").strip()
    camera_topic = os.environ.get("ISAACSIM_CAMERA_TOPIC", DEFAULT_CAMERA_TOPIC).strip()
    camera_info_topic = os.environ.get(
        "ISAACSIM_CAMERA_INFO_TOPIC",
        DEFAULT_CAMERA_INFO_TOPIC,
    ).strip()
    camera_frame_id = os.environ.get(
        "ISAACSIM_CAMERA_FRAME_ID",
        DEFAULT_CAMERA_FRAME_ID,
    ).strip()
    camera_resolution = int(
        os.environ.get("ISAACSIM_CAMERA_RESOLUTION", str(DEFAULT_CAMERA_RESOLUTION)).strip()
    )
    queue_size = int(os.environ.get("ISAACSIM_CAMERA_QUEUE_SIZE", "1").strip())

    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    if not camera_prim.IsValid():
        raise RuntimeError(f"Camera prim is not valid: {camera_prim_path}")

    camera_link_prim = camera_prim.GetParent()
    if not camera_link_prim.IsValid():
        raise RuntimeError(f"Camera link prim is not valid for camera: {camera_prim_path}")

    camera_mount_parent_prim = camera_link_prim.GetParent()
    if not camera_mount_parent_prim.IsValid():
        raise RuntimeError(
            f"Camera mount parent prim is not valid for camera link: {camera_link_prim.GetPath()}"
        )
    robot_root_prim = get_robot_physical_root_prim(stage, robot_prim_path)

    render_product = rep.create.render_product(
        camera_prim_path,
        (camera_resolution, camera_resolution),
    )
    render_product_path = getattr(render_product, "path", None) or str(render_product)

    og.Controller.edit(
        {"graph_path": ROS2_CAMERA_GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("PublishImage", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("PublishCameraInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishImage.inputs:frameId", camera_frame_id),
                ("PublishImage.inputs:topicName", camera_topic),
                ("PublishImage.inputs:nodeNamespace", node_namespace),
                ("PublishImage.inputs:queueSize", queue_size),
                ("PublishImage.inputs:type", "rgb"),
                ("PublishImage.inputs:renderProductPath", render_product_path),
                ("PublishCameraInfo.inputs:frameId", camera_frame_id),
                ("PublishCameraInfo.inputs:topicName", camera_info_topic),
                ("PublishCameraInfo.inputs:nodeNamespace", node_namespace),
                ("PublishCameraInfo.inputs:queueSize", queue_size),
                ("PublishCameraInfo.inputs:renderProductPath", render_product_path),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishImage.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishCameraInfo.inputs:execIn"),
            ],
        },
    )

    og.Controller.edit(
        {"graph_path": ROS2_DYNAMIC_TF_GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishDynamicTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishDynamicTF.inputs:topicName", "tf"),
                (
                    "PublishDynamicTF.inputs:targetPrims",
                    [usdrt.Sdf.Path(cart_prim_path)],
                ),
                ("PublishDynamicTF.inputs:nodeNamespace", node_namespace),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishDynamicTF.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishDynamicTF.inputs:timeStamp"),
            ],
        },
    )

    og.Controller.edit(
        {"graph_path": ROS2_STATIC_TF_GRAPH_PATH, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishRobotRootTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("PublishCameraLinkTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("PublishCameraOpticalTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishRobotRootTF.inputs:topicName", "tf_static"),
                ("PublishRobotRootTF.inputs:staticPublisher", True),
                (
                    "PublishRobotRootTF.inputs:targetPrims",
                    [usdrt.Sdf.Path(robot_root_prim.GetPath().pathString)],
                ),
                (
                    "PublishRobotRootTF.inputs:parentPrim",
                    [usdrt.Sdf.Path(cart_prim_path)],
                ),
                ("PublishRobotRootTF.inputs:nodeNamespace", node_namespace),
                ("PublishCameraLinkTF.inputs:topicName", "tf_static"),
                ("PublishCameraLinkTF.inputs:staticPublisher", True),
                (
                    "PublishCameraLinkTF.inputs:targetPrims",
                    [usdrt.Sdf.Path(camera_link_prim.GetPath().pathString)],
                ),
                (
                    "PublishCameraLinkTF.inputs:parentPrim",
                    [usdrt.Sdf.Path(camera_mount_parent_prim.GetPath().pathString)],
                ),
                ("PublishCameraLinkTF.inputs:nodeNamespace", node_namespace),
                ("PublishCameraOpticalTF.inputs:topicName", "tf_static"),
                ("PublishCameraOpticalTF.inputs:staticPublisher", True),
                (
                    "PublishCameraOpticalTF.inputs:targetPrims",
                    [usdrt.Sdf.Path(camera_prim.GetPath().pathString)],
                ),
                (
                    "PublishCameraOpticalTF.inputs:parentPrim",
                    [usdrt.Sdf.Path(camera_link_prim.GetPath().pathString)],
                ),
                ("PublishCameraOpticalTF.inputs:nodeNamespace", node_namespace),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishRobotRootTF.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishCameraLinkTF.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishCameraOpticalTF.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishRobotRootTF.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishCameraLinkTF.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishCameraOpticalTF.inputs:timeStamp"),
            ],
        },
    )

    print(
        "[store_shelf] Configured Isaac Sim ROS 2 camera bridge "
        f"(image={camera_topic}, camera_info={camera_info_topic}, "
        f"frame_id={camera_frame_id}, resolution={camera_resolution}x{camera_resolution}, "
        "publisher=builtin)",
        flush=True,
    )
    print(
        f"[store_shelf] Camera render product path: {render_product_path}",
        flush=True,
    )
    print(
        "[store_shelf] Configured Isaac Sim ROS 2 TF publishers "
        f"(dynamic_prims={cart_prim_path}; "
        f"robot_root={robot_root_prim.GetPath().pathString}; "
        f"camera_mount_parent={camera_mount_parent_prim.GetPath().pathString}; "
        f"camera_link={camera_link_prim.GetPath().pathString}; "
        f"camera_optical={camera_prim.GetPath().pathString})",
        flush=True,
    )
    return {
        "camera_topic": camera_topic,
        "camera_info_topic": camera_info_topic,
        "camera_frame_id": camera_frame_id,
        "camera_prim_path": camera_prim_path,
        "render_product_path": render_product_path,
        "robot_root_prim_path": robot_root_prim.GetPath().pathString,
        "camera_link_prim_path": camera_link_prim.GetPath().pathString,
        "cart_prim_path": cart_prim_path,
        "node_namespace": node_namespace,
    }


def find_articulation_root_prim_path(stage, robot_prim_path: str) -> str:
    import omni.kit.app
    from pxr import UsdPhysics

    # Imported URDF contents may not be traversable until a frame has advanced.
    omni.kit.app.get_app().update()

    articulation_root_prim = get_robot_physical_root_prim(stage, robot_prim_path)
    articulation_root = articulation_root_prim.GetPath().pathString
    if not articulation_root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Robot physical root is not a rigid body: {articulation_root}")
    if not articulation_root_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        UsdPhysics.ArticulationRootAPI.Apply(articulation_root_prim)

    print(
        f"[store_shelf] Resolved articulation root for {robot_prim_path} -> {articulation_root}",
        flush=True,
    )
    return articulation_root


def configure_ros2_joint_bridge(stage, robot_prim_path: str) -> dict[str, str]:
    import omni.graph.core as og

    joint_state_topic = os.environ.get(
        "ISAACSIM_JOINT_STATE_TOPIC", "/isaac_joint_states"
    ).strip()
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

    collider_count = 0
    collider_paths = []
    collider_descendants = []
    colliders_prim = find_prim_named_case_insensitive(stage, COLLIDERS_PRIM_NAME)
    if colliders_prim is not None:
        UsdGeom.Imageable(colliders_prim).MakeInvisible()
        for prim in Usd.PrimRange(colliders_prim):
            collider_descendants.append(
                f"{prim.GetPath().pathString}:{prim.GetTypeName() or 'typeless'}"
            )
            if not prim.IsA(UsdGeom.Gprim):
                continue
            apply_static_box_collider(prim)
            collider_count += 1
            collider_paths.append(f"{prim.GetPath().pathString}:{prim.GetTypeName()}")
    else:
        collider_descendants = [
            f"{prim.GetPath().pathString}:{prim.GetTypeName() or 'typeless'}"
            for prim in stage.Traverse()
            if COLLIDERS_PRIM_NAME in prim.GetPath().pathString.casefold()
        ]

    print(
        "[store_shelf] Configured item, shelf, and static collider physics "
        f"(colliders={collider_count}"
        + (
            f", root={colliders_prim.GetPath().pathString}"
            if colliders_prim is not None
            else ", root=missing"
        )
        + (f", paths={', '.join(collider_paths)}" if collider_paths else "")
        + (
            f", descendants={', '.join(collider_descendants)}"
            if collider_descendants
            else ""
        )
        + ")",
        flush=True,
    )


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


def set_robot_arm_joints(
    stage,
    robot_prim_path: str,
    joint_positions_rad: dict[str, float],
    *,
    label: str,
) -> dict[str, float]:
    import math
    from pxr import PhysxSchema, Usd, UsdPhysics

    applied_positions = {}
    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        return applied_positions

    for prim in Usd.PrimRange(robot_prim):
        joint_name = prim.GetName()
        joint_position_rad = joint_positions_rad.get(joint_name)
        if joint_position_rad is None:
            continue

        joint_position_deg = math.degrees(joint_position_rad)
        UsdPhysics.DriveAPI.Apply(prim, "angular").CreateTargetPositionAttr().Set(
            joint_position_deg
        )
        PhysxSchema.JointStateAPI.Apply(prim, "angular").CreatePositionAttr().Set(
            joint_position_deg
        )
        PhysxSchema.JointStateAPI.Apply(prim, "angular").CreateVelocityAttr().Set(0.0)
        applied_positions[joint_name] = joint_position_rad

    print(
        f"[store_shelf] {label}: "
        + ", ".join(
            f"{joint_name}={joint_position:.3f}rad"
            for joint_name, joint_position in sorted(applied_positions.items())
        ),
        flush=True,
    )
    return applied_positions


def randomize_robot_arm_joints(stage, robot_prim_path: str) -> dict[str, float]:
    sampled_positions = {
        joint_name: random.uniform(lower, upper)
        for joint_name, (lower, upper) in YUMI_ARM_JOINT_LIMITS_RAD.items()
    }
    return set_robot_arm_joints(
        stage,
        robot_prim_path,
        sampled_positions,
        label="Randomized robot arm joints",
    )


def set_robot_arm_joints_for_planning(stage, robot_prim_path: str) -> dict[str, float]:
    return set_robot_arm_joints(
        stage,
        robot_prim_path,
        YUMI_STORE_DEMO_READY_JOINT_POSITIONS_RAD,
        label="Set robot arm joints to store_demo_ready pose",
    )


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
        set_semantic_label(robot_prim, "background")

    cart_prim = find_prim_named(stage, CART_PRIM_NAME)
    if cart_prim is not None:
        set_semantic_label(cart_prim, "background")

    shelf_prim = find_prim_named(stage, SHELF_PRIM_NAME)
    if shelf_prim is not None:
        set_semantic_label(shelf_prim, "background")

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
    cart_translation = prim_local_translation(cart_prim)
    set_prim_local_translation(
        cart_prim,
        (cart_translation[0], DEFAULT_CART_Y, cart_translation[2]),
    )

    robot_prim_path = add_robot_at_cart_origin(stage, robot_path)
    ros2_joint_bridge = None
    ros2_camera_bridge = None
    dropoff_target_markers = None
    if configuration == "store_demo":
        ros2_joint_bridge = configure_ros2_joint_bridge(stage, robot_prim_path)
        dropoff_target_markers = add_dropoff_target_markers(stage, robot_prim_path)
        camera_prim_path = add_robot_owned_camera(stage, robot_prim_path)
        ros2_camera_bridge = configure_ros2_camera_bridge(
            stage,
            camera_prim_path=camera_prim_path,
            cart_prim_path=cart_prim.GetPath().pathString,
            robot_prim_path=robot_prim_path,
        )
    cart_base_translation = prim_local_translation(cart_prim)

    return {
        "stage": stage,
        "scene_path": scene_path,
        "robot_path": robot_path,
        "robot_prim_path": robot_prim_path,
        "ros2_joint_bridge": ros2_joint_bridge,
        "ros2_camera_bridge": ros2_camera_bridge,
        "dropoff_target_markers": dropoff_target_markers,
        "cart_prim": cart_prim,
        "cart_base_translation": cart_base_translation,
    }
