from geometry_msgs.msg import PointStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Empty


DEFAULT_LATCHED_TARGET_POINT_TOPIC = "/motion/latched_target_point"
DEFAULT_CLEAR_LATCHED_TARGET_TOPIC = "/motion/clear_latched_target"
DEFAULT_MARKER_ROOT_PATH = "/LatchedTargetMarker"
DEFAULT_MARKER_RADIUS_METERS = 0.035
DEFAULT_DEBUG_DRAW_RADIUS_PIXELS = 18.0
DEFAULT_DEBUG_DRAW_CROSSHAIR_SIZE_METERS = 0.12
DEFAULT_MAX_MARKERS = 20


class IsaacSimTargetMarkerVisualizer:
    def __init__(
        self,
        target_topic_name: str = DEFAULT_LATCHED_TARGET_POINT_TOPIC,
        clear_topic_name: str = DEFAULT_CLEAR_LATCHED_TARGET_TOPIC,
        marker_root_path: str = DEFAULT_MARKER_ROOT_PATH,
        marker_radius_meters: float = DEFAULT_MARKER_RADIUS_METERS,
        debug_draw_radius_pixels: float = DEFAULT_DEBUG_DRAW_RADIUS_PIXELS,
        debug_draw_crosshair_size_meters: float = DEFAULT_DEBUG_DRAW_CROSSHAIR_SIZE_METERS,
        max_markers: int = DEFAULT_MAX_MARKERS,
    ) -> None:
        self._target_topic_name = target_topic_name
        self._clear_topic_name = clear_topic_name
        self._marker_root_path = marker_root_path
        self._marker_radius_meters = marker_radius_meters
        self._debug_draw_radius_pixels = debug_draw_radius_pixels
        self._debug_draw_crosshair_size_meters = debug_draw_crosshair_size_meters
        self._max_markers = max_markers
        self._marker_index = 0
        self._marker_paths: list[str] = []
        self._debug_draw_points: list[tuple[float, float, float]] = []
        self._debug_draw = None
        self._debug_draw_checked = False
        self._pending_points: list[PointStamped] = []
        self._pending_clear = False
        self._executor = SingleThreadedExecutor()
        self._node = Node("isaacsim_latched_target_marker")
        self._executor.add_node(self._node)
        self._target_subscription = self._node.create_subscription(
            PointStamped,
            target_topic_name,
            self._handle_target_point,
            10,
        )
        self._clear_subscription = self._node.create_subscription(
            Empty,
            clear_topic_name,
            self._handle_clear,
            10,
        )

    def _handle_target_point(self, message: PointStamped) -> None:
        self._pending_points.append(message)

    def _handle_clear(self, _message: Empty) -> None:
        self._pending_clear = True

    def update(self) -> None:
        self._executor.spin_once(timeout_sec=0.0)
        if self._pending_clear:
            self._clear_markers()
            self._pending_clear = False

        while self._pending_points:
            self._add_marker(self._pending_points.pop(0))

    def _stage(self):
        import omni.usd

        return omni.usd.get_context().get_stage()

    def _add_marker(self, message: PointStamped) -> None:
        if message.header.frame_id not in {"", "world"}:
            self._node.get_logger().warning(
                "Latched target marker expects world-frame points; "
                f"got {message.header.frame_id!r}. Drawing it as stage coordinates."
            )

        stage = self._stage()
        if stage is None:
            return

        from pxr import Gf, Sdf, UsdGeom

        marker_path = f"{self._marker_root_path}_{self._marker_index:03d}"
        self._marker_index += 1

        sphere = UsdGeom.Sphere.Define(stage, Sdf.Path(marker_path))
        sphere.CreateRadiusAttr().Set(self._marker_radius_meters)
        sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.05, 0.0)])
        sphere.CreateDisplayOpacityAttr().Set([0.45])
        UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(
            Gf.Vec3d(
                float(message.point.x),
                float(message.point.y),
                float(message.point.z),
            )
        )

        self._marker_paths.append(marker_path)
        point = (
            float(message.point.x),
            float(message.point.y),
            float(message.point.z),
        )
        self._debug_draw_points.append(point)
        while len(self._marker_paths) > self._max_markers:
            old_marker_path = self._marker_paths.pop(0)
            stage.RemovePrim(Sdf.Path(old_marker_path))
            if self._debug_draw_points:
                self._debug_draw_points.pop(0)
        self._redraw_debug_markers()

        local_translation, world_translation = self._marker_translations(
            stage,
            marker_path,
        )
        self._node.get_logger().info(
            "Added Isaac Sim latched target marker "
            f"path={marker_path} point=({message.point.x:.3f}, "
            f"{message.point.y:.3f}, {message.point.z:.3f}) "
            f"local_translation={local_translation} "
            f"world_translation={world_translation}"
        )

    def _marker_translations(self, stage, marker_path: str) -> tuple[str, str]:
        from pxr import Usd, UsdGeom

        prim = stage.GetPrimAtPath(marker_path)
        if not prim.IsValid():
            return "invalid", "invalid"

        xformable = UsdGeom.Xformable(prim)
        local_transform = xformable.GetLocalTransformation()
        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        local_translation = local_transform.ExtractTranslation()
        world_translation = world_transform.ExtractTranslation()
        return (
            self._format_vec3(local_translation),
            self._format_vec3(world_translation),
        )

    def _format_vec3(self, value) -> str:
        return f"({float(value[0]):.3f}, {float(value[1]):.3f}, {float(value[2]):.3f})"

    def _debug_draw_interface(self):
        if self._debug_draw_checked:
            return self._debug_draw
        self._debug_draw_checked = True

        try:
            from isaacsim.util.debug_draw import _debug_draw

            self._debug_draw = _debug_draw.acquire_debug_draw_interface()
            return self._debug_draw
        except Exception:
            pass

        try:
            from omni.isaac.debug_draw import _debug_draw

            self._debug_draw = _debug_draw.acquire_debug_draw_interface()
            return self._debug_draw
        except Exception as error:
            self._node.get_logger().warning(
                f"Isaac debug draw is unavailable; using USD marker spheres only: {error}"
            )
            return None

    def _redraw_debug_markers(self) -> None:
        debug_draw = self._debug_draw_interface()
        if debug_draw is None:
            return

        try:
            debug_draw.clear_points()
            debug_draw.clear_lines()
        except Exception:
            pass

        if not self._debug_draw_points:
            return

        point_colors = [(1.0, 0.0, 0.0, 1.0)] * len(self._debug_draw_points)
        point_sizes = [self._debug_draw_radius_pixels] * len(self._debug_draw_points)
        debug_draw.draw_points(self._debug_draw_points, point_colors, point_sizes)

        line_points_0 = []
        line_points_1 = []
        line_colors = []
        line_widths = []
        half_size = self._debug_draw_crosshair_size_meters * 0.5
        for x, y, z in self._debug_draw_points:
            line_points_0.extend(
                [
                    (x - half_size, y, z),
                    (x, y - half_size, z),
                    (x, y, z - half_size),
                ]
            )
            line_points_1.extend(
                [
                    (x + half_size, y, z),
                    (x, y + half_size, z),
                    (x, y, z + half_size),
                ]
            )
            line_colors.extend(
                [
                    (1.0, 0.0, 0.0, 1.0),
                    (0.0, 1.0, 0.0, 1.0),
                    (0.1, 0.4, 1.0, 1.0),
                ]
            )
            line_widths.extend([3.0, 3.0, 3.0])
        debug_draw.draw_lines(line_points_0, line_points_1, line_colors, line_widths)

    def _clear_markers(self) -> None:
        stage = self._stage()
        if stage is None:
            return

        from pxr import Sdf

        for marker_path in self._marker_paths:
            stage.RemovePrim(Sdf.Path(marker_path))
        self._marker_paths.clear()
        self._debug_draw_points.clear()
        self._redraw_debug_markers()
        self._node.get_logger().info("Cleared Isaac Sim latched target markers")

    def close(self) -> None:
        try:
            self._executor.remove_node(self._node)
        except Exception:
            pass
        debug_draw = self._debug_draw_interface()
        if debug_draw is not None:
            try:
                debug_draw.clear_points()
                debug_draw.clear_lines()
            except Exception:
                pass
        self._node.destroy_node()
