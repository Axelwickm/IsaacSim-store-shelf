#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import xacro


def _add_boolean_flag(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    default: bool,
    help_text: str,
) -> None:
    parser.add_argument(
        f"--{name}",
        action=argparse.BooleanOptionalAction,
        default=default,
        help=help_text,
    )


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an Isaac Sim-ready YuMi URDF from the canonical xacro source."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/usd/robot/yumi_isaacsim.urdf"),
        help="Destination for the sanitized Isaac Sim URDF.",
    )
    _add_boolean_flag(
        parser,
        "with-ros2-control",
        default=False,
        help_text="Include the ros2_control topic hardware block in the exported URDF.",
    )
    _add_boolean_flag(
        parser,
        "include-left-arm",
        default=True,
        help_text="Include the left arm in the exported URDF.",
    )
    _add_boolean_flag(
        parser,
        "include-right-arm",
        default=True,
        help_text="Include the right arm in the exported URDF.",
    )
    _add_boolean_flag(
        parser,
        "include-left-gripper",
        default=True,
        help_text="Include the left gripper in the exported URDF.",
    )
    _add_boolean_flag(
        parser,
        "include-right-gripper",
        default=True,
        help_text="Include the right gripper in the exported URDF.",
    )
    _add_boolean_flag(
        parser,
        "rewrite-mesh-paths",
        default=False,
        help_text="Rewrite package mesh URIs to absolute filesystem paths for Isaac/curobo.",
    )
    return parser


def _strip_tag(root: ET.Element, tag_name: str) -> None:
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == tag_name:
                parent.remove(child)


def _replace_link_references(root: ET.Element, old_name: str, new_name: str) -> None:
    for element in root.iter():
        for attribute in ("link", "link1", "link2", "base_link", "tip_link", "parent_link", "child_link"):
            if element.get(attribute) == old_name:
                element.set(attribute, new_name)


def _strip_named_link(root: ET.Element, link_name: str, replacement_child: str | None = None) -> None:
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == "joint":
                child_parent = child.find("parent")
                child_child = child.find("child")
                if child_parent is not None and child_parent.get("link") == link_name:
                    if replacement_child is not None:
                        child_parent.set("link", replacement_child)
                    else:
                        parent.remove(child)
                elif child_child is not None and child_child.get("link") == link_name:
                    if replacement_child is not None:
                        child_child.set("link", replacement_child)
                    else:
                        parent.remove(child)
                if child in parent:
                    child_parent = child.find("parent")
                    child_child = child.find("child")
                    if (
                        child_parent is not None
                        and child_child is not None
                        and child_parent.get("link") == child_child.get("link")
                    ):
                        parent.remove(child)
            elif child.tag == "link" and child.get("name") == link_name:
                parent.remove(child)
    if replacement_child is not None:
        _replace_link_references(root, link_name, replacement_child)


def _rewrite_mesh_paths(root: ET.Element, package_share: Path) -> None:
    meshes_root = package_share / "meshes"
    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename or not filename.startswith("package://yumi_description/meshes/"):
            continue
        relative_path = filename.removeprefix("package://yumi_description/meshes/")
        mesh.set("filename", str((meshes_root / relative_path).resolve()))


def _is_arm_link_for_side(link_name: str, side: str) -> bool:
    return (
        link_name == "yumi_body"
        or link_name.startswith(f"yumi_link_") and link_name.endswith(f"_{side}")
        or link_name.startswith(f"gripper_{side}_")
    )


def _is_joint_for_side(joint: ET.Element, side: str) -> bool:
    name = joint.get("name") or ""
    if name.startswith("yumi_joint_") and name.endswith(f"_{side}"):
        return True
    if name.startswith("gripper_"):
        return name.startswith(f"gripper_{side}_")
    parent = joint.find("parent")
    child = joint.find("child")
    parent_link = parent.get("link", "") if parent is not None else ""
    child_link = child.get("link", "") if child is not None else ""
    return _is_arm_link_for_side(parent_link, side) and _is_arm_link_for_side(child_link, side)


def _strip_other_side(root: ET.Element, side: str) -> None:
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == "joint" and not _is_joint_for_side(child, side):
                parent.remove(child)
            elif child.tag == "link":
                name = child.get("name") or ""
                if not _is_arm_link_for_side(name, side):
                    parent.remove(child)


def export_urdf(
    output_path: Path,
    *,
    with_ros2_control: bool = False,
    include_left_arm: bool = True,
    include_right_arm: bool = True,
    include_left_gripper: bool = True,
    include_right_gripper: bool = True,
    rewrite_mesh_paths: bool = True,
) -> Path:
    package_share = Path(get_package_share_directory("yumi_description"))
    xacro_file = package_share / "urdf" / "yumi.urdf.xacro"
    document = xacro.process_file(
        str(xacro_file),
        mappings={
            "arms_interface": "PositionJointInterface",
            "grippers_interface": "PositionJointInterface",
            "use_ros2_control": "true" if with_ros2_control else "false",
            "joint_states_topic": "/isaac_joint_states",
            "joint_commands_topic": "/joint_command",
            "include_left_arm": "true" if include_left_arm else "false",
            "include_right_arm": "true" if include_right_arm else "false",
            "include_left_gripper": "true" if include_left_gripper else "false",
            "include_right_gripper": "true" if include_right_gripper else "false",
        },
    )
    root = ET.fromstring(document.toprettyxml(indent="  "))

    _strip_tag(root, "gazebo")
    _strip_tag(root, "transmission")
    _strip_tag(root, "ros2_control")
    _strip_named_link(root, "world")
    if include_left_arm and not include_right_arm:
        _strip_other_side(root, "l")
    elif include_right_arm and not include_left_arm:
        _strip_other_side(root, "r")
    if rewrite_mesh_paths:
        _rewrite_mesh_paths(root, package_share)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def main() -> None:
    args = _make_parser().parse_args()
    output_path = export_urdf(
        args.output.resolve(),
        with_ros2_control=args.with_ros2_control,
        include_left_arm=args.include_left_arm,
        include_right_arm=args.include_right_arm,
        include_left_gripper=args.include_left_gripper,
        include_right_gripper=args.include_right_gripper,
        rewrite_mesh_paths=args.rewrite_mesh_paths,
    )
    print(f"Wrote Isaac Sim YuMi URDF to {output_path}")


if __name__ == "__main__":
    main()
