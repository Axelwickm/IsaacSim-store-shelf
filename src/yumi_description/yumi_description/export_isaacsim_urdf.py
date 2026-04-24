#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import xacro


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


def export_urdf(output_path: Path) -> Path:
    package_share = Path(get_package_share_directory("yumi_description"))
    xacro_file = package_share / "urdf" / "yumi.urdf.xacro"
    document = xacro.process_file(
        str(xacro_file),
        mappings={
            "arms_interface": "PositionJointInterface",
            "grippers_interface": "PositionJointInterface",
        },
    )
    root = ET.fromstring(document.toprettyxml(indent="  "))

    _strip_tag(root, "gazebo")
    _strip_tag(root, "transmission")
    _strip_named_link(root, "world")
    _strip_named_link(root, "yumi_base_link", replacement_child="yumi_body")
    _rewrite_mesh_paths(root, package_share)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root)
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def main() -> None:
    args = _make_parser().parse_args()
    output_path = export_urdf(args.output.resolve())
    print(f"Wrote Isaac Sim YuMi URDF to {output_path}")


if __name__ == "__main__":
    main()
