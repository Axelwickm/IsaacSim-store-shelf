#!/usr/bin/env python3

from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node


DEFAULT_DATASET_DIR = Path("/workspace/collect_vision_data_output")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".exr")


def main() -> None:
    rclpy.init()
    node = Node("vision_dataset_viewer")
    node.declare_parameter("dataset_dir", str(DEFAULT_DATASET_DIR))
    dataset_dir = Path(str(node.get_parameter("dataset_dir").value)).resolve()

    if not dataset_dir.exists():
        node.get_logger().error(f"Dataset directory does not exist: {dataset_dir}")
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(1)

    image_files = sorted(
        path for path in dataset_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        node.get_logger().error(f"No image files found under {dataset_dir}")
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(1)

    first_image = image_files[0]
    image = cv2.imread(str(first_image), cv2.IMREAD_UNCHANGED)
    if image is None:
        node.get_logger().error(f"Failed to load image: {first_image}")
        node.destroy_node()
        rclpy.shutdown()
        raise SystemExit(1)

    node.get_logger().info(f"Showing {first_image}")
    cv2.imshow("vision_dataset_preview", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
