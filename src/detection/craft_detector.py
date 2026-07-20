"""
CRAFT-based text detection for document images.

Detects text regions, crops them, and sorts them in natural reading order
(top-to-bottom, left-to-right). Each region is tagged with a line_idx
so downstream modules can reconstruct multi-line text.
"""

import os
import cv2
import numpy as np
from craft_text_detector import Craft


class TextDetector:
    def __init__(self, output_dir=None, cuda=False):
        """
        Initialize the CRAFT text detector.

        :param output_dir: Directory to save detection visualizations (optional)
        :param cuda: Use GPU if available
        """
        self.craft = Craft(output_dir=output_dir, crop_type="poly", cuda=cuda)
        print(f"[CRAFT] Initialized (cuda={cuda})")

    def detect_and_crop(self, image_path):
        """
        Detect text regions in the image and return cropped images sorted
        in reading order.

        :param image_path: Path to the input document image
        :return: List of dicts:
                 {
                   'crop': numpy_array (BGR),
                   'bbox': (x, y, w, h),
                   'center_y': float,
                   'center_x': float,
                   'line_idx': int
                 }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        # Run CRAFT inference
        prediction_result = self.craft.detect_text(image_path)

        boxes = prediction_result.get("boxes", [])

        text_regions = []
        for poly in boxes:
            pts = np.array(poly, dtype=np.int32)

            # Get axis-aligned bounding box
            x, y, w, h = cv2.boundingRect(pts)

            # Clamp to image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)

            # Skip tiny/empty crops
            if w < 5 or h < 5:
                continue

            crop = image[y : y + h, x : x + w]

            if crop.size > 0:
                text_regions.append(
                    {
                        "crop": crop,
                        "bbox": (x, y, w, h),
                        "center_y": y + h / 2.0,
                        "center_x": x + w / 2.0,
                    }
                )

        # Sort in reading order and assign line indices
        sorted_regions = self._sort_reading_order(text_regions, image_height=image.shape[0])

        print(f"[CRAFT] Detected {len(sorted_regions)} text regions")
        return sorted_regions

    def _sort_reading_order(self, regions, image_height, line_tolerance_ratio=0.02):
        """
        Sort regions top-to-bottom, then left-to-right.
        Groups items into lines if their center_y values are close.
        Each region gets a 'line_idx' field.

        :param regions: List of region dicts
        :param image_height: Height of the original image (used for adaptive tolerance)
        :param line_tolerance_ratio: Fraction of image height used as line grouping tolerance
        """
        if not regions:
            return []

        # Adaptive tolerance based on image size
        line_tolerance = max(15, int(image_height * line_tolerance_ratio))

        # Sort by Y first
        regions.sort(key=lambda r: r["center_y"])

        lines = []
        current_line = [regions[0]]

        for region in regions[1:]:
            if abs(region["center_y"] - current_line[-1]["center_y"]) < line_tolerance:
                current_line.append(region)
            else:
                lines.append(current_line)
                current_line = [region]
        lines.append(current_line)

        # Sort each line by X, assign line_idx
        sorted_regions = []
        for line_idx, line in enumerate(lines):
            line.sort(key=lambda r: r["center_x"])
            for region in line:
                region["line_idx"] = line_idx
                sorted_regions.append(region)

        return sorted_regions

    def unload(self):
        """Unload CRAFT models from memory."""
        self.craft.unload_craftnet_model()
        self.craft.unload_refinenet_model()


if __name__ == "__main__":
    detector = TextDetector(cuda=False)
    print("CRAFT Detector initialized successfully.")
