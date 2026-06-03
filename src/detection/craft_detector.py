import os
import cv2
import numpy as np
from craft_text_detector import Craft

class TextDetector:
    def __init__(self, output_dir=None, cuda=False):
        """
        Initialize the CRAFT text detector.
        :param output_dir: Directory to save detection results (optional)
        :param cuda: Use GPU if available
        """
        self.craft = Craft(output_dir=output_dir, crop_type="poly", cuda=cuda)

    def detect_and_crop(self, image_path):
        """
        Detect text regions in the image and return cropped images sorted in reading order.
        :param image_path: Path to the input image
        :return: List of dictionaries containing {'crop': numpy_array, 'bbox': [x, y, w, h]}
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Read image to get crops manually since craft-text-detector's crop saving 
        # might just save to disk, but we want them in memory.
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        
        # Run inference
        prediction_result = self.craft.detect_text(image_path)
        
        # prediction_result contains 'boxes' (polygons), 'polys', 'heatmap'
        # craft-text-detector returns a dictionary with 'boxes' containing numpy arrays of shape (4, 2)
        boxes = prediction_result.get("boxes", [])
        
        text_regions = []
        for poly in boxes:
            # poly is a 4x2 array of points (top-left, top-right, bottom-right, bottom-left)
            pts = np.array(poly, dtype=np.int32)
            
            # Get axis-aligned bounding box
            x, y, w, h = cv2.boundingRect(pts)
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            # Extract crop
            crop = image[y:y+h, x:x+w]
            
            if crop.size > 0:
                text_regions.append({
                    "crop": crop,
                    "bbox": (x, y, w, h),
                    "center_y": y + h / 2,
                    "center_x": x + w / 2
                })
        
        # Sort text regions in reading order (top-to-bottom, left-to-right)
        # We can group by lines based on y-coordinate proximity
        sorted_regions = self._sort_reading_order(text_regions)
        
        return sorted_regions

    def _sort_reading_order(self, regions, line_tolerance=15):
        """
        Sort regions top-to-bottom, then left-to-right.
        Groups items into lines if their center_y is within line_tolerance.
        """
        if not regions:
            return []
            
        # First sort strictly by Y
        regions.sort(key=lambda r: r['center_y'])
        
        lines = []
        current_line = [regions[0]]
        
        for region in regions[1:]:
            # If the y difference is small, consider it same line
            if abs(region['center_y'] - current_line[-1]['center_y']) < line_tolerance:
                current_line.append(region)
            else:
                lines.append(current_line)
                current_line = [region]
        lines.append(current_line)
        
        # Sort each line by X coordinate
        sorted_regions = []
        for line in lines:
            line.sort(key=lambda r: r['center_x'])
            sorted_regions.extend(line)
            
        return sorted_regions

    def unload(self):
        """Unload the model from memory."""
        self.craft.unload_craftnet_model()
        self.craft.unload_refinenet_model()

if __name__ == "__main__":
    # Simple test
    detector = TextDetector(cuda=False)
    print("CRAFT Detector initialized successfully.")
