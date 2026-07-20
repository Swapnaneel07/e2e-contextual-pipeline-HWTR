"""
Visualization utilities for OCR pipeline results.

Generates side-by-side visualizations showing:
  - Original image with detected bounding boxes
  - Raw OCR text vs corrected text
"""

import textwrap
import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt


def visualize_results(image_path, text_regions, raw_text, corrected_text=None, save_path=None):
    """
    Visualize detection bounding boxes and transcription results.

    :param image_path: Path to the original input image
    :param text_regions: List of region dicts with 'bbox' key
    :param raw_text: Raw OCR output string
    :param corrected_text: LLM-corrected text string (optional)
    :param save_path: Path to save the figure (if None, displays interactively)
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Visualizer] Error: Could not read image from {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes with line-based coloring
    colors = plt.cm.Set2.colors  # Use a color palette for different lines
    for region in text_regions:
        x, y, w, h = region["bbox"]
        line_idx = region.get("line_idx", 0)
        color_rgb = colors[line_idx % len(colors)]
        color_bgr = tuple(int(c * 255) for c in color_rgb[:3])

        cv2.rectangle(image_rgb, (x, y), (x + w, y + h), color_bgr, 2)

    # Build figure
    has_correction = corrected_text is not None and corrected_text != raw_text
    fig = plt.figure(figsize=(18, 10))

    # Left: Image with bounding boxes
    ax_img = fig.add_subplot(1, 2, 1)
    ax_img.imshow(image_rgb)
    ax_img.set_title("Detected Text Regions", fontsize=14, fontweight="bold")
    ax_img.axis("off")

    # Right: Text results
    ax_txt = fig.add_subplot(1, 2, 2)
    ax_txt.axis("off")

    y_pos = 0.95
    line_height = 0.04
    wrap_width = 70

    # Raw OCR text
    ax_txt.text(0.02, y_pos, "Raw OCR Output:", fontsize=13, fontweight="bold",
                color="#1565C0", transform=ax_txt.transAxes, verticalalignment="top")
    y_pos -= line_height * 1.5

    raw_wrapped = textwrap.fill(raw_text, width=wrap_width) if raw_text else "(no text detected)"
    for line in raw_wrapped.split("\n"):
        ax_txt.text(0.02, y_pos, line, fontsize=10, transform=ax_txt.transAxes,
                    verticalalignment="top", fontfamily="monospace")
        y_pos -= line_height

    # Corrected text
    if has_correction:
        y_pos -= line_height
        ax_txt.text(0.02, y_pos, "LLM Corrected Output:", fontsize=13, fontweight="bold",
                    color="#2E7D32", transform=ax_txt.transAxes, verticalalignment="top")
        y_pos -= line_height * 1.5

        cor_wrapped = textwrap.fill(corrected_text, width=wrap_width)
        for line in cor_wrapped.split("\n"):
            ax_txt.text(0.02, y_pos, line, fontsize=10, transform=ax_txt.transAxes,
                        verticalalignment="top", fontfamily="monospace")
            y_pos -= line_height

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Visualizer] Saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    print("Visualizer module loaded. Use visualize_results() to generate output.")
