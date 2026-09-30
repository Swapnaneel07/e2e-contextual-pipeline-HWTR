"""
GNHK Dataset Loader

Loads ground truth annotations from the GNHK handwriting dataset.
Each image has a corresponding JSON file with word-level annotations
including text, polygon coordinates, line index, and type (H=handwritten, P=printed).
"""

import os
import json
import glob
import numpy as np


def load_ground_truth(json_path, include_printed=True, skip_special=True):
    """
    Load word annotations from a GNHK JSON file and reconstruct
    the full page text in reading order.

    :param json_path: Path to the GNHK annotation JSON file
    :param include_printed: If True, include printed text (type="P") in ground truth
    :param skip_special: If True, skip special tokens like %math%, %drawing%, etc.
    :return: dict with keys:
             'full_text': reconstructed page text (str),
             'words': list of word dicts,
             'lines': dict mapping line_idx -> list of words
    """
    with open(json_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    # Filter annotations
    filtered = []
    for ann in annotations:
        text = ann.get("text", "")
        ann_type = ann.get("type", "H")

        # Skip special tokens
        if skip_special and text.startswith("%") and text.endswith("%"):
            continue

        # Skip printed text if not wanted
        if not include_printed and ann_type == "P":
            continue

        # Skip empty text
        if not text.strip():
            continue

        filtered.append(ann)

    if not filtered:
        return {"full_text": "", "words": [], "lines": {}}

    # Group by line_idx and sort within each line by x position
    lines = {}
    for ann in filtered:
        line_idx = ann.get("line_idx", 0)
        if line_idx not in lines:
            lines[line_idx] = []

        # Get x position from polygon for sorting
        x_pos = ann["polygon"].get("x0", 0)
        lines[line_idx].append({
            "text": ann["text"],
            "x_pos": x_pos,
            "type": ann.get("type", "H"),
            "polygon": ann["polygon"],
            "line_idx": line_idx,
        })

    # Sort lines by line_idx, words within each line by x position
    sorted_line_indices = sorted(lines.keys())
    text_lines = []
    for line_idx in sorted_line_indices:
        words = sorted(lines[line_idx], key=lambda w: w["x_pos"])
        line_text = " ".join(w["text"] for w in words)
        text_lines.append(line_text)

    full_text = "\n".join(text_lines)

    return {
        "full_text": full_text,
        "words": filtered,
        "lines": lines,
    }


def list_dataset_samples(dataset_dir):
    """
    List all image-JSON pairs in a GNHK dataset directory.

    :param dataset_dir: Path to directory containing .jpg and .json files
    :return: List of dicts with 'image_path', 'json_path', 'sample_id'
    """
    samples = []
    image_files = sorted(glob.glob(os.path.join(dataset_dir, "*.jpg")))

    for img_path in image_files:
        json_path = img_path.replace(".jpg", ".json")
        if os.path.exists(json_path):
            sample_id = os.path.splitext(os.path.basename(img_path))[0]
            samples.append({
                "image_path": img_path,
                "json_path": json_path,
                "sample_id": sample_id,
            })

    return samples


if __name__ == "__main__":
    # Quick test
    test_dir = "input/gnhk_dataset/test_data/test"
    samples = list_dataset_samples(test_dir)
    print(f"Found {len(samples)} samples in {test_dir}")

    if samples:
        sample = samples[0]
        print(f"\nSample: {sample['sample_id']}")
        gt = load_ground_truth(sample["json_path"])
        print(f"Ground truth ({len(gt['words'])} words, {len(gt['lines'])} lines):")
        print("-" * 40)
        print(gt["full_text"][:500])
