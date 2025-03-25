import os
import json
import numpy as np # type: ignore
import cv2 # type: ignore
import argparse
from tqdm import tqdm #type: ignore

def load_annotations(annotations_path):
    with open(annotations_path, "r") as f:
        annotations = json.load(f)
    return annotations

def create_mask(image_shape, segmentation, category_id, category_map):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    if not segmentation:
        return mask  # No segmentation found, return empty mask
    
    for segment in segmentation:
        if isinstance(segment, list) and len(segment) >= 6:  # Ensure valid segmentation format
            pts = np.array(segment, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(mask, [pts], color=category_map[category_id])
    
    return mask

def process_annotations(data_root, annotations_path, output_root, task):
    os.makedirs(output_root, exist_ok=True)
    annotations = load_annotations(annotations_path)

    # Map image IDs to filenames
    image_map = {img["id"]: img["file_name"] for img in annotations["images"]}

    # Determine category mapping
    if task == "stenosis":
        category_map = {26: 255}  # Only stenosis, binary mask (white = 255)
    else:
        category_map = {cat["id"]: cat["id"] for cat in annotations["categories"]}  # Multiclass

    print(f"Loaded {len(image_map)} images, {len(annotations['annotations'])} annotations.")

    for image_id, file_name in tqdm(image_map.items(), desc="Processing images"):
        image_path = os.path.join(data_root, file_name)
        if not os.path.exists(image_path):
            print(f"WARNING: Image file not found: {file_name}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"WARNING: Failed to read image: {file_name}")
            continue

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        found_annotation = False

        for ann in annotations["annotations"]:
            # Check if annotation should be processed
            if ann["image_id"] == image_id and (task == "multiclass" or ann["category_id"] == 26):
                found_annotation = True
                mask += create_mask(image.shape, ann["segmentation"], ann["category_id"], category_map)

        mask_path = os.path.join(output_root, file_name)
        cv2.imwrite(mask_path, mask)

        if not found_annotation:
            print(f"No valid annotations for `{file_name}`, saved empty mask.")

    print(f"Mask generation complete! Masks saved in `{output_root}`.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Path to the raw images directory")
    parser.add_argument("--annotations_path", required=True, help="Path to the annotations JSON file")
    parser.add_argument("--output_root", required=True, help="Path to save the output masks")
    parser.add_argument("--task", required=True, choices=["stenosis", "multiclass"], help="Type of mask generation")
    args = parser.parse_args()

    process_annotations(args.data_root, args.annotations_path, args.output_root, args.task)