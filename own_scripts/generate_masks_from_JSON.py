import json
import os
import argparse
from PIL import Image, ImageDraw


def generate_all_masks(data_root="../dataset_test/raw", output_root="../dataset_test/raw_masks", task=None):
    """Generate segmentation masks from JSON annotations.

    - If `stenosis/` and `syntax/` folders exist, applies correct segmentation automatically.
    - If the dataset is flat (all images in one folder), the user must specify `stenosis` or `syntax`.
    - Detects available annotation files (`train.json`, `val.json`, `test.json`).
    """

    structured_mode = False  # Assume flat dataset
    detected_splits = []  # Stores detected annotation files (train, val, test)

    # Find annotation files inside `dataset_test/annotations/`
    annotations_dir = os.path.join(os.path.dirname(data_root), "annotations")
    if os.path.exists(annotations_dir):
        for possible_split in ["train", "val", "test"]:
            json_file = os.path.join(annotations_dir, f"{possible_split}.json")
            if os.path.exists(json_file):
                detected_splits.append((possible_split, json_file))  # Store split name + full path

    # Detect tasks if multiple exist (structured case)
    detected_tasks = [t for t in ["stenosis", "syntax"] if os.path.isdir(os.path.join(data_root, t))]

    if detected_tasks:
        structured_mode = True
        if len(detected_tasks) == 1 and not task:
            task = detected_tasks[0]  # If only one task exists, use it automatically
        elif len(detected_tasks) > 1 and not task:
            print(f"🟡 Detected both `stenosis/` and `syntax/`. Processing both automatically.")
        else:
            print(f"🔹 Manually specified task: {task}")

    elif not task:
        print(f"❌ ERROR: No tasks detected and no task specified. Please use `--task stenosis` or `--task syntax`.")
        return

    # If no annotation files were found, error out
    if not detected_splits:
        print(f"❌ ERROR: No annotation files found in `{annotations_dir}`. Expected `train.json`, `val.json`, or `test.json`.")
        return

    tasks_to_process = detected_tasks if structured_mode else [task]

    for current_task in tasks_to_process:
        for split, annotation_file in detected_splits:
            if structured_mode:
                output_path = os.path.join(output_root, current_task, split)
                images_path = os.path.join(data_root, current_task, split, "images")
            else:
                output_path = os.path.join(output_root, split)  # Separate masks per split
                images_path = data_root

            os.makedirs(output_path, exist_ok=True)

            if not os.path.exists(annotation_file):
                print(f"Skipping {annotation_file}, file not found.")
                continue

            with open(annotation_file, "r") as f:
                annotations = json.load(f)

            # Extract images and categories
            images = {img["id"]: img for img in annotations["images"]}
            categories = {cat["id"]: cat["name"] for cat in annotations["categories"]}
            annotations = annotations["annotations"]

            print(f"Processing {current_task}/{split}: {len(images)} images, {len(annotations)} annotations")

            for image_id, image_info in images.items():
                img_filename = image_info["file_name"]
                img_width, img_height = image_info["width"], image_info["height"]

                # Create an empty mask
                mask = Image.new("L", (img_width, img_height), 0)
                draw = ImageDraw.Draw(mask)

                annotations_found = False

                for ann in annotations:
                    if ann["image_id"] != image_id:
                        continue

                    category_id = ann["category_id"]
                    segmentation = ann["segmentation"]

                    if not segmentation:
                        print(f"[WARNING] Empty segmentation for image {img_filename} (Category: {category_id})")
                        continue

                    if current_task == "syntax":  # Multi-class segmentation
                        if category_id in categories:
                            annotations_found = True
                            for polygon in segmentation:
                                if len(polygon) < 6:
                                    print(f"[WARNING] Invalid polygon in {img_filename}, skipping...")
                                    continue
                                polygon = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                                draw.polygon(polygon, outline=category_id, fill=category_id)
                    elif current_task == "stenosis":  # Binary segmentation
                        if categories.get(category_id, "").lower() == "stenosis":
                            annotations_found = True
                            for polygon in segmentation:
                                if len(polygon) < 6:
                                    print(f"[WARNING] Invalid stenosis polygon in {img_filename}, skipping...")
                                    continue
                                polygon = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                                draw.polygon(polygon, outline=1, fill=1)

                if not annotations_found:
                    print(f"[INFO] No valid annotations found for {img_filename}, mask will be empty.")

                # Save mask
                mask_path = os.path.join(output_path, f"{img_filename.replace('.png', '_mask.png')}")
                mask.save(mask_path)

    print(f"✅ Mask generation complete! Masks saved in `{output_root}` (Task: {task if not structured_mode else 'Multiple'})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../dataset_test/raw", help="Path to the raw dataset folder")
    parser.add_argument("--output_root", type=str, default="../dataset_test/raw_masks", help="Path to store generated masks")
    parser.add_argument("--task", type=str, choices=["stenosis", "syntax"], help="Specify task (stenosis or syntax) if only one is present")

    args = parser.parse_args()
    generate_all_masks(args.data_root, args.output_root, args.task)
