import os
import argparse
import random
import shutil

def collect_nii_pairs(image_dirs, mask_dirs):
    all_pairs = []
    for image_dir, mask_dir in zip(image_dirs, mask_dirs):
        # pick up only .nii.gz files, skip any already renamed ones
        image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".nii.gz") and not f.startswith("sten_")
        ])
        mask_files = sorted([
            f for f in os.listdir(mask_dir)
            if f.endswith(".nii.gz") and not f.startswith("sten_")
        ])

        assert len(image_files) == len(mask_files), \
            f"Mismatch in {image_dir} ({len(image_files)}) vs {mask_dir} ({len(mask_files)})"

        for img_file, mask_file in zip(image_files, mask_files):
            img_path = os.path.join(image_dir, img_file)
            mask_path = os.path.join(mask_dir, mask_file)
            all_pairs.append((img_path, mask_path))

    return all_pairs

def shuffle_and_rename(pairs, method=0, output_images_dir=None, output_masks_dir=None):
    random.shuffle(pairs)

    if output_images_dir:
        os.makedirs(output_images_dir, exist_ok=True)
    if output_masks_dir:
        os.makedirs(output_masks_dir, exist_ok=True)

    for idx, (img_path, mask_path) in enumerate(pairs):
        # new names with .nii.gz
        new_img_name  = f"sten_{idx:04d}_{method:04d}.nii.gz"
        new_mask_name = f"sten_{idx:04d}.nii.gz"

        # images
        if output_images_dir:
            dst_img = os.path.join(output_images_dir, new_img_name)
            shutil.copy2(img_path, dst_img)
        else:
            dst_img = os.path.join(os.path.dirname(img_path), new_img_name)
            os.rename(img_path, dst_img)

        # masks
        if output_masks_dir:
            dst_msk = os.path.join(output_masks_dir, new_mask_name)
            shutil.copy2(mask_path, dst_msk)
        else:
            dst_msk = os.path.join(os.path.dirname(mask_path), new_mask_name)
            os.rename(mask_path, dst_msk)

        print(f"Image: {os.path.basename(img_path)} → {new_img_name}")
        print(f"Mask : {os.path.basename(mask_path)} → {new_mask_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Shuffle & rename .nii.gz image–mask pairs for StenUNet."
    )
    parser.add_argument(
        "--image_dirs", nargs='+', required=True,
        help="List of directories containing original .nii.gz images"
    )
    parser.add_argument(
        "--mask_dirs", nargs='+', required=True,
        help="List of directories containing corresponding .nii.gz masks"
    )
    parser.add_argument(
        "--method", type=int, default=0,
        help="Numeric suffix to distinguish preprocessing variants (default: 0)"
    )
    parser.add_argument(
        "--output_images_dir", default=None,
        help="If set, copy renamed images here (instead of in-place rename)"
    )
    parser.add_argument(
        "--output_masks_dir", default=None,
        help="If set, copy renamed masks here (instead of in-place rename)"
    )
    args = parser.parse_args()

    print("Collecting image–mask pairs...")
    pairs = collect_nii_pairs(args.image_dirs, args.mask_dirs)
    print(f"Collected {len(pairs)} total image–mask pairs\n")

    print("🔀 Shuffling and renaming...")
    shuffle_and_rename(
        pairs,
        method          = args.method,
        output_images_dir = args.output_images_dir,
        output_masks_dir  = args.output_masks_dir
    )
    print("\nDone renaming!")
