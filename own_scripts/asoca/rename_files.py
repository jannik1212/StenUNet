import os
import argparse
import random
import shutil

def collect_nrrd_pairs(image_dirs, mask_dirs):
    all_pairs = []
    for image_dir, mask_dir in zip(image_dirs, mask_dirs):
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nrrd") and not f.startswith("sten_")])
        mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".nrrd") and not f.startswith("sten_")])

        assert len(image_files) == len(mask_files), f"Mismatch in {image_dir} and {mask_dir}"

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
        new_img_name = f"sten_{idx:04d}_{method:04d}.nrrd"
        new_mask_name = f"sten_{idx:04d}.nrrd"

        # Decide output path (copy) or rename in-place
        if output_images_dir:
            new_img_path = os.path.join(output_images_dir, new_img_name)
            shutil.copy2(img_path, new_img_path)
        else:
            new_img_path = os.path.join(os.path.dirname(img_path), new_img_name)
            os.rename(img_path, new_img_path)

        if output_masks_dir:
            new_mask_path = os.path.join(output_masks_dir, new_mask_name)
            shutil.copy2(mask_path, new_mask_path)
        else:
            new_mask_path = os.path.join(os.path.dirname(mask_path), new_mask_name)
            os.rename(mask_path, new_mask_path)

        print(f"✔️  Image: {os.path.basename(img_path)} → {new_img_name}")
        print(f"✔️  Mask : {os.path.basename(mask_path)} → {new_mask_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dirs", nargs='+', required=True, help="List of image directories")
    parser.add_argument("--mask_dirs", nargs='+', required=True, help="List of mask directories")
    parser.add_argument("--method", type=int, default=0, help="Preprocessing method index (default: 0)")
    parser.add_argument("--output_images_dir", default=None, help="Directory to copy renamed images into")
    parser.add_argument("--output_masks_dir", default=None, help="Directory to copy renamed masks into")
    args = parser.parse_args()

    print("📦 Collecting image–mask pairs...")
    pairs = collect_nrrd_pairs(args.image_dirs, args.mask_dirs)
    print(f"✅ Collected {len(pairs)} total image–mask pairs")

    print("\n🔀 Shuffling and renaming...")
    shuffle_and_rename(pairs, args.method, args.output_images_dir, args.output_masks_dir)
