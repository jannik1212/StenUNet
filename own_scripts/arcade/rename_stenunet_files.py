import os
import argparse

def rename_files(images_dir, masks_dir=None, method=0):
    """
    Renames images and masks according to the specified preprocessing method.
    
    - Images: sten_XXXX_YYYY.png (where YYYY = preprocessing method)
    - Masks: sten_XXXX.png (no preprocessing method)
    
    If a file is already renamed, it is skipped.
    """

    # Get and sort image files, ignoring already renamed files
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.startswith("sten_")],
        key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf')  # Sort numerically
    )

    print(f"Found {len(image_files)} images to rename in {images_dir}")

    for idx, image_file in enumerate(image_files):
        old_image_path = os.path.join(images_dir, image_file)
        new_image_name = f"sten_{idx:04d}_{method:04d}.png"
        new_image_path = os.path.join(images_dir, new_image_name)
        
        os.rename(old_image_path, new_image_path)
        print(f"Renamed image: {image_file} -> {new_image_name}")

    # Process masks if directory exists
    if masks_dir and os.path.exists(masks_dir):
        mask_files = sorted(
            [f for f in os.listdir(masks_dir) if f.lower().endswith(".png") and not f.startswith("sten_")],
            key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf')
        )

        print(f"Found {len(mask_files)} masks to rename in {masks_dir}")

        for idx, mask_file in enumerate(mask_files):
            old_mask_path = os.path.join(masks_dir, mask_file)
            new_mask_name = f"sten_{idx:04d}.png"
            new_mask_path = os.path.join(masks_dir, new_mask_name)

            os.rename(old_mask_path, new_mask_path)
            print(f"Renamed mask: {mask_file} -> {new_mask_name}")
    else:
        print(f"Masks directory '{masks_dir}' not found or empty!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True, help="Path to the images directory")
    parser.add_argument("--masks_dir", default=None, help="Path to the masks directory (optional)")
    parser.add_argument("--method", type=int, default=0, help="Preprocessing method index (default: 0)")
    
    args = parser.parse_args()
    
    rename_files(args.images_dir, args.masks_dir, args.method)
