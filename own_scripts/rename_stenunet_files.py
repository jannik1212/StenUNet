import os

def rename_files(images_dir, masks_dir=None):
    """Renames images and optionally masks, handling cases where masks are missing."""
    
    # Get and sort all image files
    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
        key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else float('inf')  # Avoid errors with non-numeric names
    )

    for idx, image_file in enumerate(image_files):
        old_image_path = os.path.join(images_dir, image_file)
        new_image_name = f"sten_{idx:04d}_0000.png"
        new_image_path = os.path.join(images_dir, new_image_name)
        
        os.rename(old_image_path, new_image_path)
        print(f"Renamed image: {image_file} -> {new_image_name}")

    # Only process masks if the directory exists
    if masks_dir and os.path.exists(masks_dir):
        mask_files = sorted(
            [f for f in os.listdir(masks_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
            key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else float('inf')
        )

        for idx, mask_file in enumerate(mask_files):
            old_mask_path = os.path.join(masks_dir, mask_file)
            new_mask_name = f"sten_{idx:04d}.png"
            new_mask_path = os.path.join(masks_dir, new_mask_name)
            os.rename(old_mask_path, new_mask_path)

            print(f"Renamed mask: {mask_file} -> {new_mask_name}")

if __name__ == "__main__":
    images_dir = "../dataset_test/raw"
    masks_dir = "../nnNet_training/Raw_data/Dataset_Train_val/labelsTr"  # Can be None for test sets

    rename_files(images_dir, masks_dir)
