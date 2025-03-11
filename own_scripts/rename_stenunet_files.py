import os

def rename_files(images_dir, masks_dir):
    """Renames image and mask files to match the required format for StenUNet."""
    # Get and sort all image files numerically
    image_files = sorted(os.listdir(images_dir), key=lambda x: int(x.split('.')[0]))
    mask_files = sorted(os.listdir(masks_dir), key=lambda x: int(x.split('_')[0]))
    
    for idx, image_file in enumerate(image_files):
        old_image_path = os.path.join(images_dir, image_file)
        new_image_name = f"sten_{idx:04d}_0000.png"
        new_image_path = os.path.join(images_dir, new_image_name)
        os.rename(old_image_path, new_image_path)
        
        print(f"Renamed image: {image_file} -> {new_image_name}")
    
    for idx, mask_file in enumerate(mask_files):
        old_mask_path = os.path.join(masks_dir, mask_file)
        new_mask_name = f"sten_{idx:04d}.png"
        new_mask_path = os.path.join(masks_dir, new_mask_name)
        os.rename(old_mask_path, new_mask_path)
        
        print(f"Renamed mask: {mask_file} -> {new_mask_name}")

if __name__ == "__main__":
    images_dir = "../nnNet_training/Raw_data/Dataset_Train_val/imagesTr"
    masks_dir = "../nnNet_training/Raw_data/Dataset_Train_val/labelsTr"
    rename_files(images_dir, masks_dir)
