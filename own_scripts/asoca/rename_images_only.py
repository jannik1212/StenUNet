import os
import argparse
import random

def rename_nrrd_images(image_dir, method=0):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".nrrd")])
    random.shuffle(image_files)

    for idx, filename in enumerate(image_files):
        old_path = os.path.join(image_dir, filename)
        new_name = f"sten_{idx:04d}_{method:04d}.nrrd"
        new_path = os.path.join(image_dir, new_name)

        os.rename(old_path, new_path)
        print(f"✔️ {filename} → {new_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True, help="Directory containing .nrrd images to rename")
    parser.add_argument("--method", type=int, default=0, help="Method number to embed in filename")
    args = parser.parse_args()

    rename_nrrd_images(args.image_dir, args.method)