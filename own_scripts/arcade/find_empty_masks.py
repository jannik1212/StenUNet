import os
import numpy as np # type: ignore
from PIL import Image # type: ignore

def find_empty_label_masks(label_dir):
    empty_labels = []

    for fname in os.listdir(label_dir):
        if fname.endswith(".png"):
            path = os.path.join(label_dir, fname)
            mask = np.array(Image.open(path))

            # Convert to binary if necessary
            mask = (mask > 0).astype(np.uint8)

            if np.sum(mask) == 0:
                empty_labels.append(fname)

    return empty_labels

if __name__ == "__main__":
    # Adjust this path to point to your label folder
    label_dir = "../nnNet_training/Raw_data/Dataset_Train_val/labelsTr"

    print("🔍 Scanning for empty masks (only background)...")
    empty_files = find_empty_label_masks(label_dir)

    if empty_files:
        print(f"\n⚠️ Found {len(empty_files)} empty label mask(s):")
        for f in empty_files:
            print("   -", f)

        # Optional: prompt to delete
        confirm = input("\n❓ Do you want to delete these files now? [y/N]: ").lower()
        if confirm == 'y':
            for f in empty_files:
                os.remove(os.path.join(label_dir, f))
                print(f"🗑️ Deleted: {f}")
            print("✅ All empty masks deleted.")
        else:
            print("🚫 No files deleted.")
    else:
        print("✅ No empty label masks found!")
