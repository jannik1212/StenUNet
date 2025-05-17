import os
import numpy as np # type: ignore
from PIL import Image # type: ignore

def convert_labels_to_binary(labels_dir):
    """Converts label images from {0, 255} to {0, 1} (uint8)."""
    
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".png")]
    
    for label_file in label_files:
        path = os.path.join(labels_dir, label_file)
        mask = np.array(Image.open(path))
        
        # Convert 255 to 1
        mask = (mask > 0).astype(np.uint8)
        
        Image.fromarray(mask).save(path)
        print(f"Converted: {label_file} to binary (0/1)")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    labels_dir = os.path.join(base_dir, "../../data/labelsTr")
    
    
    print("Converting label masks to binary format...")
    convert_labels_to_binary(labels_dir)
    print("✅ All label masks converted!")
