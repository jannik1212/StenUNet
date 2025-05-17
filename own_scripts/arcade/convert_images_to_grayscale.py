import os
from PIL import Image #type: ignore

def convert_images_to_grayscale(images_dir):
    """Converts all images in the directory to grayscale (L mode)."""
    
    # Get all image files in the directory
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".png")]
    
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        
        # Open and convert to grayscale
        img = Image.open(image_path).convert("L")
        img.save(image_path)  # Overwrite original file
        
        print(f"Converted: {image_file} to grayscale.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, "../../data/imagesTr")
    
    print("Converting images to grayscale...")
    convert_images_to_grayscale(images_dir)
    
    print("All images converted to grayscale!")
