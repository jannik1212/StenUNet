import os
import glob
import numpy as np
from skimage import io, exposure
from skimage.filters import frangi

# ---- Configuration ----
# Adjust this to your dataset folder path
input_dir = 'own_scripts/arcade/frangi_masks'
# Output folder for Frangi masks (for label 0001)
output_dir = 'pre_process/frangi_masks_real'
# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Frangi filter parameters
sigmas = np.arange(6, 85, 1)
alpha = 1.0
beta  = 1.0
gamma = 0.1
black_ridges = True
mode = 'reflect'

# Process each file matching sten_*_0000.*
pattern = os.path.join(input_dir, 'sten_*_0000.*')
for filepath in glob.glob(pattern):
    # Parse filename and construct new filename with _0001 suffix
    dirname, filename = os.path.split(filepath)
    base, ext = os.path.splitext(filename)
    parts = base.split('_')  # ['sten', '0001', '0000']
    parts[-1] = '0001'       # change channel suffix
    new_base = '_'.join(parts)
    new_filename = new_base + ext
    out_path = os.path.join(output_dir, new_filename)

    # Read and normalize the image to [0,1]
    I = io.imread(filepath, as_gray=True).astype(np.float32)
    I = exposure.rescale_intensity(I, out_range=(0, 1))

    # Compute the Frangi vesselness response
    vesselness = frangi(
        I,
        sigmas=sigmas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
        mode=mode
    )

    # Normalize the response to [0,255] uint8 for saving
    v_norm = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min())
    v_uint8 = (v_norm * 255).astype(np.uint8)

    # Save the mask
    io.imsave(out_path, v_uint8)
    print(f"Saved Frangi mask: {out_path}")
