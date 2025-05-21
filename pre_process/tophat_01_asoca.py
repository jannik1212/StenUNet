#!/usr/bin/env python3
import os
import nrrd
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import white_tophat, ball
import argparse

def tv_tophat(vol, tv_weight=0.01, se_radius=5):
    """
    Perform TV denoising followed by white‐tophat filtering.
    """
    # total‐variation denoise
    tv = denoise_tv_chambolle(vol, weight=tv_weight)
    # white top‐hat with a spherical structuring element
    se = ball(se_radius)
    wt = white_tophat(tv, footprint=se)
    # re‐normalize to [0,1]
    wt = wt - wt.min()
    wt = wt / (wt.max() + 1e-8)
    return wt

def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith('.nrrd'):
            continue

        # 1) Load
        input_path = os.path.join(input_dir, fname)
        vol, hdr = nrrd.read(input_path)
        vol = vol.astype(np.float32)

        # 2) Normalize
        vol_norm = (vol - vol.min()) / (vol.max() - vol.min())

        # 3) TV + white‐tophat
        processed = tv_tophat(vol_norm, tv_weight=0.01, se_radius=5)

        # 4) Build new filename (replace last underscore part with "0001")
        base, ext = os.path.splitext(fname)
        parts = base.split('_')
        parts[-1] = '0001'
        out_name = '_'.join(parts) + ext

        # 5) Save
        output_path = os.path.join(output_dir, out_name)
        nrrd.write(output_path, processed.astype(np.float32), hdr)
        print(f"Saved preprocessed volume: {out_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Batch TV+white-tophat preprocessing of NRRD volumes"
    )
    parser.add_argument('input_dir',
                        help='Folder containing input .nrrd files (e.g. sten_0000_0000.nrrd)')
    parser.add_argument('output_dir',
                        help='Folder to save processed files (will be created if needed)')
    args = parser.parse_args()
    process_folder(args.input_dir, args.output_dir)
