#!/usr/bin/env python3
import os
import glob
import shutil
import argparse

def collect_missing(raw_dir, processed_dir, missing_dir):
    """
    Copy into `missing_dir` any file X in raw_dir for which there is
    no corresponding processed file in processed_dir.
    Raw files are named like sten_AAAA_BBBB.ext; processed like sten_AAAA_CCCC.ext.
    We match on the first two fields AAAA (so drop the last underscore-part).
    """
    os.makedirs(missing_dir, exist_ok=True)

    # 1) Gather processed IDs: basename without the final _XXXX part
    proc_pattern = os.path.join(processed_dir, 'sten_*_*.*')
    proc_files = glob.glob(proc_pattern)
    proc_ids = set()
    for p in proc_files:
        name = os.path.basename(p)
        base, _ = os.path.splitext(name)
        # split off the last "_NNNN" and rejoin
        prefix = base.rsplit('_', 1)[0]
        proc_ids.add(prefix)

    print(f"Found {len(proc_ids)} unique processed IDs in {processed_dir}")

    # 2) Walk raw directory
    raw_pattern = os.path.join(raw_dir, 'sten_*_*.*')
    raw_files = glob.glob(raw_pattern)
    print(f"Found {len(raw_files)} raw files in {raw_dir}")

    missing = 0
    for r in raw_files:
        name = os.path.basename(r)
        base, ext = os.path.splitext(name)
        prefix = base.rsplit('_', 1)[0]
        if prefix not in proc_ids:
            # copy the raw file into missing_dir
            shutil.copy(r, os.path.join(missing_dir, name))
            missing += 1

    if missing:
        print(f"Copied {missing} missing raw files into {missing_dir}")
    else:
        print("✔ All raw files have a corresponding processed file.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Collect raw images that lack a processed counterpart"
    )
    parser.add_argument('raw_dir',
                        help='Folder of all raw images (e.g. imagesTr)')
    parser.add_argument('processed_dir',
                        help='Folder of Frangi outputs (e.g. frangi_masks)')
    parser.add_argument('missing_dir',
                        help='Where to copy the missing raw images')
    args = parser.parse_args()

    collect_missing(args.raw_dir, args.processed_dir, args.missing_dir)
