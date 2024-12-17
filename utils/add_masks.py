#!/usr/bin/env python
import json
import os
import argparse
import logging


def main():
    """
    Adds masks to a NerfStudio dataset (transforms.json) and saves the new transforms.json
    """
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Add masks to a NerfStudio dataset.")
    parser.add_argument("--transforms", type=str, help="Path to the input transforms.json file.")
    parser.add_argument("--masks", type=str, help="Path to the input masks directory.")
    parser.add_argument("--output", type=str, help="Path to save the output transforms.json file.")

    args = parser.parse_args()

    transforms_path = args.transforms
    masks_path = args.masks
    output_path = args.output

    if not os.path.isfile(transforms_path):
        raise FileNotFoundError(f"The input file does not exist: {transforms_path}")

    if not os.path.isdir(masks_path):
        raise FileNotFoundError(f"The input directory does not exist: {masks_path}")

    # Load the transforms.json file
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    frames = transforms['frames']
    logging.info(f"Loaded transforms from: {transforms_path}")

    # Add masks to the frames
    for frame in frames:
        frame_name = frame['file_path'].split('/')[-1]
        mask_path = os.path.join(masks_path, f"{frame_name}")
        if not os.path.isfile(mask_path):
            logging.debug(f"Mask not found for frame: {frame_name}")
        frame['mask_path'] = mask_path
    logging.info(f"Added masks from: {masks_path}")

    # Save the updated transforms.json file
    with open(output_path, 'w') as f:
        json.dump(transforms, f, indent=4)

    logging.info(f"Transforms with masks saved to: {output_path}")


if __name__ == "__main__":
    main()
