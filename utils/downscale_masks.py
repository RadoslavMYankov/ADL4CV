from nerfstudio.process_data import process_data_utils
import os
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downscale density masks for rejection sampling. Filter out all dense regions in an image"
    )

    parser.add_argument(
        "--masks_path",
        type=str,
        required=True,
        help="Path to the input directory containing the density masks."
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the downsampled density masks."
    )

    parser.add_argument(
        "--num_downscales",
        type=int,
        default=3,
        help="Number of times to downscale the density masks."
    )

    args = parser.parse_args()
    masks_path = Path(args.masks_path)
    img_list = os.listdir(masks_path)
    img_dir = Path(args.output_path)
    img_paths = [masks_path / img for img in img_list]

    process_data_utils.copy_images_list(img_paths, image_dir=img_dir, num_downscales=args.num_downscales, keep_image_dir=True)
