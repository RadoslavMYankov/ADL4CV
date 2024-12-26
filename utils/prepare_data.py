import argparse
from pathlib import Path
from nerfstudio.process_data import colmap_utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert COLMAP sparse reconstruction to a Nerfstudio JSON file.")
    parser.add_argument("input-path", type=str, help="Path to the COLMAP sparse reconstruction.", default='data/alameda/colmap/sparse/0')
    parser.add_argument("output-path", type=str, help="Path to the output JSON file.", default='colmap_to_nerfstudio')
    args = parser.parse_args()

    bin_path = Path(args.input_path)
    output_path = Path(args.output_path)

    colmap_utils.colmap_to_json(bin_path, output_path)
