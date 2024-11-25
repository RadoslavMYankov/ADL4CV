#!/usr/bin/env python3
import open3d as o3d
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Fix the alignment of a PLY file.")
    parser.add_argument("input_path", type=str, help="Path to the input PLY file.")
    args = parser.parse_args()

    input_path = args.input_path
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' does not exist.")
        return

    pcd = o3d.io.read_point_cloud(input_path)

    rotation_matrix = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])

    pcd.rotate(rotation_matrix, center=(0, 0, 0))

    output_path = f"{os.path.splitext(input_path)[0]}_fixed.ply"
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Fixed point cloud saved to '{output_path}'.")

if __name__ == "__main__":
    main()
