# Merge ply files to generate a single point cloud

import open3d as o3d
import numpy as np
from pathlib import Path
import os
import argparse

def nerf_cs_to_colmap(nerf_pcd):
    applied_transform = np.array([
        [1.0, 0.0,  0.0,  0.0],
        [0.0, 0.0,  1.0,  0.0],
        [-0.0, -1.0, -0.0, -0.0],
        [0.0, 0.0,  0.0,  1.0]
    ], dtype=np.float64)

    nerf_pcd.transform(applied_transform)
    return nerf_pcd

def merge_pcs(sfm_input, nerf_inputs):

    # Read the SfM point cloud
    merged_pc = o3d.io.read_point_cloud(sfm_input)

    for pc_path in Path(nerf_inputs).rglob("*.ply"):
        pc_path = str(pc_path)
        # Read the NeRF point cloud
        nerf_pc = o3d.io.read_point_cloud(pc_path)
        # Convert the NeRF point cloud to the COLMAP coordinate system
        nerf_pc = nerf_cs_to_colmap(nerf_pc)
        merged_pc += nerf_pc

    return merged_pc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge point clouds")
    parser.add_argument("--sfm_input", type=str, help="Path to the SfM pointcloud.")
    parser.add_argument("--nerf_inputs", type=str, help="Path to the NeRF point clouds.")
    parser.add_argument("--output", type=str, help="Path to save the output point cloud.")
    args = parser.parse_args()

    if not args.sfm_input or not os.path.exists(args.sfm_input):
        raise ValueError("Please provide a valid path to the input SfM point cloud via --sfm_input.")
    if not args.nerf_inputs or not os.path.exists(args.nerf_inputs):
        raise ValueError("Please provide a valid path to the input point clouds via --nerf_inputs.")
    if not args.output:
        raise ValueError("Please provide a valid path to save the output point cloud via --output.")
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    merged_pcd = merge_pcs(args.sfm_input, args.nerf_inputs)
    o3d.io.write_point_cloud(args.output, merged_pcd)
    print("Point cloud saved to", args.output)
