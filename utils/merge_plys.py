#!/usr/bin/env python3
import open3d as o3d
import numpy as np
from pathlib import Path
import os
import argparse
import logging


def downsample(pc, max_points):
    """
    Downsamples a point cloud if it exceeds the maximum number of points.

    :param pc: the input point cloud
    :param max_points: the maximum number of points
    :return: the downsampled point cloud
    """
    if len(pc.points) > max_points:
        indices = np.random.choice(len(pc.points), max_points, replace=False)
        pc = pc.select_by_index(indices)
    return pc


def merge_pcs(sfm_input, nerf_inputs, max_points=None, weights=None):
    """
    Merges multiple point clouds into a single point cloud.
    :param sfm_input: the path to the SfM point cloud
    :param nerf_inputs: the path to the NeRF point clouds
    :param max_points: the maximum number of points
    :param weights: the weights for merging the point clouds
    :return:
    """
    # Read the SfM point cloud
    merged_pc = o3d.io.read_point_cloud(sfm_input)
    pc_paths = list(Path(nerf_inputs).rglob("*.ply"))
    if len(pc_paths) == 0:
        logging.warning("No NeRF point clouds found in the specified directory.")
        return merged_pc

    if max_points is not None:
        if weights is not None:
            weights = np.array(weights)
            weights /= np.sum(weights)
            if len(weights) != len(pc_paths) + 1:
                raise ValueError("The number of weights should match the number of input point clouds.")
        else:
            # Use uniform weights if not provided
            weights = np.ones(len(pc_paths) + 1) / (len(pc_paths) + 1)
        max_points = [int(max_points * w) for w in weights]
        merged_pc = downsample(merged_pc, max_points[0])

    for i, pc_path in enumerate(pc_paths):
        pc_path = str(pc_path)
        # Read the NeRF point cloud
        nerf_pc = o3d.io.read_point_cloud(pc_path)
        # Convert the NeRF point cloud to the COLMAP coordinate system
        applied_transform = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-0.0, -1.0, -0.0, -0.0],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

        nerf_pc.transform(applied_transform)
        if max_points is not None:
            nerf_pc = downsample(nerf_pc, max_points[i + 1])
        merged_pc += nerf_pc

    return merged_pc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge point clouds")
    parser.add_argument("--sfm_input", type=str, help="Path to the SfM pointcloud.")
    parser.add_argument("--nerf_inputs", type=str, help="Path to the NeRF point clouds.")
    parser.add_argument("--output", type=str, help="Path to save the output point cloud.")
    parser.add_argument("--max_points", type=int, default=None, help="Maximum number of points to merge.")
    parser.add_argument("--weights", type=float, nargs="+", help="Weights for merging the point clouds.", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not args.sfm_input or not os.path.exists(args.sfm_input):
        raise ValueError("Please provide a valid path to the input SfM point cloud via --sfm_input.")
    if not args.nerf_inputs or not os.path.exists(args.nerf_inputs):
        raise ValueError("Please provide a valid path to the input point clouds via --nerf_inputs.")
    if not args.output:
        raise ValueError("Please provide a valid path to save the output point cloud via --output.")
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    merged_pcd = merge_pcs(args.sfm_input, args.nerf_inputs, args.max_points, args.weights)
    o3d.io.write_point_cloud(args.output, merged_pcd)
    logging.info(f"Point cloud with {len(merged_pcd.points)} points saved to: {args.output}")
