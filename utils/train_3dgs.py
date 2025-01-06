#!/usr/bin/env python
import os
import argparse
import logging
import subprocess

import numpy as np
import pandas as pd
import open3d as o3d
import time
import json


def merge_pcs(sfm_pc, nerf_inputs):
    merged_pc = o3d.io.read_point_cloud(sfm_pc)
    for pc_path in nerf_inputs:
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
        merged_pc += nerf_pc

    return merged_pc


def main():
    parser = argparse.ArgumentParser(
        description="Train 3DGS on a 3D scene with utilizing Nerfstudio."
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="alameda-3dgs",
        help="Name of the project."
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the input dataset directory or transforms.json file."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the output 3DGS models."
    )
    parser.add_argument(
        "--sfm",
        type=str,
        help="Path to the SfM pointcloud."
    )
    parser.add_argument(
        "--plys",
        type=str,
        default=None,
        help="Path to pointclouds directory."
    )
    parser.add_argument(
        "--max_num_iterations",
        type=int,
        nargs='+',
        default=[100000],
        help="List of maximum number of iterations for training the 3DGS."
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"The input file does not exist: {args.data}")
    if (args.plys is not None) and (not os.path.exists(args.plys)):
        raise FileNotFoundError(f"The input file does not exist: {args.plys}")
    if not os.path.exists(args.sfm):
        raise FileNotFoundError(f"The input file does not exist: {args.sfm}")
    if not os.path.exists(os.path.join(args.output, "initializations")):
        os.makedirs(os.path.join(args.output, "initializations"))

    logging.basicConfig(level=logging.INFO)

    if os.path.isdir(args.data):
        # If the input is a directory, assume it contains the transforms.json file
        transforms_path = os.path.join(args.data, "transforms.json")
    else:
        transforms_path = args.data

    if not os.path.isfile(transforms_path):
        raise FileNotFoundError(f"The input file does not exist: {transforms_path}")

    transforms = json.load(open(transforms_path, 'r'))

    # Read the SfM point cloud
    sfm_pointcloud = o3d.io.read_point_cloud(args.sfm)
    # Save the SfM point cloud to the output directory for standardization
    o3d.io.write_point_cloud(os.path.join(args.output, "initializations", "sfm.ply"), sfm_pointcloud)

    initializations = {
        "sfm": sfm_pointcloud,
    }
    merging_times = {
        "sfm": 0,
    }
    training_times = {}

    if args.plys is not None:
        # Determine the different numbers of iterations for which plys are available (cluster_x_yits.ply)
        plys = [f for f in os.listdir(args.plys) if f.endswith('.ply')]
        iterations = list(set([int(p.split('_')[-1].split('its.')[0]) for p in plys]))
        logging.info(f"Iterations found: {iterations}")

        for it in iterations:
            logging.debug(f"Merging plys for {it} iterations")
            nerf_inputs = [os.path.join(args.plys, f) for f in plys if f.endswith(f"{it}its.ply")]
            start_time = time.time()
            # Merge ply files to generate a single point cloud
            merged_pcd = merge_pcs(args.sfm, nerf_inputs)
            merging_time = time.time() - start_time
            merging_times[f"local_{it}its"] = merging_time
            # Save the merged point cloud
            o3d.io.write_point_cloud(os.path.join(args.output, "initializations", f"local_{it}its.ply"), merged_pcd)
            initializations[f"local_{it}its"] = merged_pcd
            logging.info(f"Merged {len(nerf_inputs)} plys for {it} iterations in {merging_time:.2f} seconds")

    for initialization_name, initialization in initializations.items():
        # Generate a new transforms.json with the correct initialization
        init_transforms = transforms.copy()
        # Adjust the paths to the images
        init_transforms['frames'] = [
            {
                **frame,
                "file_path": os.path.relpath(
                    os.path.join(args.data, "images", frame["file_path"].split('/')[-1]),
                    os.path.join(args.output, "initializations")
                )
            }
            for frame in init_transforms['frames']
        ]
        # Set the initialization point cloud
        init_transforms['ply_file_path'] = os.path.relpath(
            os.path.join(args.output, "initializations", f"{initialization_name}.ply"),
            os.path.join(args.output, "initializations")
        )
        # Save the new transforms.json
        with open(os.path.join(args.output, "initializations", f"{initialization_name}_transforms.json"), 'w') as f:
            json.dump(init_transforms, f, indent=4)
        logging.debug(f"Saved transforms for {initialization_name} initialization")

        for max_num_iterations in args.max_num_iterations:
            logging.info(f"Training 3DGS with {initialization_name} initialization for {max_num_iterations} iterations")
            train_command = (
                f"ns-train splatfacto --data {os.path.join(args.output, 'initializations', f'{initialization_name}_transforms.json')} "
                f"--pipeline.model.cull-alpha-tresh 0.005 "
                f"--timestamp {initialization_name}_{max_num_iterations}3dgsits --project-name {args.project_name} "
                f"--output-dir {args.output} "
                f"--vis tensorboard --max-num-iterations {max_num_iterations} "
                f"--machine.num-devices 1 "
            )
            start_time = time.time()
            try:
                subprocess.run(train_command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Training failed for {initialization_name} initialization: {e}")
                continue
            training_duration = time.time() - start_time
            if initialization_name not in training_times:
                training_times[initialization_name] = {}
            training_times[initialization_name][max_num_iterations] = training_duration
            logging.info(f"Training completed for {initialization_name} initialization with {max_num_iterations} iterations in {training_duration:.2f} seconds")


    # Save the metrics
    training_times_df = pd.DataFrame(training_times)
    training_times_df.to_csv(os.path.join(args.output, "training_times.csv"), index_label="iterations")
    merging_times_df = pd.DataFrame(merging_times, index=[0])
    merging_times_df.to_csv(os.path.join(args.output, "merging_times.csv"), index=False)

if __name__ == "__main__":
    main()
