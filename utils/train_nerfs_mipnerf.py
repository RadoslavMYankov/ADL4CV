#!/usr/bin/env python
import json
import os
import argparse
import logging
import numpy as np
import pandas as pd
import subprocess
from time import time
import open3d as o3d
from pathlib import Path


def export_pointcloud(input_path, output_path, num_points):
    """
    Uses ns-export to export a point cloud from a NeRF model.
    :param input_path: The path to the input NeRF model directory.
    :param output_path: The path to save the exported point cloud.
    :param num_points: The number of points to export.
    """
    export_command = (
        f"ns-export pointcloud --load-config {input_path} --output-dir {output_path}"
        f" --num-points {num_points} --save-world-frame True"
    )
    logging.info(f"Exporting point cloud from {input_path} to {output_path} with {num_points} points.")
    try:
        subprocess.run(export_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Export failed: {e}")
        return
    logging.info(f"Export completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Train NeRFs on local regions of a 3D scene utilizing Nerfstudio."
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="alameda-local-nerfs",
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
        help="Path to save the output NeRF models."
    )
    parser.add_argument(
        "--max_num_iterations",
        type=int,
        nargs='+',
        default=[100000],
        help="List of maximum number of iterations for training the local NeRFs."
    )
    parser.add_argument(
        "--save_plys",
        action="store_true",
        default=True,
        help="Whether to save the point clouds as PLY files."
    )
    parser.add_argument(
        "--num_points",
        type=int,
        nargs='+',
        default=50000,
        help="Number of points to export in the point cloud."
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"The input file or directory does not exist: {args.data}")
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    logging.basicConfig(level=logging.INFO)
    for scene in os.listdir(args.data):
        if os.path.join(args.data, scene).endswith(".json"):
            transforms_path = args.data
        else:
            transforms_path = os.path.join(args.data, scene, "transforms_merged_clusters.json")

        for max_iterations in args.max_num_iterations:
            # Train local NeRF for the cluster
            ns_train_command = (
                f"ns-train nerfacto --data {transforms_path} --output-dir {args.output} "
                f"--pipeline.model.predict-normals True "
                f"--timestamp {scene}_{max_iterations}its --project-name {args.project_name} "
                f"--vis tensorboard --max-num-iterations {max_iterations} "
                f"--machine.num-devices 1 "
                f"--save-only-latest-checkpoint True --steps-per-eval-batch {10 * max_iterations} --steps_per_eval_image {10 * max_iterations}"  # Skip eval stuff to be faster
            )

            logging.info(f"Starting training for scene {scene} with {max_iterations} iterations.")
            start_time = time()
            try:
                subprocess.run(ns_train_command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                logging.error(f"Training failed for scene {scene}: {e}")
                continue
            training_duration = time() - start_time

            logging.info(f"Training completed for scene {scene} with {max_iterations} iterations in {training_duration:.2f} seconds.")


            if args.save_plys:
                for points in args.num_points:
                    # Export point cloud from the trained model
                    model_path = os.path.join(args.output, f"{scene}", "nerfacto", f"{scene}_{max_iterations}its")
                    start_time = time()
                    export_pointcloud(
                        os.path.join(model_path, "config.yml"),
                        model_path,
                        num_points=points
                    )
                    export_duration = time() - start_time
                    # Copy the point cloud to the cluster directory
                    point_cloud_path = os.path.join(model_path, f"point_cloud.ply")
                    if os.path.exists(point_cloud_path):
                        ply_dir = os.path.join(args.output, "point_clouds")
                        if not os.path.exists(ply_dir):
                            os.makedirs(ply_dir)
                        ply_path = os.path.join(ply_dir, f"{scene}_{max_iterations}its_{points}pts.ply")
                        os.rename(point_cloud_path, ply_path)
                    logging.info(f"Point cloud exported from the trained model in {export_duration:.2f} seconds.")
            else:
                export_duration = None

            # Store the time taken for training
            with open(os.path.join(args.output, f"{scene}_{max_iterations}its_metrics.json"), 'w') as f:
                json.dump({
                    "training_duration": training_duration,
                    "export_duration": export_duration
                }, f, indent=4)

    logging.info("Training automation completed for all scenes.")


if __name__ == "__main__":
    main()
