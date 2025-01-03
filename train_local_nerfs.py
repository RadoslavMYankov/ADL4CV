#!/usr/bin/env python
import json
import os
import argparse
import logging
import pandas as pd
import subprocess


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
        "--clusters",
        type=str,
        help="Path to the clusters.csv file containing image cluster information."
    )
    parser.add_argument(
        "--min_sparse_images",
        type=int,
        default=5,
        help="Minimum number of sparse images required for a cluster to be used for training a local NeRF."
    )
    parser.add_argument(
        "--max_clusters",
        type=int,
        default=5,
        help="Maximum number of clusters to use for training local NeRFs."
    )
    parser.add_argument(
        "--max_num_iterations",
        type=int,
        default=100000,
        help="Maximum number of iterations for training the local NeRFs."
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"The input file or directory does not exist: {args.data}")
    if not os.path.exists(args.clusters):
        raise FileNotFoundError(f"The clusters file does not exist: {args.clusters}")
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    logging.basicConfig(level=logging.INFO)
    if args.data.endswith(".json"):
        transforms_path = args.data
    else:
        transforms_path = os.path.join(args.data, "transforms.json")

    # Load the transforms.json file containing all images of the original dataset
    try:
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"The input file does not exist: {transforms_path}")
    frames = transforms['frames']

    clusters = pd.read_csv(args.clusters)
    # Sort by number of sparse images
    sparse_images = clusters[clusters['is_sparse'] == True].groupby('cluster_id').size()
    cluster_ids = sparse_images.sort_values(ascending=False).index
    # Filter out clusters with less than min_sparse_images
    cluster_ids = cluster_ids[sparse_images >= args.min_sparse_images]
    # Remove the noise cluster
    cluster_ids = cluster_ids[cluster_ids != -1]
    # Limit the number of clusters
    cluster_ids = cluster_ids[:args.max_clusters]

    for cluster_id in cluster_ids:
        image_names = clusters[clusters['cluster_id'] == cluster_id]['image_name'].values
        image_names_lower = [name.lower() for name in image_names]
        cluster_dir = os.path.join(args.output, f"cluster_{cluster_id}")

        cluster_frames = [
            {
                **frame,
                "file_path": os.path.relpath(
                    os.path.join(args.data, "images", frame["file_path"].split('/')[-1]),
                    cluster_dir
                )
            }
            for frame in frames
            if frame['file_path'].split('/')[-1].lower() in image_names_lower
        ]

        if not cluster_frames:
            logging.warning(f"Cluster {cluster_id} has no valid frames and will be skipped.")
            continue

        # Create cluster transforms.json
        cluster_transforms = transforms.copy()
        cluster_transforms['frames'] = cluster_frames

        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)

        cluster_transforms_path = os.path.join(cluster_dir, f'cluster_{cluster_id}_transforms.json')
        with open(cluster_transforms_path, 'w') as f:
            json.dump(cluster_transforms, f, indent=4)

        logging.info(f"Saved transforms for cluster {cluster_id}.")

        # Train local NeRF for the cluster
        ns_train_command = (
            f"ns-train nerfacto --data {cluster_transforms_path} --output-dir {cluster_dir} "
            f"--timestamp cluster_{cluster_id} --project-name {args.project_name} "
            f"--vis tensorboard --max-num-iterations {args.max_num_iterations} "
            f"--machine.num-devices 1"
        )

        logging.info(f"Training NeRF for cluster {cluster_id} with command: {ns_train_command}")
        try:
            subprocess.run(ns_train_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Training failed for cluster {cluster_id}: {e}")
            continue

        logging.info(f"Training completed for cluster {cluster_id}.")

    logging.info("Training automation completed for all clusters.")


if __name__ == "__main__":
    main()
