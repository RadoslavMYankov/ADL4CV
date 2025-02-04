import os
import argparse
import logging
import pandas as pd
import open3d as o3d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display the number of points for multiple PLY files.")
    parser.add_argument("input_path", type=str, help="Path to folder containing the scenes.")
    args = parser.parse_args()

    input_path = args.input_path
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"The specified input path does not exist: {input_path}")

    plys = []

    # Traverse each top-level folder in the input path
    for scene_name in os.listdir(input_path):
        if not os.path.isdir(os.path.join(input_path, scene_name)):
            continue
        ply_dir = os.path.join(input_path, scene_name, "plys")
        sfm_pc_path = os.path.join(ply_dir, "sparse_pc.ply")
        if not os.path.exists(sfm_pc_path):
            if os.path.exists(os.path.join(input_path, scene_name, "sparse_pc.ply")):
                sfm_pc_path = os.path.join(input_path, scene_name, "sparse_pc.ply")
            else:
                logging.warning(f"The SfM point cloud for scene {scene_name} could not be found.")
                continue
        # Read the SfM point cloud
        sfm_pc = o3d.io.read_point_cloud(sfm_pc_path)
        num_points = len(sfm_pc.points)
        plys.append({"scene": scene_name, "num_points": num_points})
        logging.debug(f"Scene: {scene_name}, Num Points: {num_points}")

    plys_df = pd.DataFrame(plys)
    print(plys_df)
