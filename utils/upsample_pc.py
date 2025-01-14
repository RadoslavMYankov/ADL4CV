import open3d as o3d
import numpy as np
import argparse
import logging

def upsample_pointcloud(pcd, target_num_points, sigma=0.01, max_deviation=0.04):
    """
    Upsample a point cloud by cloning and jittering points, preserving all attributes.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        target_num_points (int): The desired number of points in the upsampled point cloud.
        sigma (float): Standard deviation of the Gaussian noise added to the cloned points.
        max_deviation (float): Maximum allowable deviation for Gaussian noise (caps at 4 * sigma).

    Returns:
        o3d.geometry.PointCloud: The upsampled point cloud with preserved attributes.
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    normals = np.asarray(pcd.normals) if pcd.has_normals() else None

    current_num_points = points.shape[0]
    if target_num_points <= current_num_points:
        raise ValueError("Target number of points must be greater than the current number of points.")

    # Calculate how many new points are needed
    num_new_points = target_num_points - current_num_points

    # Randomly choose points (with replacement) to clone
    indices = np.random.choice(current_num_points, size=num_new_points, replace=True)

    # Generate new points by adding Gaussian noise, capped at max_deviation
    cloned_points = points[indices]
    noise = np.clip(
        np.random.normal(scale=sigma, size=cloned_points.shape),
        -max_deviation, max_deviation
    )
    jittered_points = cloned_points + noise

    # Create a new point cloud for the jittered points
    jittered_pcd = o3d.geometry.PointCloud()
    jittered_pcd.points = o3d.utility.Vector3dVector(jittered_points)
    if colors is not None:
        cloned_colors = colors[indices]
        jittered_pcd.colors = o3d.utility.Vector3dVector(cloned_colors)
    if normals is not None:
        cloned_normals = normals[indices]
        jittered_pcd.normals = o3d.utility.Vector3dVector(cloned_normals)

    # Combine the original and jittered point clouds
    combined_pcd = pcd + jittered_pcd

    return combined_pcd

def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Upsample a point cloud by cloning and jittering points.")
    parser.add_argument("source", type=str, help="Path to the source point cloud file.")
    parser.add_argument("output", type=str, help="Path to save the upsampled point cloud file.")
    parser.add_argument("--target_num", type=int, required=True, help="Target number of points in the upsampled point cloud.")
    parser.add_argument("--sigma", type=float, default=0.01, help="Standard deviation of Gaussian noise (default: 0.01).")
    parser.add_argument("--max_deviation", type=float, default=0.04, help="Maximum allowable deviation for noise (default: 0.04).")

    args = parser.parse_args()

    # Load the source point cloud
    pcd = o3d.io.read_point_cloud(args.source)
    if pcd.is_empty():
        logging.error(f"The point cloud at {args.source} is empty or could not be loaded.")
        raise ValueError(f"The point cloud at {args.source} is empty or could not be loaded.")

    logging.info(f"Loaded point cloud from {args.source} with {len(pcd.points)} points.")

    # Perform upsampling
    upsampled_pcd = upsample_pointcloud(
        pcd,
        target_num_points=args.target_num,
        sigma=args.sigma,
        max_deviation=args.max_deviation
    )

    # Save the upsampled point cloud
    o3d.io.write_point_cloud(args.output, upsampled_pcd)
    logging.info(f"Upsampled point cloud saved to {args.output}")

if __name__ == "__main__":
    main()
