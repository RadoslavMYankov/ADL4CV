from merge_plys import downsample
import open3d as o3d
import argparse
import numpy as np

def nerf_cs_to_colmap(nerf_pcd):
    applied_transform = np.array([
        [1.0, 0.0,  0.0,  0.0],
        [0.0, 0.0,  1.0,  0.0],
        [-0.0, -1.0, -0.0, -0.0],
        [0.0, 0.0,  0.0,  1.0]
    ], dtype=np.float64)

    nerf_pcd.transform(applied_transform)
    return nerf_pcd

def merge_and_prune(sfm_input, nerf_inputs, num_points, output_path):
    sfm_pcd = o3d.io.read_point_cloud(sfm_input)
    num_sfm_points = len(sfm_pcd.points)
    print(num_sfm_points)
    num_nerf_points = num_points - num_sfm_points
    print(num_nerf_points)
    
    nerf_pcd = o3d.io.read_point_cloud(nerf_inputs)
    nerf_pcd = downsample(nerf_pcd, num_nerf_points)
    nerf_pcd = nerf_cs_to_colmap(nerf_pcd)
    merged_pcd = sfm_pcd + nerf_pcd
    o3d.io.write_point_cloud(output_path, merged_pcd)

def merge_and_prune_mipnerf(sfm_input, nerf_inputs, num_points, output_path):
    sfm_pcd = o3d.io.read_point_cloud(sfm_input)
    num_sfm_points = len(sfm_pcd.points)
    #print(num_sfm_points)
    num_nerf_points = num_points
    print(num_nerf_points)
    
    nerf_pcd = o3d.io.read_point_cloud(nerf_inputs)
    nerf_pcd = downsample(nerf_pcd, num_nerf_points)
    nerf_pcd = nerf_cs_to_colmap(nerf_pcd)
    merged_pcd = sfm_pcd + nerf_pcd
    o3d.io.write_point_cloud(output_path, merged_pcd)

def prune(path_to_pc, num_points, output_path):
    pc = o3d.io.read_point_cloud(path_to_pc)
    pc = downsample(pc, num_points)
    pc = nerf_cs_to_colmap(pc)
    o3d.io.write_point_cloud(output_path, pc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Downsample a point cloud.")
    
    parser.add_argument("--sfm_input", type=str, help="Path to the SfM pointcloud.")
    parser.add_argument("--path_to_nerf_pc", type=str, required=True, help="Path to the input point cloud.")
    parser.add_argument("--num_points", type=int, required=True, help="Number of points to downsample to.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output point cloud.")
    parser.add_argument("--merge", action="store_true", help="Whether to merge the point clouds or just prune")
    

    args = parser.parse_args()
    #print(args.merge)

    if args.merge:
        print("Merging and pruning")
        merge_and_prune(args.sfm_input, args.path_to_nerf_pc, args.num_points, args.output_path)
    else:
        print("Pruning")
        prune(args.path_to_nerf_pc, args.num_points, args.output_path)