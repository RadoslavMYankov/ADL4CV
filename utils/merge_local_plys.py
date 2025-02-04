import open3d as o3d
from merge_plys import downsample
from prune_plys import nerf_cs_to_colmap
import os

def merge_and_prune(sfm_input, nerf_inputs, num_points, output_path):
    sfm_pcd = o3d.io.read_point_cloud(sfm_input)
    num_sfm_points = len(sfm_pcd.points)
    print(num_sfm_points)
    merged_pcd = sfm_pcd
    
    for nerf_pc in os.listdir(nerf_inputs):
        #print(nerf_pc)
        if not nerf_pc.endswith(".ply"):
            continue
        nerf_pcd = o3d.io.read_point_cloud(os.path.join(nerf_inputs, nerf_pc))
        nerf_pcd = downsample(nerf_pcd, num_points)
        #print(len(nerf_pcd.points))
        nerf_pcd = nerf_cs_to_colmap(nerf_pcd)
        merged_pcd += nerf_pcd
        print(len(merged_pcd.points))
    
    #print(len(merged_pcd.points))
    o3d.io.write_point_cloud(output_path, merged_pcd)


if __name__ == "__main__":
    sfm_input = "/home/team5/project/data/zipnerf/alameda/sparse_pc.ply"
    nerf_inputs = "/home/team5/project/alameda_pcs/global"
    num_points = 100000
    output_path = "/home/team5/project/data/zipnerf/alameda/plys/sfm_gloabl_nerfs_add_100k.ply"
    merge_and_prune(sfm_input, nerf_inputs, num_points, output_path)