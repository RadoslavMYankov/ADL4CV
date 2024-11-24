#merge ply files to generate a single point cloud

import open3d as o3d

path_to_sfm = '/home/team5/project/data/alameda/sparse_pc.ply'
path_to_nerf = '/home/team5/project/data/alameda/local_nerf_sparse_50k.ply'

sfm_pcd = o3d.io.read_point_cloud(path_to_sfm)
nerf_pcd = o3d.io.read_point_cloud(path_to_nerf)

# Merge the point clouds
merged_pcd = sfm_pcd + nerf_pcd

# Save the merged point cloud	
output_ply_file_path = "sfm_nerf_all_50k.ply"

o3d.io.write_point_cloud(output_ply_file_path, merged_pcd)
