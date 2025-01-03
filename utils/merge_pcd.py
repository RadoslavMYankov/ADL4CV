#merge ply files to generate a single point cloud

import open3d as o3d
import numpy as np
from nerfstudio.cameras import camera_utils
from nerfstudio.utils.io import load_from_json
from pathlib import Path
import torch

def nerf_cs_to_colmap(nerf_pcd):
    applied_transform = np.array([
        [1.0, 0.0,  0.0,  0.0],
        [0.0, 0.0,  1.0,  0.0],
        [-0.0, -1.0, -0.0, -0.0],
        [0.0, 0.0,  0.0,  1.0]
    ], dtype=np.float64)

    nerf_pcd.transform(applied_transform)
    return nerf_pcd

def merge_pcs(path_to_local_nerfs, path_to_sfm):

    sfm_pcd = o3d.io.read_point_cloud(path_to_sfm)
    

    # Merge the point clouds
    merged_pcd = sfm_pcd

    for i in range(1):
        #path_to_nerf = path_to_local_nerfs + 'pc_c' + str(i) + '.ply'
        path_to_nerf = path_to_local_nerfs
        print(path_to_nerf)
        nerf_pcd = o3d.io.read_point_cloud(path_to_nerf)
        nerf_pcd = nerf_cs_to_colmap(nerf_pcd)

        merged_pcd += nerf_pcd  

    #merged_pcd = merged_pcd.rotate(rotation_matrix, center=(0, 0, 0))

    #return nerf_pcd
    return merged_pcd


if __name__ == '__main__':
    path_to_local_nerfs = '/home/team5/project/nerf_cluster_0_manual.ply'
    #path_to_local_nerfs = '/home/team5/project/outputs/alameda/nerfacto/global_nerf/point_cloud.ply'
    path_to_sfm = '/home/team5/project/data/alameda/sparse_pc.ply'
    merged_pcd = merge_pcs(path_to_local_nerfs, path_to_sfm)
    output_ply_file_path = "nerf_sfm_cluster_0_manual.ply"
    o3d.io.write_point_cloud(output_ply_file_path, merged_pcd)
    print("Point cloud saved to", output_ply_file_path)
