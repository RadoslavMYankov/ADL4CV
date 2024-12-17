#merge ply files to generate a single point cloud

import open3d as o3d
import numpy as np
from nerfstudio.cameras import camera_utils
from nerfstudio.utils.io import load_from_json
from pathlib import Path
import torch

def merge_pcs(path_to_local_nerfs, path_to_sfm):

    sfm_pcd = o3d.io.read_point_cloud(path_to_sfm)
    
    '''for frame in meta["frames"]:
        filepath = Path(frame["file_path"])
        fname = data_dir / Path("images") / filepath.name
        fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]

    for frame in frames:
        poses.append(np.array(frame["transform_matrix"]))

    poses = torch.from_numpy(np.array(poses).astype(np.float32))
    poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
        poses,
        method ='up',
        center_method= 'poses',
    )'''

    rotation_matrix = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])

    '''points3D = torch.from_numpy(np.asarray(sfm_pcd.points, dtype=np.float32))
    points3D = (
        torch.cat(
            (
                points3D,
                torch.ones_like(points3D[..., :1]),
            ),
            -1,
        )
        @ transform_matrix.T
    )

    points3D = points3D[:, :3].numpy()  # Convert back to numpy array

    points3D_rgb = (np.asarray(sfm_pcd.colors) * 255).astype(np.uint8)  # Keep colors as numpy array

    # Update sfm_pcd with the transformed points and colors
    sfm_pcd.points = o3d.utility.Vector3dVector(points3D)
    sfm_pcd.colors = o3d.utility.Vector3dVector(points3D_rgb / 255.0)  # Normalize colors to [0, 1]'''

    angle_radians_x = np.deg2rad(90)

    # Create the rotation matrix
    rotation_matrix_x = np.array([
        [1, 0, 0],
        [0, np.cos(angle_radians_x), -np.sin(angle_radians_x)],
        [0, np.sin(angle_radians_x), np.cos(angle_radians_x)]
    ])

    angle_radians_y = np.deg2rad(-90)
    rotation_matrix_y = np.array([
        [np.cos(angle_radians_y), 0, np.sin(angle_radians_y)],
        [0, 1, 0],
        [-np.sin(angle_radians_y), 0, np.cos(angle_radians_y)]
    ])



    # Merge the point clouds
    merged_pcd = sfm_pcd

    for i in range(1):
        #path_to_nerf = path_to_local_nerfs + 'pc_c' + str(i) + '.ply'
        path_to_nerf = path_to_local_nerfs
        print(path_to_nerf)
        nerf_pcd = o3d.io.read_point_cloud(path_to_nerf)
        nerf_pcd = nerf_pcd.rotate(rotation_matrix, center=(0, 0, 0))
        nerf_pcd = nerf_pcd.rotate(rotation_matrix_x, center=(0, 0, 0))
        nerf_pcd = nerf_pcd.rotate(rotation_matrix_y, center=(0, 0, 0))

        merged_pcd += nerf_pcd  

    return nerf_pcd
    #return merged_pcd


if __name__ == '__main__':
    path_to_local_nerfs = '/home/team5/project/nerf_init.ply'
    #path_to_local_nerfs = '/home/team5/project/outputs/alameda/nerfacto/global_nerf/point_cloud.ply'
    path_to_sfm = '/home/team5/project/data/alameda/sparse_pc.ply'
    merged_pcd = merge_pcs(path_to_local_nerfs, path_to_sfm)
    output_ply_file_path = "nerf_init_aligned.ply"
    o3d.io.write_point_cloud(output_ply_file_path, merged_pcd)
    print("Point cloud saved to", output_ply_file_path)
