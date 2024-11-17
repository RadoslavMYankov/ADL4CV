import open3d as o3d
import numpy as np

# Load the SfM point cloud
sfm_pcd = o3d.io.read_point_cloud("/home/team5/project/data/alameda/sparse_pc.ply")
sfm_points = np.asarray(sfm_pcd.points)
sfm_colors = np.asarray(sfm_pcd.colors) if sfm_pcd.has_colors() else None

# Load the NeRF point cloud
nerf_pcd = o3d.io.read_point_cloud("/home/team5/project/data/alameda/sparce_pc_nerf.ply")
nerf_points = np.asarray(nerf_pcd.points)
nerf_colors = np.asarray(nerf_pcd.colors) if nerf_pcd.has_colors() else None

# Identify sparse areas in the SfM point cloud
voxel_size = 0.05  # Adjust voxel size as needed
sfm_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(sfm_pcd, voxel_size)

# Create a dictionary to count the number of points in each voxel
voxel_density = {}
for point in sfm_points:
    voxel_index = tuple(np.floor(point / voxel_size).astype(int))
    if voxel_index in voxel_density:
        voxel_density[voxel_index] += 1
    else:
        voxel_density[voxel_index] = 1

voxel_density_threshold = 5  # Adjust threshold as needed
sparse_voxels = {voxel for voxel, count in voxel_density.items() if count < voxel_density_threshold}


# Select points from NeRF that fall within the sparse voxels
def is_point_in_sparse_voxel(point, sparse_voxels, voxel_size):
    voxel_index = np.floor(point / voxel_size).astype(int)
    return tuple(voxel_index) in sparse_voxels

selected_nerf_points = []
selected_nerf_colors = []

for point, color in zip(nerf_points, nerf_colors):
    if is_point_in_sparse_voxel(point, sparse_voxels, voxel_size):
        selected_nerf_points.append(point)
        selected_nerf_colors.append(color)

selected_nerf_points = np.array(selected_nerf_points)
selected_nerf_colors = np.array(selected_nerf_colors)

# Merge the point clouds
num_sfm_points = len(sfm_points)
num_nerf_points_needed = 1000000 - num_sfm_points

if len(selected_nerf_points) > num_nerf_points_needed:
    selected_nerf_points = selected_nerf_points[:num_nerf_points_needed]
    selected_nerf_colors = selected_nerf_colors[:num_nerf_points_needed]

merged_points = np.vstack((sfm_points, selected_nerf_points))
merged_colors = np.vstack((sfm_colors, selected_nerf_colors)) if sfm_colors is not None and selected_nerf_colors is not None else None

# Create the merged point cloud
merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(merged_points)

if merged_colors is not None:
    merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

# Save the merged point cloud to a new PLY file
o3d.io.write_point_cloud("/path/to/merged_point_cloud.ply", merged_pcd)
print("Merged point cloud saved to '/path/to/merged_point_cloud.ply'")