import open3d as o3d
import numpy as np
import struct

def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<I", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            params = struct.unpack("<" + "d" * 4, f.read(8 * 4))
            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras

def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = struct.unpack("<" + "d" * 4, f.read(8 * 4))
            tvec = struct.unpack("<" + "d" * 3, f.read(8 * 3))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = ""
            while True:
                char = f.read(1).decode("utf-8")
                if char == "\x00":
                    break
                name += char
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            xys = []
            point3D_ids = []
            for _ in range(num_points2D):
                xys.append(struct.unpack("<" + "d" * 2, f.read(8 * 2)))
                point3D_ids.append(struct.unpack("<Q", f.read(8))[0])
            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
                "xys": np.array(xys),
                "point3D_ids": np.array(point3D_ids),
            }
    return images

def read_points3D_binary(path):
    points3D = {}
    with open(path, "rb") as f:
        num_points3D = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points3D):
            point3D_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<" + "d" * 3, f.read(8 * 3))
            rgb = struct.unpack("<" + "B" * 3, f.read(3))
            error = struct.unpack("<d", f.read(8))[0]
            track_length = struct.unpack("<Q", f.read(8))[0]
            track = []
            for _ in range(track_length):
                image_id = struct.unpack("<I", f.read(4))[0]
                point2D_idx = struct.unpack("<I", f.read(4))[0]
                track.append((image_id, point2D_idx))
            points3D[point3D_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track,
            }
    return points3D

# Load the COLMAP data
cameras = read_cameras_binary("/home/team5/project/data/alameda/colmap/sparse/0/cameras.bin")
images = read_images_binary("/home/team5/project/data/alameda/colmap/sparse/0/images.bin")
points3D = read_points3D_binary("/home/team5/project/data/alameda/colmap/sparse/0/points3D.bin")

# Define the subset of image IDs
subset_image_ids = set(list(images.keys())[:100])

# Identify points associated with the subset of images
subset_point_ids = set()
for point_id, point_data in points3D.items():
    for image_id, _ in point_data["track"]:
        if image_id in subset_image_ids:
            subset_point_ids.add(point_id)
            break

# Extract the relevant points and colors
subset_points = []
subset_colors = []
for point_id in subset_point_ids:
    point_data = points3D[point_id]
    subset_points.append(point_data["xyz"])
    subset_colors.append(point_data["rgb"])

subset_points = np.array(subset_points)
subset_colors = np.array(subset_colors) / 255.0  # Normalize colors to [0, 1]

# Create an Open3D point cloud
subset_pcd = o3d.geometry.PointCloud()
subset_pcd.points = o3d.utility.Vector3dVector(subset_points)
subset_pcd.colors = o3d.utility.Vector3dVector(subset_colors)

# Save the point cloud to a new PLY file
output_ply_file_path = "subset_point_cloud.ply"
o3d.io.write_point_cloud(output_ply_file_path, subset_pcd)
print(f"Subset point cloud saved to '{output_ply_file_path}'")