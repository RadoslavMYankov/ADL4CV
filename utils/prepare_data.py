import pycolmap
import numpy as np
from plyfile import PlyData, PlyElement
import json
import os
from nerfstudio.data.utils.colmap_parsing_utils import qvec2rotmat, read_cameras_binary, read_images_binary, read_points3D_binary


def create_ply_file(bin_path):
    reconstruction = pycolmap.Reconstruction(bin_path)
    points3D = reconstruction.points3D

    # Prepare data for .ply file
    vertices = np.array([
        (point.xyz[0], point.xyz[1], point.xyz[2], point.color[0], point.color[1], point.color[2])
        for point in points3D.values()
    ], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    # Write to .ply
    ply_data = PlyData([PlyElement.describe(vertices, 'vertex')], text=True)
    ply_data.write('sparse_pc.ply')

def create_transforms(bin_path):
    reconstruction = pycolmap.Reconstruction(bin_path)
    images = read_images_binary(os.path.join(bin_path, 'images.bin'))

    # Define JSON structure
    transforms_data = {
        "frames": []
    }



    # Populate frames with camera poses and image paths
    for image_id, image in images.items():
        # Convert quaternion to rotation matrix
        rotation_matrix = qvec2rotmat(image.qvec)
        #rotation_matrix = image.rotation_matrix()
        
        # Combine rotation and translation to form a 4x4 transformation matrix
        transform_matrix = np.hstack((rotation_matrix, image.tvec.reshape(3, 1)))
        transform_matrix = np.vstack((transform_matrix, [0, 0, 0, 1]))  # Homogeneous coordinates

        transforms_data["frames"].append({
            "file_path": f"images/{image.name}",  # Adjust path to match your dataset structure
            "transform_matrix": transform_matrix.tolist()
        })

    # Save to transforms.json
    with open('transforms.json', 'w') as f:
        json.dump(transforms_data, f, indent=4)

def create_base_cam(bin_path):
    reconstruction = pycolmap.Reconstruction(bin_path)
    camera = next(iter(reconstruction.cameras.values()))  # Assuming one camera

    # Create JSON for base_cam
    base_cam_data = {
        "width": camera.width,
        "height": camera.height,
        "fl_x": camera.fx,       # Focal length in x-direction
        "fl_y": camera.fy,       # Focal length in y-direction
        "cx": camera.cx,         # Principal point in x
        "cy": camera.cy,         # Principal point in y
        "k1": camera.k1,         # Radial distortion coefficient
        "k2": camera.k2,         # Radial distortion coefficient
        "p1": camera.p1,         # Tangential distortion coefficient
        "p2": camera.p2          # Tangential distortion coefficient
    }

    # Save to base_cam.json
    with open('base_cam.json', 'w') as f:
        json.dump(base_cam_data, f, indent=4)

if __name__ == '__main__':
    bin_path = 'data/alameda/colmap/sparse/0'
    create_ply_file(bin_path)
