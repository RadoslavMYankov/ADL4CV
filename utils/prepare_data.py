import argparse
import pycolmap
import numpy as np
from plyfile import PlyData, PlyElement
import json
import os
from pathlib import Path
from nerfstudio.data.utils.colmap_parsing_utils import qvec2rotmat, read_cameras_binary, read_images_binary, read_points3D_binary


def create_ply_file(bin_path):
    reconstruction = pycolmap.Reconstruction(bin_path)
    points3D = reconstruction.points3D

    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([0, 2, 1]), :]
    applied_transform[2, :] *= -1
    
    points = np.array([[p.xyz[0], p.xyz[1], p.xyz[2]] for p in points3D.values()], dtype=np.float32)
    points = np.einsum('ij,bj->bi', applied_transform[:3, :3], points) + applied_transform[:3, 3]

    points_rgb = np.array([p.color for p in points3D.values()], dtype=np.uint8)

    with open("sparse_pc.ply", "w") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points3D)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uint8 red\n")
        f.write("property uint8 green\n")
        f.write("property uint8 blue\n")
        f.write("end_header\n")

        for coord, color in zip(points, points_rgb):
            x, y, z = coord[0], coord[1], coord[2]
            r, g, b = color[0], color[1], color[2]
            f.write(f"{x:8f} {y:8f} {z:8f} {r} {g} {b}\n")
    print("Sparse point cloud saved to sparse_pc.ply")


def create_transforms(bin_path):
    images = read_images_binary(os.path.join(bin_path, 'images.bin'))
    camera = read_cameras_binary(os.path.join(bin_path, 'cameras.bin'))
    camera_list = list(camera.values())

    print(camera_list[0].params)
    # Define JSON structure
    transforms_data = {
        "w": camera_list[0].width,  # Image width
        "h": camera_list[0].height,  # Image height
        "fl_x": camera_list[0].params[0],       # Focal length in x-direction
        "fl_y": camera_list[0].params[1],       # Focal length in y-direction
        "cx": camera_list[0].params[2],         # Principal point in x
        "cy": camera_list[0].params[3],         # Principal point in y
        "k1": 0,         # Radial distortion coefficient
        "k2": 0,         # Radial distortion coefficient
        "p1": 0,         # Tangential distortion coefficient
        "p2": 0,          # Tangential distortion coefficient
        "frames": []
    }



    # Populate frames with camera poses and image paths
    for image_id, image in images.items():
        rotation = qvec2rotmat(image.qvec)

        translation = image.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        c2w[0:3, 1:3] *= -1
        
        c2w = c2w[np.array([0, 2, 1, 3]), :]
        c2w[2, :] *= -1

        name = image.name
        name = Path(f"./images/{name}")

        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": image_id,
        }
        transforms_data["frames"].append(frame)

    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([0, 2, 1]), :]
    applied_transform[2, :] *= -1
    transforms_data["applied_transform"] = applied_transform.tolist()

    # Save to transforms.json
    with open('transforms.json', 'w') as f:
        json.dump(transforms_data, f, indent=4)
    print("Transforms saved to transforms.json")

def create_base_cam(bin_path):
    # not implemented yet do not use
    camera = read_cameras_binary(os.path.join(bin_path, 'cameras.bin'))
    camera_list = list(camera.values())

    
    # Create JSON for base_cam - single camera
    base_cam_data = {
        "width": camera_list[0].width,
        "height": camera_list[0].height,
        # parameters not avaialble in the camera dictionary
        #"fl_x": camera_list[0].fx,       # Focal length in x-direction
        #"fl_y": camera_list[0].fy,       # Focal length in y-direction
        #"cx": camera_list[0].cx,         # Principal point in x
        #"cy": camera_list[0].cy,         # Principal point in y
        #"k1": camera_list[0].k1,         # Radial distortion coefficient
        #"k2": camera_list[0].k2,         # Radial distortion coefficient
        #"p1": camera_list[0].p1,         # Tangential distortion coefficient
        #"p2": camera_list[0].p2          # Tangential distortion coefficient
    }

    # Save to base_cam.json
    with open('base_cam.json', 'w') as f:
        json.dump(base_cam_data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin_path', type=str, default='data/alameda/colmap/sparse/0')
    args = parser.parse_args()
    bin_path = args.bin_path

    create_ply_file(bin_path)
    create_transforms(bin_path)
    #create_base_cam(bin_path)
