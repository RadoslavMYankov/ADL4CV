import pycolmap
import numpy as np
from plyfile import PlyData, PlyElement
import json
import os
from pathlib import Path
from nerfstudio.data.utils.colmap_parsing_utils import qvec2rotmat, read_cameras_binary, read_images_binary, read_points3D_binary
from nerfstudio.process_data import colmap_utils    


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
    bin_path = Path('data/alameda/colmap/sparse/0')
    output_path = Path('sanity_check')

    colmap_utils.colmap_to_json(bin_path, output_path)

    #create_ply_file(bin_path)
    #create_transforms(bin_path)
    #create_base_cam(bin_path)
