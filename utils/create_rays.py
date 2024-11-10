import numpy as np
import json

def load_transforms(transforms_path):
    """
    Load camera parameters from a NeRF-style transforms.json file.
    
    Parameters:
        transforms_path (str): Path to the transforms.json file.
    
    Returns:
        List of dictionaries, each with intrinsic matrix, extrinsic matrix, and image dimensions.
    """
    with open(transforms_path, 'r') as f:
        transforms_data = json.load(f)

    frames_data = []
    for frame in transforms_data['frames']:
        pose_matrix = np.array(frame['transform_matrix'])  # 4x4 extrinsic matrix
        width = transforms_data['w']
        height = transforms_data['h']
        focal = transforms_data['fl_x']  # Assuming fl_x is the focal length

        intrinsic_matrix = np.array([
            [focal, 0, width / 2],
            [0, focal, height / 2],
            [0, 0, 1]
        ])
        
        frames_data.append({
            "intrinsic_matrix": intrinsic_matrix,
            "extrinsic_matrix": pose_matrix,
            "image_id": frame['file_path'],  # Optional, can use a custom ID
            "width": width,
            "height": height
        })
    
    return frames_data

def generate_rays_from_transforms(transforms_path):
    frames_data = load_transforms(transforms_path)
    all_rays = []

    for frame_data in frames_data:
        intrinsic_matrix = frame_data['intrinsic_matrix']
        extrinsic_matrix = frame_data['extrinsic_matrix']
        width = frame_data['width']
        height = frame_data['height']
        image_id = frame_data['image_id']

        # Generate rays for each pixel in this frame
        rays = generate_rays(width, height, intrinsic_matrix, extrinsic_matrix)
        
        # Add image ID to each ray
        for origin, direction in rays:
            all_rays.append((origin, direction, image_id))

    return all_rays


def generate_rays(width, height, intrinsic_matrix, extrinsic_matrix):
    """
    Generates rays for each pixel in an image based on camera intrinsic and extrinsic parameters.
    
    Parameters:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix (3x3).
        extrinsic_matrix (np.ndarray): Camera extrinsic matrix (4x4).
    
    Returns:
        rays (list): List of rays, each ray is represented as a tuple (origin, direction).
    """
    intrinsic_inv = np.linalg.inv(intrinsic_matrix)
    camera_origin = extrinsic_matrix[:3, 3]

    rays = []
    for y in range(height):
        for x in range(width):
            pixel = np.array([x, y, 1])
            camera_coords = intrinsic_inv @ pixel
            direction = extrinsic_matrix[:3, :3] @ camera_coords
            direction = direction / np.linalg.norm(direction)  # Normalize direction
            rays.append((camera_origin, direction))
    return rays

def store_rays_for_rad_splat(rays, output_path):
    ray_origins = np.array([ray[0] for ray in rays])       # (N, 3)
    ray_directions = np.array([ray[1] for ray in rays])    # (N, 3)
    image_ids = np.array([ray[2] for ray in rays])         # (N,)

    np.savez(output_path, ray_origins=ray_origins, ray_directions=ray_directions, image_ids=image_ids)
    

transforms_path = 'data/alameda/transforms.json'
all_rays = generate_rays_from_transforms(transforms_path)
output_path = 'data/alameda/rays_for_rad_splat.npz'
store_rays_for_rad_splat(all_rays, output_path)