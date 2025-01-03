import argparse
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycolmap
import tqdm
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import json


def init_worker(global_data):
    """
    Initializes global variables in each worker process.
    """
    global points3D, model_colors, point_ids, tracks, K, tracks_for_image
    points3D, model_colors, point_ids, tracks, K, tracks_for_image = global_data


def project_to_img(image_data):
    """
    Projects 3D points onto a 2D image plane using pre-extracted data.

    Args:
        image_data (dict): A dictionary containing all necessary data for the image.

    Returns:
        dict: Contains filtered 2D projected points and image dimensions.
    """
    try:
        image_id = image_data['image_id']
        image_name = image_data['image_name']
        cam_from_world = image_data['cam_from_world']
        use_track = image_data['use_track']
        use_color = image_data['use_color']
        color_tolerance = image_data['color_tolerance']
        images_path = image_data['images_path']

        # Extract R and T from cam_from_world
        extrinsic_matrix = cam_from_world
        R, T = extrinsic_matrix[:, :3], extrinsic_matrix[:, 3]
        R_vec = cv2.Rodrigues(R)[0]  # Convert rotation matrix to vector

        # Project 3D points to the 2D image plane
        projected_points2D = cv2.projectPoints(points3D, R_vec, T, K, None)[0].reshape(-1, 2)

        # Load image
        img_path = os.path.join(images_path, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image {image_name} not found at {img_path}.")
        height, width = img.shape[:2]

        # Filter points within image bounds
        x = projected_points2D[:, 0]
        y = projected_points2D[:, 1]
        in_bounds_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)

        if not np.any(in_bounds_mask):
            return None  # No points to process

        points2D_in_bounds = projected_points2D[in_bounds_mask]
        model_colors_in_bounds = model_colors[in_bounds_mask]
        point_ids_in_bounds = point_ids[in_bounds_mask]

        # Initialize valid_mask
        num_points = len(points2D_in_bounds)

        if use_color:
            # Extract pixel colors from the image
            coords = points2D_in_bounds.astype(int)
            projected_colors = img[coords[:, 1], coords[:, 0]] / 255.0

            # Compute color differences
            color_diff = np.linalg.norm(projected_colors - model_colors_in_bounds, axis=1)
            color_mask = color_diff <= color_tolerance
        else:
            color_mask = np.zeros(num_points, dtype=bool)

        if use_track:
            # Track membership check
            visible_point_ids = tracks_for_image[image_id]
            track_mask = np.in1d(point_ids_in_bounds, list(visible_point_ids), assume_unique=True)
        else:
            track_mask = np.zeros(num_points, dtype=bool)

        # Combine masks
        valid_mask = color_mask | track_mask

        # Return filtered points
        filtered_points2D = points2D_in_bounds[valid_mask]
        return {
            'image_id': image_id,
            'points2D': filtered_points2D,
            'height': height,
            'width': width
        }
    except Exception as e:
        #print("Projection error fix here")
        print(f"Error processing image {image_name}: {e}")
        return None


def density_map(points2D, image_shape, sigma=10):
    """
    Create a density map from projected 2D points with Gaussian smoothing.

    Args:
        points2D (np.ndarray): Array of 2D points (N, 2).
        image_shape (tuple): Shape of the density map (height, width).
        sigma (int): Standard deviation for Gaussian blur.

    Returns:
        np.ndarray: Smoothed density map.
    """
    height, width = image_shape[:2]
    density_map = np.zeros((height, width), dtype=np.float32)

    # Convert to integer indices
    x = points2D[:, 0].astype(np.int32)
    y = points2D[:, 1].astype(np.int32)

    # Filter points within image boundaries
    valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid_mask]
    y = y[valid_mask]

    # Accumulate counts using NumPy's advanced indexing
    np.add.at(density_map, (y, x), 1)

    # Apply Gaussian blur using OpenCV
    ksize = int(6 * sigma + 1)
    smoothed_density_map = cv2.GaussianBlur(density_map, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    return smoothed_density_map


def process_image(image_data):
    """
    Processes a single image: projects 3D points, generates density map, and computes statistics.

    Args:
        image_data (dict): A dictionary containing all necessary data for the image.

    Returns:
        dict: Contains image_id, number of projected points, and fraction of high-density area.
    """
    try:
        result = project_to_img(image_data)
        if result is None:
            return {
                "image_id": image_data['image_id'],
                "num_projected_points": 0,
                "fraction_high_density_area": 0.0
            }

        points2D = result['points2D']
        height = result['height']
        width = result['width']
        density_threshold = image_data['density_threshold']
        #print("Results extracted")

        # Generate the density map for the projected points
        smoothed_density_map = density_map(points2D, (height, width), sigma=10)

        # Calculate the fraction of the image area with density above the threshold
        high_density_area = np.sum(smoothed_density_map > density_threshold)

        # Export the density map as a mask
        if 'mask_path' in image_data:
            #print("ADD MASK PATH")
            mask_prefix = image_data['mask_path']
            #print(mask_prefix)
            # If the folder does not exist, create it
            os.makedirs(mask_prefix, exist_ok=True)
            mask_path = f"{mask_prefix}/{image_data['image_name']}"
            # Areas with high density get masked out
            cv2.imwrite(mask_path, (smoothed_density_map < density_threshold).astype(np.uint8) * 255)
            logging.debug(f"Saved mask to {mask_path}")

        total_area = height * width
        fraction_high_density = high_density_area / total_area

        num_projected_points = points2D.shape[0]

        return {
            "image_id": image_data['image_id'],
            "num_projected_points": num_projected_points,
            "fraction_high_density_area": fraction_high_density
        }
    except Exception as e:
        #print("Error processing image fix here")
        print(f"Error processing image {image_data['image_name']}: {e}")
        return {
            "image_id": image_data['image_id'],
            "num_projected_points": 0,
            "fraction_high_density_area": 0.0
        }

def compute_overlap(task):
    """
    Computes the overlap between two sets of 3D point IDs.

    Args:
        task (tuple): Contains indices and sets of point IDs for two images.

    Returns:
        tuple: Indices of the images and their overlap score.
    """
    i, j, points1, points2 = task
    intersection = points1.intersection(points2)
    min_len = min(len(points1), len(points2))
    if min_len > 0:
        overlap = len(intersection) / min_len
    else:
        overlap = 0.0
    return (i, j, overlap)


def compute_overlap_matrix(images, method="poses"):
    """
    Computes the overlap matrix for a list of images based on shared 3D points or camera poses.

    Args:
        images (list): List of image objects.
        method (str): Method to compute overlap: "shared_points" or "poses".
        t_weight (float): Weight for translation distance.
        q_weight (float): Weight for quaternion distance
    Returns:
        np.ndarray: Overlap matrix.
    """
    if method not in ("shared_points", "poses"):
        raise ValueError(f"Invalid method: {method}")

    num_images = len(images)
    overlap_matrix = np.zeros((num_images, num_images), dtype=np.float32)

    if method == "shared_points":
        # Extract points3D IDs for each image
        points3D_per_image = []
        for image in images:
            point3D_ids = set(point2D.point3D_id for point2D in image.get_valid_points2D())
            points3D_per_image.append(point3D_ids)

        # Prepare tasks for parallel computation
        tasks = []
        for i in range(num_images):
            for j in range(i + 1, num_images):
                tasks.append((i, j, points3D_per_image[i], points3D_per_image[j]))

        # Use ProcessPoolExecutor to parallelize the computation
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count() - 2) as executor:
            futures = {executor.submit(compute_overlap, task): task[:2] for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Computing shared points overlaps"):
                i, j, overlap = future.result()
                overlap_matrix[i, j] = overlap_matrix[j, i] = overlap
    elif method == "poses":
        # Compute pose overlap matrix
        pose_distances = compute_pose_overlap(images)

        # Invert the distances to get overlap scores
        overlap_matrix = 1 - pose_distances
    return overlap_matrix


def cluster_images(overlap_matrix, images, eps=0.5, min_samples=2):
    """
    Clusters images based on the overlap matrix using DBSCAN.

    Args:
        overlap_matrix (np.ndarray): Overlap matrix.
        images (list): List of image objects.
        eps (float): DBSCAN eps parameter.
        min_samples (int): DBSCAN min_samples parameter.

    Returns:
        dict: Clusters with cluster labels as keys and lists of image IDs as values.
        np.ndarray: Cluster labels for each image.
    """
    # Compute the distance matrix
    distance_matrix = 1 - overlap_matrix

    # Run DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed", n_jobs=-1)
    cluster_labels = dbscan.fit_predict(distance_matrix)

    # Build clusters using NumPy for efficiency
    cluster_ids = np.unique(cluster_labels)
    clusters = {cluster_id: [] for cluster_id in cluster_ids}

    # Extract image IDs once to avoid repeated access
    image_ids = np.array([image.image_id for image in images])

    for cluster_id in cluster_ids:
        indices = np.where(cluster_labels == cluster_id)[0]
        clusters[cluster_id] = image_ids[indices].tolist()

    return clusters, cluster_labels


def compute_pose_overlap(images, rotation_weight=0.1, max_rotation_distance=0.5, rotation_penalty=10.):
    """
    Compute pose overlap between images based on translation and quaternion distances.

    Args:
        images (list): List of image objects.
        t_weight (float): Weight for translation distance.
        q_weight (float): Weight for quaternion distance.
        eps (float): DBSCAN eps parameter.
        min_samples (int): DBSCAN min_samples parameter.

    Returns:
        np.ndarray: Overlap matrix based on pose distances.
    """
    images_with_poses = [image for image in images if hasattr(image, "cam_from_world")]
    translations = np.array([image.cam_from_world.translation for image in images_with_poses])
    rotations = np.array([image.cam_from_world.rotation.quat for image in images_with_poses])

    # Normalize quaternion vectors
    rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)

    # Compute distances
    # Euclidean distance for translations
    t_distances = np.linalg.norm(translations[:, np.newaxis] - translations, axis=-1)
    # Angular distance for quaternions
    q_distances = 2 * np.arccos(np.clip(np.abs(np.sum(rotations[:, np.newaxis] * rotations, axis=-1)), 0, 1))

    # Add a penalty for large rotation distances
    distances = t_distances + np.where(q_distances > max_rotation_distance, rotation_penalty, rotation_weight * q_distances)
    return distances


def save_cluster_plot(cluster_labels, output_path):
    """
    Saves a scatter plot of cluster assignments to disk.

    Args:
        cluster_labels (np.ndarray): Cluster labels for each image.
        output_path (str): File path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    indices = np.arange(len(cluster_labels))
    plt.scatter(indices, cluster_labels, c=cluster_labels, cmap="tab10", s=100)
    plt.title("DBSCAN Cluster Assignments")
    plt.xlabel("Image Index")
    plt.ylabel("Cluster Label")
    plt.xticks(indices)
    plt.savefig(output_path)
    plt.close()

def save_cluster_images(clusters, images, output_path, images_path):
    """
    Saves a single image containing grids of images for all clusters.

    Args:
        clusters (dict): Clusters with cluster labels as keys and lists of image IDs as values.
        images (list): List of image objects.
        output_path (str): File path to save the combined image.
        images_path (str): Path to the images directory.
    """
    import cv2
    import numpy as np
    import os

    # Create a mapping from image_id to image object for quick access
    image_id_to_image = {image.image_id: image for image in images}

    # Collect grids for each cluster
    cluster_grids = []
    max_cluster_grid_width = 0  # To keep track of the maximum cluster grid width

    for cluster_id, cluster_image_ids in clusters.items():
        if cluster_id == -1:
            cluster_name = "Noise"
        else:
            cluster_name = f"Cluster {cluster_id}"

        # Load images for the cluster
        cluster_images = []
        for image_id in cluster_image_ids:
            image = image_id_to_image.get(image_id)
            if image is not None:
                image_path = os.path.join(images_path, image.name)
                img = cv2.imread(image_path)
                if img is not None:
                    cluster_images.append(img)
                else:
                    logging.warning(f"Image {image.name} not found at {image_path}")
            else:
                logging.warning(f"Image with ID {image_id} not found in provided images")

        if not cluster_images:
            continue  # Skip empty clusters

        # Resize images to a common height while maintaining aspect ratio
        max_height = 200  # Adjust as needed
        resized_images = []
        for img in cluster_images:
            aspect_ratio = img.shape[1] / img.shape[0]
            new_height = max_height
            new_width = int(aspect_ratio * new_height)
            resized_img = cv2.resize(img, (new_width, new_height))
            resized_images.append(resized_img)

        # Determine grid size (rows and columns)
        num_images = len(resized_images)
        cols = min(5, num_images)  # Max 5 images per row
        rows = (num_images + cols - 1) // cols  # Ceiling division

        # Calculate width and height of the grid
        max_img_width = max(img.shape[1] for img in resized_images)
        grid_width = cols * max_img_width
        grid_height = rows * max_height + 50  # Add space for cluster label

        # Create a blank canvas for the cluster grid
        grid_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        # Add cluster label text at the top of the grid
        cv2.putText(grid_img, cluster_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Place images onto the grid
        for idx, img in enumerate(resized_images):
            row = idx // cols
            col = idx % cols
            y = row * max_height + 50  # Offset by 50 pixels for the cluster label
            x = col * max_img_width
            h, w = img.shape[:2]
            grid_img[y:y+h, x:x+w] = img

        cluster_grids.append(grid_img)

        # Update the maximum cluster grid width
        if grid_width > max_cluster_grid_width:
            max_cluster_grid_width = grid_width

    # Now, pad each cluster grid to have the same width
    for idx, grid in enumerate(cluster_grids):
        h, w = grid.shape[:2]
        if w < max_cluster_grid_width:
            # Pad the grid image to the right
            pad_width = max_cluster_grid_width - w
            padding = np.zeros((h, pad_width, 3), dtype=np.uint8)
            grid = np.hstack((grid, padding))
            cluster_grids[idx] = grid

    # Combine all cluster grids into a single image by stacking vertically
    combined_img = np.vstack(cluster_grids)

    # Save the combined image
    cv2.imwrite(output_path, combined_img)
    logging.info(f"Combined cluster images saved to {output_path}")


def save_projected_images_by_cluster(clusters, images, image_data_list, points3D, model_colors, point_ids, tracks, K, tracks_for_image, output_dir):
    """
    Saves images of each cluster in a separate folder with red dots marking the projected and filtered points.

    Args:
        clusters (dict): Clusters with cluster labels as keys and lists of image IDs as values.
        images (list): List of image objects.
        image_data_list (list): List of dictionaries containing per-image data.
        points3D (np.ndarray): Array of 3D points.
        model_colors (np.ndarray): Array of colors for each 3D point.
        point_ids (np.ndarray): Array of point IDs.
        tracks (dict): Mapping from point IDs to sets of image IDs.
        K (np.ndarray): Camera intrinsic matrix.
        tracks_for_image (dict): Mapping from image IDs to sets of point IDs visible in that image.
        output_dir (str): Base directory to save the images.
    """
    import os
    import cv2

    # Create a mapping from image_id to image_data
    image_data_dict = {data['image_id']: data for data in image_data_list}

    # Iterate over clusters
    for cluster_id, cluster_image_ids in clusters.items():
        if cluster_id == -1:
            cluster_name = "Noise"
        else:
            cluster_name = f"Cluster_{cluster_id}"

        # Create directory for the cluster
        cluster_dir = os.path.join(output_dir, cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)

        # Process images in the cluster
        for image_id in cluster_image_ids:
            image_data = image_data_dict.get(image_id)
            if image_data is None:
                logging.warning(f"Image data for image ID {image_id} not found.")
                continue

            # Load the image
            image_name = image_data['image_name']
            images_path = image_data['images_path']
            img_path = os.path.join(images_path, image_name)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Image {image_name} not found at {img_path}")
                continue

            # Project and filter points
            result = project_to_img_local(image_data, points3D, model_colors, point_ids, tracks, K, tracks_for_image)
            if result is None:
                logging.warning(f"No projected points for image {image_name}")
                continue

            points2D = result['points2D']

            # Overlay the projected points on the image
            for point in points2D:
                x, y = int(point[0]), int(point[1])
                cv2.circle(img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Red dots

            # Save the image
            output_image_path = os.path.join(cluster_dir, image_name)
            cv2.imwrite(output_image_path, img)
            logging.info(f"Saved projected image to {output_image_path}")

def project_to_img_local(image_data, points3D, model_colors, point_ids, tracks, K, tracks_for_image):
    """
    Projects 3D points onto a 2D image plane using pre-extracted data.

    Args:
        image_data (dict): A dictionary containing all necessary data for the image.
        points3D (np.ndarray): Array of 3D points.
        model_colors (np.ndarray): Array of colors for each 3D point.
        point_ids (np.ndarray): Array of point IDs.
        tracks (dict): Mapping from point IDs to sets of image IDs.
        K (np.ndarray): Camera intrinsic matrix.
        tracks_for_image (dict): Mapping from image IDs to sets of point IDs visible in that image.

    Returns:
        dict: Contains filtered 2D projected points and image dimensions.
    """
    try:
        image_id = image_data['image_id']
        image_name = image_data['image_name']
        cam_from_world = image_data['cam_from_world']
        use_track = image_data['use_track']
        use_color = image_data['use_color']
        color_tolerance = image_data['color_tolerance']
        images_path = image_data['images_path']

        # Extract R and T from cam_from_world
        extrinsic_matrix = cam_from_world
        R, T = extrinsic_matrix[:, :3], extrinsic_matrix[:, 3]
        R_vec = cv2.Rodrigues(R)[0]  # Convert rotation matrix to vector

        # Project 3D points to the 2D image plane
        projected_points2D = cv2.projectPoints(points3D, R_vec, T, K, None)[0].reshape(-1, 2)

        # Load image to get dimensions
        img_path = os.path.join(images_path, image_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image {image_name} not found at {img_path}.")
        height, width = img.shape[:2]

        # Filter points within image bounds
        x = projected_points2D[:, 0]
        y = projected_points2D[:, 1]
        in_bounds_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)

        if not np.any(in_bounds_mask):
            return None  # No points to process

        points2D_in_bounds = projected_points2D[in_bounds_mask]
        model_colors_in_bounds = model_colors[in_bounds_mask]
        point_ids_in_bounds = point_ids[in_bounds_mask]

        # Initialize valid_mask
        num_points = len(points2D_in_bounds)

        if use_color:
            # Extract pixel colors from the image
            coords = points2D_in_bounds.astype(int)
            projected_colors = img[coords[:, 1], coords[:, 0]] / 255.0

            # Compute color differences
            color_diff = np.linalg.norm(projected_colors - model_colors_in_bounds, axis=1)
            color_mask = color_diff <= color_tolerance
        else:
            color_mask = np.zeros(num_points, dtype=bool)

        if use_track:
            # Track membership check
            visible_point_ids = tracks_for_image[image_id]
            track_mask = np.in1d(point_ids_in_bounds, list(visible_point_ids), assume_unique=True)
        else:
            track_mask = np.zeros(num_points, dtype=bool)

        # Combine masks
        valid_mask = color_mask | track_mask

        # Return filtered points
        filtered_points2D = points2D_in_bounds[valid_mask]
        return {
            'image_id': image_id,
            'points2D': filtered_points2D,
            'height': height,
            'width': width
        }
    except Exception as e:
        #print("Local projection error fix here")
        print(f"Error processing image {image_name}: {e}")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detect sparsity in images using COLMAP reconstruction.")
    parser.add_argument("--images_path", type=str, default="../data/alameda/images",
                        help="Path to the images directory.")
    parser.add_argument("--colmap_path", type=str, default="../data/alameda/colmap/sparse/0",
                        help="Path to the COLMAP model directory.")
    parser.add_argument("--density_threshold", type=float, default=0.00000001,
                        help="Density threshold for high-density area detection.")
    parser.add_argument("--mask_path", type=str, default=None,
                        help="Path to save the high-density area mask for each image.")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute density statistics even if a CSV file exists.")
    parser.add_argument("--clustering_method", type=str, default="poses",
                        help="Clustering method: 'shared_points' or 'poses'")
    parser.add_argument("--eps", type=float, default=0.5,
                        help="DBSCAN epsilon parameter.")
    parser.add_argument("--min_samples", type=int, default=10,
                        help="DBSCAN min_samples parameter.")
    parser.add_argument("--cluster-images", type=str, default="all",
                        help="Cluster 'all'/'sparse' images")
    parser.add_argument("--cluster-output", type=str, default="clusters.csv",
                        help="Path to save the cluster IDs.")
    parser.add_argument("--sparsity-threshold", type=float, default=0.15,
                        help="Threshold for detecting sparse images.")
    # parser.add_argument("--n-clusters", type=int, default=5,
    #                     help="Number of clusters to generate.")
    args = parser.parse_args()

    if args.clustering_method not in ("shared_points", "poses"):
        raise ValueError(f"Invalid clustering method: {args.clustering_method}")
    if args.cluster_images not in ("all", "sparse"):
        raise ValueError(f"Invalid cluster images option: {args.cluster_images}")

    logging.basicConfig(level=logging.INFO)
    reconstruction = pycolmap.Reconstruction(args.colmap_path)
    print(reconstruction.summary())

    # Ensure that reconstruction is properly initialized
    if reconstruction is None:
        raise ValueError("Reconstruction object is not initialized.")

    # Prepare global data structures
    # Extract camera intrinsics
    camera = reconstruction.cameras[next(iter(reconstruction.cameras))]
    K = np.array([
        [camera.focal_length_x, 0, camera.principal_point_x],
        [0, camera.focal_length_y, camera.principal_point_y],
        [0, 0, 1]
    ], dtype=np.float32)

    # Extract 3D points, colors, and track information
    points3D_list = []
    model_colors_list = []
    point_ids_list = []
    tracks = {}

    for pid, point in reconstruction.points3D.items():
        points3D_list.append(point.xyz)
        model_colors_list.append(point.color / 255.0)  # Normalize color
        point_ids_list.append(pid)
        # Extract track information
        tracks[pid] = set(element.image_id for element in point.track.elements)

    points3D = np.array(points3D_list, dtype=np.float32)
    model_colors = np.array(model_colors_list, dtype=np.float32)
    point_ids = np.array(point_ids_list)
    logging.info(f"Extracted {len(points3D)} 3D points.")

    # Extract per-image data
    image_data_list = []
    tracks_for_image = {}

    for image in reconstruction.images.values():
        image_id = image.image_id
        image_name = image.name
        cam_from_world = image.cam_from_world.matrix()

        # Extract track information for the image
        visible_point_ids = set(p.point3D_id for p in image.points2D if p.has_point3D())
        tracks_for_image[image_id] = visible_point_ids

        image_data = {
            'image_id': image_id,
            'image_name': image_name,
            'cam_from_world': cam_from_world,
            'use_track': True,
            'use_color': True,
            'color_tolerance': 0.1,
            'images_path': args.images_path,
            'density_threshold': args.density_threshold,
            'mask_path': args.mask_path if args.mask_path is not None else None
        }
        image_data_list.append(image_data)
    logging.info(f"Extracted {len(image_data_list)} images.")

    if not os.path.exists("alameda_density_results.csv") or args.recompute:
        logging.info("Processing images to compute density statistics")
        # Package the global data into a tuple
        global_data = (points3D, model_colors, point_ids, tracks, K, tracks_for_image)

        # Prepare a list to store results
        results = []

        num_processes = multiprocessing.cpu_count() - 2
        logging.info(f"Using {num_processes} worker processes")

        # Use ProcessPoolExecutor with initializer
        with ProcessPoolExecutor(max_workers=num_processes, initializer=init_worker, initargs=(global_data,)) as executor:
            # Submit all tasks to the executor
            futures = {executor.submit(process_image, image_data): image_data['image_id'] for image_data in image_data_list}

            # Collect results as they complete
            for future in tqdm(as_completed(futures), total=len(futures)):
                image_id = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as exc:
                    print(f'Image {image_id} generated an exception: {exc}')
        logging.info(f"Processed {len(results)} images.")

        # Create a DataFrame from the results
        df = pd.DataFrame(results)
        print(df)

        # Save the DataFrame to a CSV file
        df.to_csv("alameda_density_results.csv", index=False)
    else:
        logging.info("Loading precomputed density statistics")
        df = pd.read_csv("alameda_density_results.csv")

    logging.info(f"Start clustering {args.cluster_images} images based on {args.clustering_method}")

    # Obtain the images with the lowest density
    lowest_density_images = df[df["fraction_high_density_area"] < args.sparsity_threshold]["image_id"]
    logging.info(f"Detected {len(lowest_density_images)} images with a density below {args.sparsity_threshold}")

    if args.cluster_images == "sparse":
        ims = [reconstruction.images[image_id] for image_id in lowest_density_images]
    elif args.cluster_images == "all":
        ims = list(reconstruction.images.values())

    # Compute the overlap matrix
    logging.info("Computing overlap matrix...")
    overlap_matrix = compute_overlap_matrix(ims, method=args.clustering_method)

    # Cluster parameters
    eps = args.eps
    min_samples = args.min_samples
    clusters, cluster_labels = cluster_images(overlap_matrix, ims, eps=eps, min_samples=min_samples)

    # Save the cluster IDs to a CSV file
    cluster_df = pd.DataFrame({
        "image_id": [image.image_id for image in ims],
        "cluster_id": cluster_labels
    })
    cluster_df.to_csv(args.cluster_output, index=False)
    logging.info(f"Computed {len(clusters)} clusters.")

    if args.cluster_images == "sparse":
        # Save the cluster images
        output_path = "alameda_cluster_images.png"
        save_cluster_images(clusters, ims, output_path, images_path=args.images_path)
        logging.info(f"Cluster images saved to {output_path}")

        # Prepare image_data_list for the images in 'ims'
        ims_image_ids = [image.image_id for image in ims]
        image_data_list_filtered = [image_data for image_data in image_data_list if image_data['image_id'] in ims_image_ids]

        # Save images with projected points for each cluster
        output_dir = "clusters_with_projections"
        save_projected_images_by_cluster(
            clusters,
            ims,
            image_data_list_filtered,
            points3D,
            model_colors,
            point_ids,
            tracks,
            K,
            tracks_for_image,
            output_dir
        )
        logging.info(f"Cluster images with projected points saved to {output_dir}")
    elif args.cluster_images == "all":
        # Detect which clusters contain the most sparse images
        sparse_clusters = []
        for cluster_id, image_ids in clusters.items():
            num_sparse_images = len(set(image_ids).intersection(lowest_density_images))
            sparse_clusters.append((cluster_id, num_sparse_images, len(image_ids)))
        sparse_clusters.sort(key=lambda x: x[1], reverse=True)
        logging.info(f"Number of sparse and total images per cluster: {sparse_clusters}")
