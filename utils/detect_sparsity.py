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

IMAGES_PATH = "../data/alameda/images"
COLMAP_PATH = "../data/alameda/colmap/sparse/0"

# Define the density threshold
density_threshold = 0.00000001


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

        # Extract R and T from cam_from_world
        extrinsic_matrix = cam_from_world
        R, T = extrinsic_matrix[:, :3], extrinsic_matrix[:, 3]
        R_vec = cv2.Rodrigues(R)[0]  # Convert rotation matrix to vector

        # Project 3D points to the 2D image plane
        projected_points2D = cv2.projectPoints(points3D, R_vec, T, K, None)[0].reshape(-1, 2)

        # Load image
        img_path = os.path.join(IMAGES_PATH, image_name)
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

        # Generate the density map for the projected points
        smoothed_density_map = density_map(points2D, (height, width), sigma=10)

        # Calculate the fraction of the image area with density above the threshold
        high_density_area = np.sum(smoothed_density_map > density_threshold)
        total_area = height * width
        fraction_high_density = high_density_area / total_area

        num_projected_points = points2D.shape[0]

        return {
            "image_id": image_data['image_id'],
            "num_projected_points": num_projected_points,
            "fraction_high_density_area": fraction_high_density
        }
    except Exception as e:
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


def compute_overlap_matrix(images):
    """
    Computes the overlap matrix for a list of images based on shared 3D points.

    Args:
        images (list): List of image objects.

    Returns:
        np.ndarray: Overlap matrix.
    """
    num_images = len(images)
    overlap_matrix = np.zeros((num_images, num_images), dtype=np.float32)

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
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = {executor.submit(compute_overlap, task): task[:2] for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing overlaps"):
            i, j, overlap = future.result()
            overlap_matrix[i, j] = overlap_matrix[j, i] = overlap

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

def save_cluster_images(clusters, images, output_path):
    """
    Saves a single image containing grids of images for all clusters.

    Args:
        clusters (dict): Clusters with cluster labels as keys and lists of image IDs as values.
        images (list): List of image objects.
        output_path (str): File path to save the combined image.
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
                image_path = os.path.join(IMAGES_PATH, image.name)
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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    reconstruction = pycolmap.Reconstruction(COLMAP_PATH)
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
            'use_track': True,  # Set your flags as needed
            'use_color': True,
            'color_tolerance': 0.05
        }
        image_data_list.append(image_data)
    logging.info(f"Extracted {len(image_data_list)} images.")

    if not os.path.exists("alameda_density_results.csv"):
        logging.info("Processing images to compute density statistics")
        # Package the global data into a tuple
        global_data = (points3D, model_colors, point_ids, tracks, K, tracks_for_image)

        # Prepare a list to store results
        results = []

        num_processes = multiprocessing.cpu_count()
        logging.info(f"Using {num_processes} worker processes")

        # Use ProcessPoolExecutor with initializer
        with ProcessPoolExecutor(max_workers=num_processes, initializer=init_worker, initargs=(global_data,)) as executor:
            # Submit all tasks to the executor
            futures = {executor.submit(process_image, image_data): image_data['image_id'] for image_data in image_data_list}

            # Collect results as they complete
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
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

    logging.info("Start clustering sparse images based on density")
    # Obtain the images with the lowest density
    num_images = 80
    lowest_density_images = df.nsmallest(num_images, "fraction_high_density_area")["image_id"].values
    ims = [reconstruction.images[image_id] for image_id in lowest_density_images]

    # Compute the overlap matrix
    logging.info("Computing overlap matrix...")
    overlap_matrix = compute_overlap_matrix(ims)

    # Cluster parameters
    eps = 0.75
    min_samples = 2
    clusters, cluster_labels = cluster_images(overlap_matrix, ims, eps=eps, min_samples=min_samples)

    logging.info(f"Computed {len(clusters)} clusters.")

    # Save the cluster images
    output_path = "alameda_cluster_images.png"
    save_cluster_images(clusters, ims, output_path)
    logging.info(f"Cluster images saved to {output_path}")
