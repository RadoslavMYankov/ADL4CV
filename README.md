# Enhancing 3D Gaussian Splatting Optimization with NeRF Priors

## Overview

This project aims to enhance 3D Gaussian Splatting (3DGS) by leveraging priors from pretrained Neural Radiance Fields (NeRFs).

## Getting Started
### Prepare the data
Unposed images - run COLMAP following the [nerfstudio guide](https://docs.nerf.studio/quickstart/custom_dataset.html)

If the bin files with camera poses are available 
run `prepare_data.py`

```bash
python detect_sparsity.py \
    --input-path /data/sparse/0
    --output-path /data/
````


### Running the Sparsity Detection Script - Only Relevant for Local NeRF Pipeline

The first step in the pipeline is executing `detect_sparsity.py`, which analyzes the sparsity of images using COLMAP reconstruction.

#### Example Usage

```bash
python detect_sparsity.py \
    --images_path ../data/alameda/images \
    --colmap_path ../data/alameda/colmap/sparse/0 \
    --density_threshold 0.00000001 \
    --mask_path ../output/masks \
    --recompute \
    --clustering_method poses \
    --eps 0.5 \
    --min_samples 10 \
    --cluster-images all \
    --cluster-output clusters.csv \
    --sparsity-threshold 0.15
```

### Script Description

`detect_sparsity.py` identifies sparse regions in the SfM initialization to improve 3DGS initialization. It performs the following tasks:

- Loads COLMAP sparse reconstruction data to assess 3D point density.
- Computes density statistics for each image and identifies sparse regions.
- Optionally clusters images using DBSCAN based on poses or shared 3D points.
- Saves clustering results and sparsity masks for further processing.

### Create the Local Clusters

We use the generated clusters as guidance and manually select additional training images for the local clusters (we need to get enough training views for accurate depth estimation). Having selected the frames,
we run `generate_clusters.py` which works on a sorted array of all frames and generates the training JSON files for the individual clusters.

### Training the NeRF Models (Applies to both Global and Local Pipelines) 

Execute either  `train_local_nerfs.py` (local) or  `train_nerfs_mipnerf.py` (global nerfs). We also export the point clouds in this step.  

Example usage: 

```bash
python train_local_nerfs.py \
    --project-name alameda-local-nerfs \
    --data ../data/alameda/transforms.json \
    --output ../output/local_nerfs \
    --clusters ../output/clusters.csv \
    --min_sparse_images 5 \
    --max_clusters 5 \
    --max_num_iterations 100000 \
    --save_plys \
    --num_points 50000
```
`train_local_nerfs.py` trains NeRF models on local clusters based on identified sparse/dense regions:

 - Loads the dataset and image cluster information.

 - Filters clusters based on the minimum number of sparse images. (not relevant for manual clusters)

 - Trains a NeRF model for each selected cluster using Nerfstudio.

 - Optionally exports point clouds as PLY files.

### Merging Point Clouds

`merge_local_plys.py` with arguments 
nerf_inputs - containing the nerf point clouds
sfm_input - path to the SfM pointcloud
output_path

### Traing 3DGS

use `train_3dgs_final.py` 

Example Usage:

```bash
python train_3dgs.py \
    --project-name alameda-3dgs \
    --data ../data/alameda/transforms.json \
    --output ../output/3dgs_models \
    --plys ../output/plys \
    --max_num_iterations 30000
```


### Miscelaneous 

the visualization tools can be found in `utils\visualizations_final`

cleaning the NeRFs we use [Optimizing the Unseen](https://github.com/leosegre/Optimize-the-Unseen). Please refer to the paper and the documentation for further information.

#### Contributors
Radoslav Yankov
Lucas Schnack
