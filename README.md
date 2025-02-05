Running the Sparsity Detection Script

The first step in the pipeline is executing detect_sparsity.py, which analyzes the sparsity of images using COLMAP reconstruction.

Example Usage

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

Script Description

detect_sparsity.py identifies sparse regions in the SfM initialization to improve 3DGS initialization. It performs the following tasks:

Loads COLMAP sparse reconstruction data to assess 3D point density.

Computes density statistics for each image and identifies sparse regions.

Optionally clusters images using DBSCAN based on poses or shared 3D points.

Saves clustering results and sparsity masks for further processing.