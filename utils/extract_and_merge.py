import json
import open3d as o3d
import argparse 
import merge_plys
from train_local_nerfs import export_pointcloud

# DEPRECATED

def main():
    parser = argparse.ArgumentParser(
        description="Automate pc extraction and merging."
    )
    parser.add_argument('--load_config', type=str, default='nerf_config.yml', help='Path to the input NeRF model directory.')
    parser.add_argument('--extract_points', type=int, nargs='+', default=[100000], help='number of points to be extracted.')
    parser.add_argument('--total_points', type=int, nargs='+', default=[100000], help='number of points to be extracted.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Path to save the exported point cloud.')
    parser.add_argument('--path_to_sfm', type=str, default='sparse_pc.ply', help='Path to the SfM pointcloud.')
    parser.add_argument('--merged_output_path', type=str, default='bicycle/processed/', help='Path to save the merged point cloud.')

    args = parser.parse_args()
    for num_points in args.num_points:
        out_path = f'{args.output_dir}/point_cloud_{num_points}'
        export_pointcloud(args.load_config, out_path, num_points)

        export_path = f'{args.output_dir}/point_cloud_{num_points}/point_cloud.ply'
        merged_pcd = merge_plys.merge_pcs(args.path_to_sfm, export_path, max_points=args.total_points)
        output_ply_file_path = f'{args.merged_output_path}/merged_point_cloud_{num_points}.ply'
        o3d.io.write_point_cloud(output_ply_file_path, merged_pcd)


if __name__ == '__main__':
    main()