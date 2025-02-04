from prune_plys import merge_and_prune_mipnerf
import os

if __name__ == "__main__":
    data_dir = "/home/team5/project/data/360_v2"
    parh_to_nerf_pcs = "/home/team5/project/mipnerf_nerfs/point_clouds"
    num_points = [50000, 100000, 150000]
    for scene in os.listdir(data_dir):
        scene_dir = os.path.join(data_dir, scene)
        sfm_path = os.path.join(scene_dir, "processed", "sparse_pc.ply")
        print(sfm_path)
        for nerf_pc in os.listdir(parh_to_nerf_pcs):
            nerf_pc_path = os.path.join(parh_to_nerf_pcs, nerf_pc)
            if scene in nerf_pc:
                output_dir = os.path.join(scene_dir, "plys")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                for num_point in num_points:
                    base_name = nerf_pc.rsplit('_', 1)[0]
                    new_filename = f"merged_sfm_{base_name}_{num_point}pts.ply"
                    print("Output File:", new_filename)
                    output_path = os.path.join(output_dir, new_filename)
                    merge_and_prune_mipnerf(sfm_path, nerf_pc_path, num_point, output_path)
                    
                


