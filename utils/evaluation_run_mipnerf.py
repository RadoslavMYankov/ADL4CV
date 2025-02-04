"python utils/train_3dgs_rado.py --project-name bicycle_3dgs --data data/bicycle/processed/ --output bicycle_3dgs_compare_100k_points_init --plys data/bicycle/plys/ --max_num_iterations 30000"


import os
import subprocess

# Define the base directory containing the scenes
base_directory = "/home/team5/project/data/360_v2"

# Iterate over each scene in the base directory
for scene in os.listdir(base_directory):
    print(f"Evaluating scene: {scene}")
    if os.path.isdir(os.path.join(base_directory, scene)):
        command = f"python utils/train_3dgs_rado.py --project-name {scene}_3dgs --data data/360_v2/{scene}/processed/ --output {scene}_3dgs_evaluation --plys data/{scene}/plys/ --max_num_iterations 30000"
        subprocess.run(command, shell=True, check=True)