import os
import subprocess

# Define the base directory containing the scenes
base_directory = "/home/team5/project/data/360_v2"

# Iterate over each scene in the base directory
for scene in os.listdir(base_directory):
    scene_path = os.path.join(base_directory, scene)
    print(f"Processing scene: {scene_path}")
    if os.path.isdir(scene_path):
        output_dir = os.path.join(scene_path, "processed")
        command = f"ns-process-data images --data {scene_path} --output-dir {output_dir}"
        subprocess.run(command, shell=True, check=True)