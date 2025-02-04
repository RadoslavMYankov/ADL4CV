import json
import os

def merge_transforms(files, output_file):
    merged_data = None

    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            if merged_data is None:
                merged_data = data
            else:
                merged_data['frames'].extend(data['frames'])

    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=4)

    print(f"Merged transforms saved to {output_file}")

if __name__ == "__main__":
    # Paths to the input JSON files
    transforms_1 = "/home/team5/project/data/zipnerf/nyc/transforms_cluster_1.json"
    transforms_2 = "/home/team5/project/data/zipnerf/nyc/transforms_cluster_2.json"
    transforms_3 = "/home/team5/project/data/zipnerf/nyc/transforms_cluster_3.json"
    transforms_4 = "/home/team5/project/data/zipnerf/nyc/transforms_cluster_4.json"
    transforms_5 = "/home/team5/project/data/zipnerf/nyc/transforms_cluster_5.json"

    # Path to the output JSON file
    output_file = "/home/team5/project/data/zipnerf/nyc/transforms_merged_clusters.json"

    # Merge the transforms
    merge_transforms([transforms_1, transforms_2, transforms_3, transforms_4, transforms_5], output_file)