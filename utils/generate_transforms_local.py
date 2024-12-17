import json

def generate_transforms_cluster(cluster_ids, transforms_data, output_file):
    # Filter the frames list to only include entries in the cluster
    transforms_data['frames'] = [frame for frame in transforms_data['frames'] if frame.get('colmap_im_id', float('inf')) in cluster_ids]

    # Save the updated JSON file
    with open(output_file, "w") as f:
        json.dump(transforms_data, f, indent=4)

    print(f"Updated transforms.json saved to {output_file}")

def generate_transforms_train(transforms_data, output_file, interval=10):
    # Filter the frames list to skip every interval frames
    transforms_data['frames'] = transforms_data['frames'][::interval]

    # Save the updated JSON file
    with open(output_file, "w") as f:
        json.dump(transforms_data, f, indent=4)

def generate_transforms_val(transforms_path, output_file, interval=10):
    # Load the transforms.json file
    frames = []
    for cluster in range(8):
        transform_path = transforms_path + f"transforms_cluster_{cluster}.json"
        with open(transform_path, "r") as f:
            transforms_data = json.load(f)
            frames.extend(transforms_data['frames'][::interval])

    # Combine the frames from each cluster
    transforms_data['frames'] = frames

    # Save the updated JSON file
    with open(output_file, "w") as f:
        json.dump(transforms_data, f, indent=4)

if __name__ == "__main__":
    # Load clusters
    '''with open("/home/team5/project/data/alameda/alameda_clusters.json", "r") as f:
        clusters = json.load(f)
        ids = clusters["7"]'''

    # Load transforms.json
    '''with open("/home/team5/project/data/alameda/transforms.json", "r") as f:
        transforms_data = json.load(f)'''

    output_file = "/home/team5/project/data/alameda/transforms_clusters_1_val.json"
    path_to_transforms = "/home/team5/project/data/alameda/"

    generate_transforms_val(path_to_transforms, output_file)

    #generate_transforms_cluster(ids, transforms_data, output_file)
    #generate_transforms_train(transforms_data, output_file)