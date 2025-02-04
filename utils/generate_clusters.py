import json
import os
import numpy as np
import pandas as pd

def generate_transforms_cluster(transforms_data, output_file, img_names):
    # Filter the frames list to only include entries in the cluster
    transforms_new = transforms_data.copy()
    transforms_new['frames'] = [frame for frame in transforms_data['frames']\
                                  if os.path.basename(frame.get('file_path'))\
                                  in img_names]

    # Save the updated JSON file
    with open(output_file, "w") as f:
        json.dump(transforms_new, f, indent=4)

    print(f"Updated transforms.json saved to {output_file}")

def read_images(images_path):
    image_names = os.listdir(images_path)
    return image_names

if __name__ == "__main__":
    clusters = [str(i) for i in range(1, 6)]
    #clusters = ["42"]	
    images_path = "/home/team5/project/data/nyc/images"
    images_names = sorted(read_images(images_path))

    # Define the clusters
    image_ids = {}
    image_ids['1'] = [i for i in range(600, 647)]
    image_ids['2'] = [i for i in range(125, 179)]
    image_ids['3'] = [i for i in range(693, 891)]
    image_ids['4'] = [i for i in range(901, 970)]
    image_ids['5'] = [i for i in range(318, 431)]
    clusters_dict = {}
    clusters_dict['1'] = images_names[600:647]
    clusters_dict['2'] = images_names[125:179]
    clusters_dict['3'] = images_names[693:891]
    clusters_dict['4'] = images_names[901:970]
    clusters_dict['5'] = images_names[318:431]
    #clusters_dict['42'] = images_names[:1500]

    # Load transforms.json
    with open("/home/team5/project/data/nyc/transforms.json", "r") as f:
        transforms_data = json.load(f)

    all_clusters_df = []
    for cluster in clusters:
        output_file = f"/home/team5/project/data/nyc/transforms_cluster_{cluster}.json"
        
        img_names = clusters_dict[cluster]
        
        generate_transforms_cluster(transforms_data, output_file, img_names)

        # Print lengths for debugging
        print(f"Cluster {cluster}:")
        print(f"  Number of image names: {len(img_names)}")
        print(f"  Number of image IDs: {len(image_ids[cluster])}")

        cluster_df = pd.DataFrame({"image_name": img_names, "image_id": image_ids[cluster], "cluster_id": int(cluster), "is_sparse": True})
        cluster_df.to_csv(f"/home/team5/project/data/nyc/cluster_{cluster}.csv", index=False)
        all_clusters_df.append(cluster_df)

    all_clusters_df = pd.concat(all_clusters_df)
    all_clusters_df.to_csv("/home/team5/project/data/nyc/all_clusters.csv", index=False)
