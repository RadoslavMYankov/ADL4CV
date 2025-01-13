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
    images_path = "/home/team5/project/data/alameda/images"
    images_names = sorted(read_images(images_path))

    # Define the clusters
    image_ids = {}
    image_ids['1'] = [i for i in range(364)]
    image_ids['2'] = [i for i in range(545, 589)]
    image_ids['3'] = [i for i in range(624, 864)]
    image_ids['4'] = [i for i in range(893, 1105)]
    image_ids['5'] = [i for i in range(1530, 1613)]
    clusters_dict = {}
    clusters_dict['1'] = images_names[:364]
    clusters_dict['2'] = images_names[545:589]
    clusters_dict['3'] = images_names[624:864]
    clusters_dict['4'] = images_names[893:1105]
    clusters_dict['5'] = images_names[1530:1613]
    #clusters_dict['42'] = images_names[:1500]

    # Load transforms.json
    with open("/home/team5/project/data/alameda/transforms.json", "r") as f:
        transforms_data = json.load(f)

    all_clusters_df = []
    for cluster in clusters:
        output_file = f"/home/team5/project/data/alameda/transforms_cluster_{cluster}.json"
        img_names = clusters_dict[cluster]
        
        generate_transforms_cluster(transforms_data, output_file, img_names)

        cluster_df = pd.DataFrame({"image_name": img_names, "image_id": image_ids[cluster], "cluster_id": int(cluster), "is_sparse": True})
        cluster_df.to_csv(f"/home/team5/project/data/alameda/cluster_{cluster}.csv", index=False)
        all_clusters_df.append(cluster_df)
        print(len(img_names))

    all_clusters_df = pd.concat(all_clusters_df)
    all_clusters_df.to_csv("/home/team5/project/data/alameda/all_clusters.csv", index=False)
