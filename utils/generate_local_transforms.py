import json
import os
import numpy as np

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

    for cluster in clusters:
        output_file = f"/home/team5/project/data/alameda/transforms_cluster_{cluster}.json"
        img_names = clusters_dict[cluster]
        
        generate_transforms_cluster(transforms_data, output_file, img_names)
