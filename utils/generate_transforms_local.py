import json

#read the json file
path_to_transforms = '/home/team5/project/data/alameda/transforms.json'
with open(path_to_transforms) as f:
    transforms_data = json.load(f)

#extract the frames
transforms_data['frames'] = transforms_data['frames'][:365]

# Save the updated JSON file
with open("/home/team5/project/data/alameda/transforms_365.json", "w") as f:
    json.dump(transforms_data, f, indent=4)