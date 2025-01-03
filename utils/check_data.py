import json
import os

data_dir = "data/alameda/images_2"
print(len(os.listdir(data_dir)))

json_file = "data/alameda/transforms.json"
json_data = json.load(open(json_file))
print(len(json_data["frames"]))


#check for corrupt images
import cv2
for image in os.listdir(data_dir):
    try:
        img = cv2.imread(os.path.join(data_dir, image))
        if img.shape != (793, 1394, 3):
            print(img.shape)
    except:
        print(image)
