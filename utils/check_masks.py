from PIL import Image
import os

# Convert all masks to 1-channel images (grayscale)
def convert_masks_to_grayscale(masks_path, output_path):
    for mask_path in os.listdir(masks_path):
        #print(mask_path)
        mask = Image.open(os.path.join(masks_path, mask_path))
        mask = mask.convert('L')
        mask.save(output_path + mask_path.split('/')[-1])

def main():
    masks_path = "/home/team5/project/data/bicycle/processed/masks_8"
    output_path = "/home/team5/project/data/bicycle/processed/masks_8/"
    convert_masks_to_grayscale(masks_path, output_path)

def check_single_image(mask_path):
    mask = Image.open(mask_path)
    print(mask.mode)

if __name__ == "__main__":
    main()
    #masks_path = "/home/team5/project/data/bicycle/processed/masks_8/frame_00001.JPG"
    #check_single_image(masks_path)