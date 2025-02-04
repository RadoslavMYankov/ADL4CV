import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def plot_triplets_with_match_images(root_dir, merged_csv, output_dir):
    """
    1) Looks in each subdirectory of root_dir (each subdirectory is a 'method').
    2) Collects only images that have 'match' in the filename, e.g. '25700_kitchen_match_1.png'.
    3) Groups images by (scene, match_index), so e.g. scene='kitchen', match_index=1.
    4) If multiple methods have the same (scene, match_index), it creates a triplet side-by-side.
    5) Saves each triplet as {scene}_match_{match_index}_triplet.png in output_dir.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load the merged CSV as a Pandas DataFrame
    df = pd.read_csv(merged_csv)

    # Find all method subdirectories
    methods = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    # Use a regex to parse filenames that match, e.g. "25700_kitchen_match_1.png"
    pattern = re.compile(r"^(\d+)_(\w+)_match_(\d+)\.png$")

    # Dictionary keyed by (scene, match_idx) mapping to {method_name: (filepath, step)}
    images_dict = {}

    for method in methods:
        method_path = os.path.join(root_dir, method)
        for filename in os.listdir(method_path):
            # Only proceed if 'match' is in the filename
            if "match" in filename:
                match_obj = pattern.match(filename)
                if match_obj:
                    step, scene, match_idx = match_obj.groups()
                    key = (scene, match_idx)
                    images_dict.setdefault(key, {})[method] = (os.path.join(method_path, filename), int(step))

    # Create triplets where multiple methods have the same key
    for (scene, match_idx), method_files in images_dict.items():
        # You can require exactly 3 methods, or at least 2, etc. Here we just create
        # a triplet if 3 methods are present:
        if len(method_files) < 2:
            # Skip if we don't have 3 matching images for this key
            continue

        # Sort the methods alphabetically (or define a custom order if you prefer)
        sorted_methods = sorted(method_files.keys())

        # Create a figure with as many subplots as there are methods in this group + 1 for GT
        fig, axes = plt.subplots(1, len(sorted_methods) + 1, figsize=(5 * (len(sorted_methods) + 1), 5))

        # Use the left half of the first image as the Ground Truth
        first_method = sorted_methods[0]
        first_filepath, first_step = method_files[first_method]
        first_img = Image.open(first_filepath)
        w, h = first_img.size
        half_w = w // 2
        left_img = first_img.crop((0, 0, half_w, h))

        # Plot the Ground Truth image
        axes[0].imshow(left_img)
        axes[0].set_title("Ground Truth", pad=5)  # Add padding above the title
        axes[0].axis("off")

        # Plot the Reconstruction images
        for ax, method in zip(axes[1:], sorted_methods):
            filepath, step = method_files[method]
            img = Image.open(filepath)
            right_img = img.crop((half_w, 0, w, h))

            # Find the matching row in the CSV
            row = df[(df["Method"] == method) & (df["Step"] == step)]
            if row.empty:
                print("Warning: No matching row found for", method, step)
                # No matching row found, skip
                continue

            # Get the training time
            training_time = row["Training Time"].values[0]

            # Plot the images
            ax.imshow(right_img)
            ax.set_title(f"{method} ({training_time:.0f}s)", pad=5)  # Add padding above the title
            ax.axis("off")

        plt.subplots_adjust(wspace=0, hspace=0)  # Remove space between images
        plt.tight_layout(pad=0)
        out_filename = f"{scene}_match_{match_idx}_triplet.png"
        out_filepath = os.path.join(output_dir, out_filename)
        plt.savefig(out_filepath, bbox_inches="tight", pad_inches=0)
        plt.close()

if __name__ == "__main__":
    scene = "bicycle"

    root_directory = f"{scene}//selected_images"
    merged_csv_file = f"{scene}//tensorboard//merged.csv"
    output_directory = f"{scene}//annotated_images"

    plot_triplets_with_match_images(root_directory, merged_csv_file, output_directory)
    print("Images saved in:", output_directory)