import os
import io
from PIL import Image
from tensorboard.backend.event_processing import event_accumulator

def export_images(logdir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load the event files in the given directory, focusing on images
    ea = event_accumulator.EventAccumulator(logdir, size_guidance={
        event_accumulator.IMAGES: 0
    })
    ea.Reload()

    # Get all image tags from the log
    image_tags = ea.Tags()['images']

    # Export each image for each tag
    for tag in image_tags:
        images = ea.Images(tag)
        for i, image_event in enumerate(images):
            step = image_event.step
            # Decode the raw image bytes
            img = Image.open(io.BytesIO(image_event.encoded_image_string))
            # Construct the output filename
            filename = f"{tag.replace('/', '_')}_step{step}_{i}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)

if __name__ == "__main__":
    log_directory = "/home/team5/project/mipnerf_evaluations/counter/processed/splatfacto/merged_sfm_counter_1000its_150000pts_30000_its" # Directory containing your event file
    output_directory = "qualitative_counter/Ours"     # Directory to save the exported images
    export_images(log_directory, output_directory)