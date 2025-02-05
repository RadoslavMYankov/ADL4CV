import os
import cv2
import numpy as np

def create_video_from_images(image_dir, output_video_path, fps=1/3, transition_duration=1):
    """
    Create a video from images in a directory with sliding transitions.
    
    Parameters:
    - image_dir: Directory containing the images.
    - output_video_path: Path to save the output video.
    - fps: Frames per second (default is 1 frame per 3 seconds).
    - transition_duration: Duration of the sliding transition in seconds.
    """
    images = [img for img in os.listdir(image_dir) if img.endswith(".png") or img.endswith(".jpg")]
    images.sort()  # Ensure the images are in order

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' or 'mp4v' for .avi or .mp4 files
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    num_frames_per_image = int(fps * 3)  # Number of frames to display each image
    num_transition_frames = int(fps * transition_duration)  # Number of frames for the transition

    for i in range(len(images) - 1):
        image_path = os.path.join(image_dir, images[i])
        next_image_path = os.path.join(image_dir, images[i + 1])
        frame = cv2.imread(image_path)
        next_frame = cv2.imread(next_image_path)

        # Display the current image for the specified duration
        for _ in range(num_frames_per_image):
            video.write(frame)

        # Create sliding transition frames
        for j in range(num_transition_frames):
            alpha = j / num_transition_frames
            transition_frame = cv2.addWeighted(frame, 1 - alpha, next_frame, alpha, 0)
            video.write(transition_frame)

    # Display the last image for the specified duration
    last_image_path = os.path.join(image_dir, images[-1])
    last_frame = cv2.imread(last_image_path)
    for _ in range(num_frames_per_image):
        video.write(last_frame)

    video.release()
    print(f"Video saved to {output_video_path}")

if __name__ == "__main__":
    scene = "berlin"
    image_directory = f"{scene}//annotated_images"
    output_video = f"comparison_video.mp4"

    create_video_from_images(image_directory, output_video)