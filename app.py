import os
from PIL import Image
import numpy as np
import tensorflow as tf

def augment_image(image):
    """Apply augmentation to a single image using TensorFlow."""
    # Randomly flip the image horizontally
    image = tf.image.random_flip_left_right(image)
    # Randomly adjust brightness
    image = tf.image.random_brightness(image, max_delta=0.1)
    # Randomly adjust contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # Random rotation (in degrees)
    k = np.random.randint(0, 4)  # Rotate 0, 90, 180, or 270 degrees
    image = tf.image.rot90(image, k=k)
    
    return image.numpy()

def augment_images_in_folder(folder_path, output_folder, augment_count=5):
    """Augment images in a given folder and save them to an output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for subdir, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')):
                file_path = os.path.join(subdir, file)
                image = Image.open(file_path)
                image = image.convert("RGB")  # Ensure it's in RGB format
                image_np = np.array(image)

                for i in range(augment_count):
                    # Augment the image
                    augmented_image = augment_image(image_np)

                    # Create output directory structure
                    relative_path = os.path.relpath(subdir, folder_path)
                    output_subdir = os.path.join(output_folder, relative_path)
                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)

                    # Save the augmented image with a unique name
                    output_file_path = os.path.join(output_subdir, f"aug_{i}_{file}")
                    Image.fromarray(augmented_image).save(output_file_path)
                    print(f"Saved augmented image: {output_file_path}")

# Example usage
input_folder = "E:/TEJ_WORKING_DATA/PROJECTS/DL/PROJ_5/paddy-leaf-disease-main/train"  # Change this to your folder path
output_folder = "E:/TEJ_WORKING_DATA/PROJECTS/DL/PROJ_5/paddy-leaf-disease-main/train_aug"  # Change this to your output folder path
augment_images_in_folder(input_folder, output_folder, augment_count=5)


