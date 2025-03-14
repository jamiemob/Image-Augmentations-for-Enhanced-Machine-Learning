import tensorflow as tf
import matplotlib.pyplot as plt
import os
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define separate augmentation techniques
augmentations = {
    "Rotation": ImageDataGenerator(rotation_range=40),
    "Width Shift": ImageDataGenerator(width_shift_range=0.2),
    "Height Shift": ImageDataGenerator(height_shift_range=0.2),
    "Shear": ImageDataGenerator(shear_range=0.2),
    "Zoom": ImageDataGenerator(zoom_range=0.2),
    "Horizontal Flip": ImageDataGenerator(horizontal_flip=True),
    "Brightness": ImageDataGenerator(brightness_range=[0.5, 1.5])
}

# Define dataset path (adjust if necessary)
dataset_path = "C:/Users/user/PycharmProjects/Pythonmultimediacat1/data"

# Collect all images from all subdirectories
all_images = []
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if os.path.isdir(category_path):  # Ensure it's a folder
        images = os.listdir(category_path)
        for img in images:
            all_images.append(os.path.join(category_path, img))

# Randomly select 5 images from the dataset
random_images = random.sample(all_images, 5)

# Define augmentation visualization function
def visualize_separate_augmentations(image_path):
    img = load_img(image_path)  # Load image
    img_array = img_to_array(img)  # Convert to array
    img_array = img_array.reshape((1,) + img_array.shape)  # Reshape for augmentation

    fig, ax = plt.subplots(1, len(augmentations) + 1, figsize=(20, 5))

    # Display original image
    ax[0].imshow(load_img(image_path))
    ax[0].set_title("Original")

    # Apply each augmentation separately
    for i, (aug_name, aug_gen) in enumerate(augmentations.items(), 1):
        aug_iter = aug_gen.flow(img_array, batch_size=1)
        aug_img = next(aug_iter)[0].astype('uint8')
        ax[i].imshow(aug_img)
        ax[i].set_title(aug_name)

    plt.show()

# Apply augmentation visualization on the selected images
for img_path in random_images:
    print(f"Processing {img_path}")
    visualize_separate_augmentations(img_path)
