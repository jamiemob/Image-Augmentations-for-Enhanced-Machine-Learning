# Image Augmentation with TensorFlow/Keras

## Overview
This project performs image augmentation on a set of images using TensorFlow/Keras. The script selects five random images from a specified dataset directory, applies various augmentation techniques, and displays the original and augmented images in a grid format.

## Dependencies
Ensure you have the following Python packages installed before running the script:

- `os` (built-in)
- `random` (built-in)
- `cv2` (OpenCV)
- `numpy`
- `matplotlib`
- `tensorflow`

Install missing dependencies using:
```bash
pip install opencv-python numpy matplotlib tensorflow
```

## Features
- Loads images from the specified dataset directory.
- Randomly selects five images for augmentation.
- Applies the following augmentation techniques:
  - Rotation (up to 30 degrees)
  - Horizontal flipping
  - Brightness adjustment
  - Zoom (up to 20%)
  - Addition of Gaussian noise
- Displays the original and augmented images in a structured layout.

## Script Breakdown
### 1. Define Dataset Path
```python
DATASET_PATH = "Dataset"
```
This variable specifies the directory where the images are stored.

### 2. Load and List Images
The script iterates through the dataset directory and collects all image file paths with extensions `.jpg`, `.png`, or `.jpeg`.
```python
all_images = []
for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.endswith((".jpg", ".png", ".jpeg")):
            all_images.append(os.path.join(root, file))
```
### 3. Random Image Selection
```python
random_images = random.sample(all_images, 5)
```
Five images are randomly selected from the dataset for augmentation.

### 4. Load Images
```python
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (128, 128))  # Resize for uniformity
    return img
```
This function reads an image using OpenCV, converts it to RGB format, and resizes it to 128x128 pixels.

### 5. Define Augmentation Transformations
Augmentations are performed using the `ImageDataGenerator` class from TensorFlow/Keras:
```python
data_gen = ImageDataGenerator(
    rotation_range=30,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
    zoom_range=0.2
)
```

### 6. Add Gaussian Noise
A custom function adds Gaussian noise to an image:
```python
def add_gaussian_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype(np.uint8)
    noisy_image = cv2.add(image, gauss)
    return np.clip(noisy_image, 0, 255)
```

### 7. Apply Augmentations and Display Results
Each selected image undergoes different augmentation techniques. The original and augmented images are displayed using Matplotlib:
```python
fig, axes = plt.subplots(5, 6, figsize=(15, 12))
axes = axes.flatten()

for idx, image_path in enumerate(random_images):
    original_img = load_image(image_path)
    augmented_images = [
        data_gen.random_transform(original_img),  # Rotation
        cv2.flip(original_img, 1),  # Horizontal Flip
        data_gen.random_transform(original_img),  # Brightness Adjustment
        data_gen.random_transform(original_img),  # Zoom
        add_gaussian_noise(original_img)  # Gaussian Noise
    ]

    axes[idx * 6].imshow(original_img)
    axes[idx * 6].set_title("Original")
    titles = ["Rotated", "Flipped", "Brightness", "Zoom", "Noise"]

    for j, aug_img in enumerate(augmented_images):
        axes[idx * 6 + j + 1].imshow(aug_img.astype(np.uint8))
        axes[idx * 6 + j + 1].set_title(titles[j])

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.show()
```
## CNN Training and Comparison

### Steps:
1. **Train on Original Data**:  
   ```bash
   python train.py --data_dir data/original --model_name original_model.h5
2. **Train on Augmented Data**:
   ```bash
   python train.py --data_dir data/augmented --model_name augmented_model.h5
4. **Performance Analysis**:
   - The augmented model shows reduced overfitting and better generalization.
## Usage
1. Place your dataset of images in the `Dataset/` directory.
2. Run the script:
```bash
python script.py
```
3. The script will display the original images along with their augmented versions.
 
## Notes
- Ensure `Dataset/` contains sufficient images to allow random selection.
- You can modify augmentation parameters by adjusting `ImageDataGenerator` settings.

## License
This project is open-source and available for modification and distribution.

