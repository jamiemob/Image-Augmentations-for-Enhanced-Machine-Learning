import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
import random
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Define dataset path
dataset_path = "C:/Users/user/PycharmProjects/Pythonmultimediacat1/data"

# Define image size and batch size
img_size = (128, 128)
batch_size = 32

# Create train and validation generators (Original Dataset)
train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(dataset_path, target_size=img_size,
                                                    batch_size=batch_size, class_mode='categorical',
                                                    subset='training')
val_generator = train_datagen.flow_from_directory(dataset_path, target_size=img_size,
                                                  batch_size=batch_size, class_mode='categorical',
                                                  subset='validation')

# Create train generator for Augmented dataset
train_datagen_aug = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, validation_split=0.2
)

train_generator_aug = train_datagen_aug.flow_from_directory(dataset_path, target_size=img_size,
                                                            batch_size=batch_size, class_mode='categorical',
                                                            subset='training')


# CNN Model Function
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(train_generator.class_indices), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Train Model on Original Dataset
cnn_original = create_cnn_model()
start_time_original = time.time()
history_original = cnn_original.fit(train_generator, epochs=10, validation_data=val_generator)
training_time_original = time.time() - start_time_original  # Record training time

# Train Model on Augmented Dataset
cnn_augmented = create_cnn_model()
start_time_augmented = time.time()
history_augmented = cnn_augmented.fit(train_generator_aug, epochs=10, validation_data=val_generator)
training_time_augmented = time.time() - start_time_augmented  # Record training time


# Function to Compare Performance
def compare_model_performance(history1, history2, model1_name, model2_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Validation Accuracy over Epochs
    axes[0].plot(history1.history['val_accuracy'], label=f"{model1_name} Validation Accuracy")
    axes[0].plot(history2.history['val_accuracy'], label=f"{model2_name} Validation Accuracy")
    axes[0].set_title("Validation Accuracy Comparison")
    axes[0].legend()

    # Compare Training Time
    models_names = [model1_name, model2_name]
    times = [training_time_original, training_time_augmented]
    axes[1].bar(models_names, times, color=['blue', 'green'])
    axes[1].set_title("Training Time Comparison")
    axes[1].set_ylabel("Time (seconds)")

    plt.show()


# Confusion Matrix Function
def plot_confusion_matrix(model, generator, model_name):
    y_true = []
    y_pred = []

    # Get true labels and predicted labels
    for images, labels in generator:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels, axis=1))

        if len(y_true) >= generator.samples:
            break  # Stop once we process all images

    cm = confusion_matrix(y_true, y_pred)
    class_names = list(generator.class_indices.keys())

    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))


# Compare Model Performance
compare_model_performance(history_original, history_augmented, "Original", "Augmented")

# Confusion Matrices for Both Models
plot_confusion_matrix(cnn_original, val_generator, "Original Model")
plot_confusion_matrix(cnn_augmented, val_generator, "Augmented Model")



# plot_confusion_matrix(cnn_original, val_generator, "Original Model")
# plot_confusion_matrix(cnn_augmented, val_generator, "Augmented Model")
