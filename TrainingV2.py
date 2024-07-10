import math
from pathlib import Path
from typing import List
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
# from SURE.custom.Core.Model import BuildModel, TrainModel, GroundTruth, ComputeIOUs  # Importing necessary functions and classes
from Core.ImageHandling import LoadPILImages
from Core.Model import BuildModel, TrainModel, GroundTruth, ComputeIOUs  # Importing necessary functions and classes

physical_devices = tf.config.list_physical_devices('GPU')

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
# Define paths
training_sets = Path('SURE/datasets/handpicked/2/subsets')
save_directory = Path('SURE/datasets/handpicked/2/models')


# Function to get GroundTruth objects
def get_ground_truths(images_dir: Path, segmentations_dir: Path) -> List[GroundTruth]:
    image_paths = list(images_dir.glob('*.png'))
    segmentation_paths = list(segmentations_dir.glob('*.png'))
    
    ground_truths = []
    for img_path in image_paths:
        seg_path = segmentations_dir / img_path.name
        if seg_path in segmentation_paths:
            ground_truths.append(GroundTruth(img_path, seg_path))
    
    return ground_truths

def retrieve_ground_truths(root_path: Path):
    training_images_dir = root_path / "training" / "images"
    training_segmentations_dir = root_path / "training" / "segmentations"
    validation_images_dir = root_path / "validation" / "images"
    validation_segmentations_dir = root_path / "validation" / "segmentations"
    test_images_dir = root_path / "testing" / "images"
    test_segmentations_dir = root_path / "testing" / "segmentations"

    # Get all ground truths
    all_training_ground_truths = get_ground_truths(training_images_dir, training_segmentations_dir)
    validation_ground_truths = get_ground_truths(validation_images_dir, validation_segmentations_dir)
    test_ground_truths = get_ground_truths(test_images_dir, test_segmentations_dir)
    
    return all_training_ground_truths, validation_ground_truths, test_ground_truths
# Define model parameters
image_size = (512, 512)  # Adjust if your image size is different
dropout_rate = 0.125
first_layer_filter_count = 8

# Training parameters
learning_rate = 0.001
patience = 10
epochs = 10
batch_size = 8
save_name_prefix = "UNet"
save_lite = False
save_all = True

# Image counts to test
  # Add more or adjust as needed

# Lists to store results
train_ious = []
val_ious = []
test_ious = []

folder_names = [folder for folder in training_sets.iterdir() if folder.is_dir()]
folder_names.sort(key=lambda x: int(x.name))
# image_counts = [folder.name for folder in training_sets.iterdir() if folder.is_dir()]

alldata = []


for folder in folder_names[:1]:
    print(f"Training with {folder} images")
    
    # Create the model
    model = BuildModel(image_size, dropout_rate, first_layer_filter_count)
    
    training_ground_truths, validation_ground_truths, test_ground_truths = retrieve_ground_truths(folder)
    
    # Train the model
    TrainModel(
        model=model,
        learningRate=learning_rate,
        patience=patience,
        epochs=epochs,
        batchSize=batch_size,
        trainingData=training_ground_truths,
        validationData=validation_ground_truths,
        saveDirectory=save_directory / str(folder.name),
        saveNamePrefix=f"{save_name_prefix}_{folder.name}_images",
        saveLite=save_lite,
        saveAll=save_all
    )
    
    # Compute IoU for training, validation, and test sets
    train_iou = ComputeIOUs(model, LoadPILImages([gt.imagePath for gt in training_ground_truths]), LoadPILImages([gt.segmentationPath for gt in training_ground_truths]))
    train_iou_mean = np.mean(train_iou)
    train_iou_std = np.std(train_iou)
    
    val_iou = ComputeIOUs(model, LoadPILImages([gt.imagePath for gt in validation_ground_truths]), LoadPILImages([gt.segmentationPath for gt in validation_ground_truths]))
    val_iou_mean = np.mean(val_iou)
    val_iou_std = np.std(val_iou)
    
    test_iou = ComputeIOUs(model, LoadPILImages([gt.imagePath for gt in test_ground_truths]), LoadPILImages([gt.segmentationPath for gt in test_ground_truths]))
    test_iou_mean = np.mean(test_iou)
    test_iou_std = np.std(test_iou)
    
    alldata.append({
        "image_count": int(folder.name),
        "train_iou_mean": train_iou_mean,
        "train_iou_std": train_iou_std,
        
        "val_iou_mean": val_iou_mean,
        "val_iou_std": val_iou_std,
        
        "test_iou_mean": test_iou_mean,
        "test_iou_std": test_iou_std
    })
    
    print(f"Training IoU: {train_iou_mean:.4f}")
    print(f"Validation IoU: {val_iou_mean:.4f}")
    print(f"Test IoU: {test_iou_mean:.4f}")
    print(f"Finished training with {folder.name} images\n")

print("Training complete for all image counts")


df = pd.DataFrame(alldata)
df.to_csv("iou_data")
# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(df["image_count"], df["train_iou_mean"], 'b-', label='Training IoU')
plt.plot(df["image_count"], df["val_iou_mean"], 'g-', label='Validation IoU')
plt.plot(df["image_count"], df["test_iou_std"], 'r-', label='Test IoU')
plt.xlabel('Number of Training Images')
plt.ylabel('Mean IoU')
plt.title('Model Performance vs. Number of Training Images')
plt.legend()
plt.grid(True)
plt.savefig('iou_vs_training_images.png')
plt.show()

# Find the best performing model based on validation IoU
best_count = df["image_count"][np.argmax(val_ious)]
print(f"The best performing model used {best_count} training images.")
print(f"Best Validation IoU: {max(val_ious):.4f}")
print(f"Corresponding Test IoU: {test_ious[np.argmax(val_ious)]:.4f}")
