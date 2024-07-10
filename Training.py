import math
from pathlib import Path
from typing import List
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
# from SURE.custom.Core.Model import BuildModel, TrainModel, GroundTruth, ComputeIOUs  # Importing necessary functions and classes
from Core.Model import BuildModel, TrainModel, GroundTruth, ComputeIOUs  # Importing necessary functions and classes

physical_devices = tf.config.list_physical_devices('GPU')

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass
# Define paths
output_dir = Path('SURE/datasets/handpicked/1/test')
training_images_dir = output_dir / "training" / "images"
training_segmentations_dir = output_dir / "training" / "segmentations"
validation_images_dir = output_dir / "validation" / "images"
validation_segmentations_dir = output_dir / "validation" / "segmentations"
test_images_dir = output_dir / "testing" / "images"
test_segmentations_dir = output_dir / "testing" / "segmentations"

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

# Get all ground truths
all_training_ground_truths = get_ground_truths(training_images_dir, training_segmentations_dir)
validation_ground_truths = get_ground_truths(validation_images_dir, validation_segmentations_dir)
test_ground_truths = get_ground_truths(test_images_dir, test_segmentations_dir)

# Shuffle the training data
np.random.shuffle(all_training_ground_truths)

# Define model parameters
image_size = (512, 512)  # Adjust if your image size is different
dropout_rate = 0.1
first_layer_filter_count = 8

# Training parameters
learning_rate = 0.001
patience = 10
epochs = 50
batch_size = 4
save_directory = Path('ModelCheckpoints')
save_name_prefix = "UNet"
save_lite = False
save_all = True

# Image counts to test
image_counts = [100]  # Add more or adjust as needed

# Lists to store results
train_ious = []
val_ious = []
test_ious = []

for count in image_counts:
    print(f"Training with {count} images")
    
    # Create the model
    model = BuildModel(image_size, dropout_rate, first_layer_filter_count)

    # Select subset of training data
    training_subset = all_training_ground_truths[:count]
    
    # Train the model
    TrainModel(
        model=model,
        learningRate=learning_rate,
        patience=patience,
        epochs=epochs,
        batchSize=batch_size,
        trainingData=training_subset,
        validationData=validation_ground_truths,
        saveDirectory=save_directory,
        saveNamePrefix=f"{save_name_prefix}_{count}_images",
        saveLite=save_lite,
        saveAll=save_all
    )
    
    # Compute IoU for training, validation, and test sets
    train_iou = np.mean(ComputeIOUs(model, [gt.imagePath for gt in training_subset], [gt.segmentationPath for gt in training_subset]))
    val_iou = np.mean(ComputeIOUs(model, [gt.imagePath for gt in validation_ground_truths], [gt.segmentationPath for gt in validation_ground_truths]))
    test_iou = np.mean(ComputeIOUs(model, [gt.imagePath for gt in test_ground_truths], [gt.segmentationPath for gt in test_ground_truths]))
    
    train_ious.append(train_iou)
    val_ious.append(val_iou)
    test_ious.append(test_iou)
    
    print(f"Training IoU: {train_iou:.4f}")
    print(f"Validation IoU: {val_iou:.4f}")
    print(f"Test IoU: {test_iou:.4f}")
    print(f"Finished training with {count} images\n")

print("Training complete for all image counts")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(image_counts, train_ious, 'b-', label='Training IoU')
plt.plot(image_counts, val_ious, 'g-', label='Validation IoU')
plt.plot(image_counts, test_ious, 'r-', label='Test IoU')
plt.xlabel('Number of Training Images')
plt.ylabel('Mean IoU')
plt.title('Model Performance vs. Number of Training Images')
plt.legend()
plt.grid(True)
plt.savefig('iou_vs_training_images.png')
plt.show()

# Find the best performing model based on validation IoU
best_count = image_counts[np.argmax(val_ious)]
print(f"The best performing model used {best_count} training images.")
print(f"Best Validation IoU: {max(val_ious):.4f}")
print(f"Corresponding Test IoU: {test_ious[np.argmax(val_ious)]:.4f}")
