import math

import Augmentor
from pathlib import Path
from typing import List
import numpy as np
import shutil
import re


def AugmentImages(imagesDirectory: Path, segmentationsDirectory: Path, outputDirectory: Path,
                  count: int, distort=True):
    print("Images directory:", imagesDirectory.absolute())
    print("Segmentations directory:", segmentationsDirectory.absolute())
    print("Output directory:", outputDirectory.absolute())

    # Verificar la existencia de los directorios
    if not imagesDirectory.exists():
        raise OSError(f"The images directory does not exist: {imagesDirectory.absolute()}")
    if not segmentationsDirectory.exists():
        raise OSError(f"The segmentations directory does not exist: {segmentationsDirectory.absolute()}")
    if not outputDirectory.exists():
        raise OSError(f"The output directory does not exist: {outputDirectory.absolute()}")

    augmentor = Augmentor.Pipeline(source_directory=str(imagesDirectory.absolute()),
                                   output_directory=str(outputDirectory.absolute()))
    augmentor.set_save_format("auto")
    print(segmentationsDirectory.absolute())
    augmentor.ground_truth(str(segmentationsDirectory.absolute()))

    # Random transformations to apply
    augmentor.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)
    augmentor.flip_left_right(probability=0.5)
    augmentor.flip_top_bottom(probability=0.5)
    augmentor.zoom_random(probability=0.5, percentage_area=0.7)
    if distort:
        augmentor.shear(probability=1, max_shear_left=20, max_shear_right=20)
        augmentor.random_distortion(probability=0.5, grid_width=5, grid_height=5, magnitude=3)
        augmentor.skew(probability=0.5, magnitude=0.3)
    # Resize images to 512x512 (in case some were cropped)
    augmentor.resize(1, 512, 512)

    # Execute the transformations
    augmentor.sample(count)

    # Rearrange saved directory
    outputImagesPath = outputDirectory / "images"
    outputSegmentationsPath = outputDirectory / "segmentations"

    files = [path for path in outputDirectory.iterdir() if path.is_file()]
    segmentationFiles = [path for path in files if re.match("_groundtruth*", path.stem)]
    imageFiles = [x for x in files if x not in segmentationFiles]

    for segmentationFile in segmentationFiles:
        newFilename = re.sub(".*_", "", segmentationFile.name)
        segmentationFile.rename(outputSegmentationsPath / newFilename)

    for imageFile in imageFiles:
        newFilename = re.sub(".*_", "", imageFile.name)
        imageFile.rename(outputImagesPath / newFilename)


def SplitData(imagePaths: List[Path], segmentationPaths: List[Path], validationFraction: float,
              testingFraction: float,
              outputDirectory: Path):
    # Sort paths alphabetically
    imagePaths.sort(key=lambda x: x.stem)
    segmentationPaths.sort(key=lambda x: x.stem)

    # Only use paths that have matched names in image and segmentations directory
    imagePaths = np.asarray([path for path in imagePaths if
                             path.stem in [segPath.stem for segPath in segmentationPaths]])
    segmentationPaths = np.asarray([path for path in segmentationPaths if
                                    path.stem in [imagePath.stem for imagePath in imagePaths]])

    # Carry out the split!
    permutation = np.random.permutation(len(imagePaths))
    numValidation = math.ceil(len(imagePaths) * validationFraction)
    numTesting = math.ceil(len(imagePaths) * testingFraction)
    validationIndices = permutation[:numValidation]
    testingIndices = permutation[numValidation:(numValidation + numTesting)]
    trainingIndices = permutation[(numValidation + numTesting):]

    # Save the images
    _CopyToPath(imagePaths[trainingIndices], outputDirectory / "training" / "images")
    _CopyToPath(imagePaths[validationIndices], outputDirectory / "validation" / "images")
    _CopyToPath(imagePaths[testingIndices], outputDirectory / "testing" / "images")
    _CopyToPath(segmentationPaths[trainingIndices], outputDirectory / "training" / "segmentations")
    _CopyToPath(segmentationPaths[validationIndices],
                outputDirectory / "validation" / "segmentations")
    _CopyToPath(segmentationPaths[testingIndices], outputDirectory / "testing" / "segmentations")


def _CopyToPath(paths: List[Path], output: Path):
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        newPath = output / path.name
        shutil.copy(path, newPath)
        
if __name__ == '__main__':
    images_dir = Path('Resources/Images')
    segmentations_dir = Path('Resources/Segmentation')
    output_dir = Path('Oversampling')


    count = 100

    AugmentImages(imagesDirectory=images_dir, segmentationsDirectory=segmentations_dir, outputDirectory=output_dir, count=count)

    image_paths = list(images_dir.glob('*.png'))  
    segmentation_paths = list(segmentations_dir.glob('*.png')) 


    validation_fraction = 0.15
    testing_fraction = 0.05

    SplitData(imagePaths=image_paths, segmentationPaths=segmentation_paths, validationFraction=validation_fraction, testingFraction=testing_fraction, outputDirectory=output_dir)
