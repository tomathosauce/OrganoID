# Augment.py -- Sub-program to augment training data.

from CommandLine.Program import Program
from Core.DataPreparation import AugmentImagesCustomV1
import argparse
import pathlib


class Augment(Program):
    def Name(self):
        return "augment"

    def Description(self):
        return "Augment images for training."

    def RunProgram(self, inputPath: str, outputPath: str, count = 2000):
        self.MakeDirectory(outputPath / "images")
        self.MakeDirectory(outputPath / "segmentations")
        [self.AssertDirectoryExists(x) for x in [outputPath,
                                                 outputPath / "images",
                                                 outputPath / "segmentations",
                                                 inputPath,
                                                 inputPath / "images",
                                                 inputPath / "segmentations"]]
        AugmentImagesCustomV1(inputPath / "images",
                      inputPath / "segmentations",
                      outputPath,
                      count)
