from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class FERPlusDataset(Dataset):
    def __init__(
        self,
        csv_path,
        img_root: str,
        split: Literal["Training", "PublicTest", "PrivateTest"],
        img_size=48,
        augmentation_transforms=None,
    ):
        """
        csv_path: path to fer2013new.csv
        img_root: directory with all images
        split: e.g. 'Training', 'PublicTest', 'PrivateTest'
        augmentation_transforms: torchvision transforms (on PIL image) to be
            applied before normalisation.
        """
        normalisation = _normalise(img_size)
        self.transforms = (
            normalisation
            if not augmentation_transforms
            else transforms.Compose([augmentation_transforms, normalisation])
        )
        df = pd.read_csv(csv_path)

        # Drop rows without filenames
        df = df[df["Image name"].notna()]

        # filter by split
        df = df[df["Usage"] == split].reset_index(drop=True)

        split_to_path = {
            "Training": "FER2013Train",
            "PublicTest": "FER2013Valid",
            "PrivateTest": "FER2013Test",
        }
        split_path = split_to_path[split]
        self.images = [
            Image.open(Path(img_root) / split_path / filename).convert("RGB")
            for filename in df["Image name"].tolist()
        ]

        # Follows order in FER+ csv
        self.classes = [
            "neutral",
            "happiness",
            "surprise",
            "sadness",
            "anger",
            "disgust",
            "fear",
            "contempt",
        ]
        votes = df[self.classes].values.astype("float32")  # shape [N, C]

        # convert votes -> probability distributions (soft targets)
        sums = votes.sum(axis=1, keepdims=True)
        sums[sums == 0.0] = 1.0  # avoid division by zero
        self.targets = torch.from_numpy(votes / sums)  # [N, C], each row sums to 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.transforms(img)

        target = self.targets[idx]

        return img, target


def _normalise(img_size):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
