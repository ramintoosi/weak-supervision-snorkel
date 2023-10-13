"""
Data module

This module contains utility functions and classes for training a Convolutional Neural Network (CNN) using weak
supervision techniques. It includes functions to create data loaders for different dataset splits, as well as a custom
dataset class for weakly supervised learning with Snorkel labels.

Classes:
    - SnorkelDataset: Custom dataset class for weakly supervised learning using Snorkel labels.

Functions:
    - get_transforms: Get data transformation pipelines for different dataset splits.
    - get_data_loader: Get data loaders for the specified dataset splits.
    - get_data_loader_snorkel: Get data loaders for weakly supervised learning using Snorkel labels.

Note:
    This module assumes that the data is organized into different splits (e.g., 'train', 'val', 'test')
    within the root directory.

"""

import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def get_data_loader(root: str, splits: list['str']) -> dict[str, DataLoader]:
    """
    Creates and returns data loaders for different dataset splits.
    :param root: Root directory containing the dataset.
    :param splits: List of dataset splits (e.g., ['train', 'val', 'test']).
    :return: Dataloaders, a dictionary containing data loaders for each split.
    """
    dataloaders = dict()
    transforms = get_transforms()
    for split in splits:
        dataset = ImageFolder(os.path.join(root, split), transform=transforms[split])
        dataloaders[split] = DataLoader(dataset=dataset,
                                        shuffle=True if split == 'train' else 0,
                                        num_workers=6,
                                        batch_size=128,
                                        pin_memory=True)

    return dataloaders


def get_data_loader_snorkel(root: str, splits: list['str'], label_type: str) -> dict[str, DataLoader]:
    """
    Get data loaders for the specified dataset splits for snorkel generated dataset.
    :param root: Root directory of the dataset that contains "data.pkl" files in each split.
    :param splits: List of dataset splits (e.g., ['train', 'val', 'test']).
    :param label_type: Type of labels, either 'hard' or 'soft'.
    :return: Dictionary of data loaders for each split.
    """
    dataloaders = dict()
    transforms = get_transforms()
    for split in splits:
        dataset = SnorkelDataset(os.path.join(root, split, 'data.pkl'),
                                 transforms=transforms[split],
                                 label_type=label_type)

        dataloaders[split] = DataLoader(dataset=dataset,
                                        shuffle=True if split == 'train' else 0,
                                        num_workers=6,
                                        batch_size=64,
                                        pin_memory=True)

    return dataloaders


class SnorkelDataset(Dataset):
    """
    Custom dataset for weakly supervised learning using Snorkel labels.
    """

    def __init__(self, data_path: str, label_type: str, transforms: transforms.Compose | None = None):
        """
        :param data_path: Path to the pickled data file.
        :param label_type: Type of labels, either 'hard' or 'soft'.
        :param transforms: Data transformations to apply to images.
        """
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        self.labels = data['labels']
        if label_type.lower() == 'hard':
            self.labels = np.argmax(self.labels, axis=1)
        self.images = data["images"]
        self.transforms = transforms

    def __len__(self) -> int:
        """
        Get the length of the dataset.
        :return: Number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx: int):
        """
        Get an item from the dataset.
        :param idx: Index of the item to retrieve.
        :return: Tuple containing the image and its corresponding label.
        """
        image = Image.open(str(self.images[idx])).convert("RGB")

        if transforms:
            image = self.transforms(image)

        return image, self.labels[idx]


def get_transforms() -> dict[str, transforms.Compose]:
    """
    Get data transformation pipelines for different dataset splits.
    :return: Dictionary of data transformations for 'train', 'val', and 'test' splits.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    return data_transforms
