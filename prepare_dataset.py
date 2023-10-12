"""
This module performs data splitting and organizes images into different splits based on classes.
"""

import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def stat(root_path: str) -> dict:
    """
    Analyze and compute statistics on the data by counting images for each class.

    :param root_path: Root directory containing the images.
    :returns: Dictionary mapping class names to lists of image paths.
    """
    root = Path(root_path)
    all_images = list(root.glob('**/*.jpg'))
    data = defaultdict(lambda: [])
    for image_path in all_images:
        _, _, class_name, _ = str(image_path).split('/')
        data[class_name].append(image_path)
    print('Data Stat:')
    for class_name in data:
        print(f'{class_name}: {len(data[class_name])}')

    return data


def split_manual(data: dict[str, list], save_dir: str, splits: list[float] | None = None,
                 random_seed: int = 56) -> None:
    """
    Split the data into different subsets (train, val, test) and organize images accordingly.

    :param data: Dictionary mapping class names to lists of image paths.
    :param save_dir: Directory to save the split data.
    :param splits: List of ratios for train, val, and test splits.
    :param random_seed: Random seed for reproducibility.
    """
    x = []
    y = []
    X = dict()
    Y = dict()
    for class_name, image_list in data.items():
        x.extend(image_list)
        y.extend([class_name] * len(image_list))
    if splits is None:
        splits = [0.7, 0.2, 0.1]
    splits = np.divide(splits, np.sum(splits))
    x_train_val, X['test'], y_train_val, Y['test'] = train_test_split(x, y,
                                                                      test_size=splits[2],
                                                                      random_state=random_seed,
                                                                      stratify=y)
    X['train'], X['val'], Y['train'], Y['val'] = train_test_split(x_train_val, y_train_val,
                                                                  test_size=splits[1] / (splits[1] + splits[0]),
                                                                  random_state=random_seed,
                                                                  stratify=y_train_val)
    os.makedirs(save_dir, exist_ok=True)
    save_path = Path(save_dir)
    for split in X:
        os.makedirs(save_path / split, exist_ok=True)
        for class_name in data:
            os.makedirs(save_path / split / class_name, exist_ok=True)
        for img_path, class_name in tqdm(zip(X[split], Y[split]), total=len(X[split]), desc=split):
            shutil.copyfile(img_path, save_path / split / class_name / Path(img_path).name)


if __name__ == '__main__':
    data = stat('dataset')
    split_manual(data, save_dir='data_split_manual')
