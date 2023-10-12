"""
This script downloads images from the Open Images Dataset based on requested classes and splits.
Ref: https://raw.githubusercontent.com/openimages/dataset/master/downloader.py
"""

import os
from concurrent import futures

import boto3
import botocore
import numpy as np
import pandas as pd
from tqdm import tqdm

BUCKET_NAME = 'open-images-dataset'
BUCKET = boto3.resource(
    's3', config=botocore.config.Config(
        signature_version=botocore.UNSIGNED)).Bucket(BUCKET_NAME)


def download_one_image(bucket, split: str, path: str, image_id: str, progress_bar: tqdm):
    """
    Download a single image from the specified split.

    :param bucket: S3 bucket resource.
    :param split: Dataset split ('train', 'val', 'test').
    :param path: Path to store the downloaded image.
    :param image_id: Image ID.
    :param progress_bar: TQDM progress bar.
    """
    try:
        filename = os.path.join(path, f'{image_id}.jpg')
        if not os.path.isfile(filename):
            bucket.download_file(f'{split}/{image_id}.jpg', filename)
    except botocore.exceptions.ClientError as exception:
        # TODO: log
        # print(f'\nERROR when downloading image `{split}/{image_id}`: {str(exception)}\n')
        pass
    progress_bar.update(1)


def get_class_label(requested_classnames: list[str]) -> dict[str, str]:
    """
    Retrieve class labels for requested class names.

    :param requested_classnames: List of requested class names.
    :returns: Dictionary mapping class labels to class names.
    """
    classnames: pd.DataFrame = pd.read_csv('oiv7/oidv7-class-descriptions.csv')
    classes_trainable = set(line.strip() for line in open('oiv7/oidv7-classes-trainable.txt'))

    requested_labels = dict()
    for cname in requested_classnames:
        ind = classnames.index[classnames['DisplayName'].str.lower() == cname.lower()].tolist()
        label = classnames.values[ind][0][0]
        # check if the class is trainable
        if label not in classes_trainable:
            raise TypeError(f"Class {cname} is not trainable!")
        requested_labels[label] = cname.lower()
    return requested_labels


def get_image_ids(splits: list[str], requested_labels: dict[str, str], num_images: int) -> dict[str, list]:
    """
    Retrieve image IDs for requested splits and labels.

    :param splits: List of dataset splits.
    :param requested_labels: Dictionary mapping class labels to class names.
    :param num_images: Number of images to retrieve.
    :returns: Dictionary mapping splits to lists of image ids.
        """
    image_ids_per_split = dict()
    for split in splits:
        filename = f'oiv7/oidv7-{split}-annotations-human-imagelabels.csv'
        example_per_label = np.zeros(2)
        image_ids = []
        chunk: pd.DataFrame
        with pd.read_csv(filename, chunksize=10 ** 6) as pd_reader:
            for chunk in pd_reader:
                if np.all(example_per_label > num_images):
                    break
                for i, label in enumerate(requested_labels):
                    if example_per_label[i] > num_images:
                        continue
                    verified_chunk = chunk[(chunk['LabelName'] == label) & chunk['Confidence'] == 1.0]
                    example_per_label[i] += len(verified_chunk)
                    image_ids.extend(verified_chunk.values.tolist())
        image_ids_per_split[split] = image_ids
    return image_ids_per_split


def download_images(requested_labels: dict[str, str], image_ids_dict: dict[str, list], path: str):
    """
    Download images based on requested labels and splits.

    :param requested_labels: Dictionary mapping class labels to class names.
    :param image_ids_dict: Dictionary mapping splits to lists of image ids.
    :param path: Path to store the downloaded images.
    """
    os.makedirs(path, exist_ok=True)
    for split, image_ids in image_ids_dict.items():
        if split == 'val':
            split = 'validation'
        os.makedirs(f'{path}/{split}', exist_ok=True)
        for _, classname in requested_labels.items():
            os.makedirs(f'{path}/{split}/{classname}', exist_ok=True)
        progress_bar = tqdm(total=len(image_ids), desc=f'Downloading {split} images', leave=True)
        with futures.ThreadPoolExecutor(5) as executor:
            all_futures = [
                executor.submit(download_one_image,
                                BUCKET, split,
                                f'{path}/{split}/{requested_labels[image_property[2]]}',
                                image_property[0], progress_bar)
                for image_property in image_ids
            ]
            for future in futures.as_completed(all_futures):
                future.result()
        progress_bar.close()


if __name__ == '__main__':
    req_lab = get_class_label(['sea', 'Jungle'])
    img_ids = get_image_ids(['train', 'val', 'test'], req_lab, 10000)
    download_images(requested_labels=req_lab, image_ids_dict=img_ids, path='dataset')
