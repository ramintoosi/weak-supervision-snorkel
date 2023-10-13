"""
This script defines labeling functions using various techniques for weak supervision with Snorkel.
"""

import os

import cv2
import fasttext.util
import numpy as np
import torch
from scipy.stats import mode
from snorkel.labeling import labeling_function
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from collections.abc import Callable

JUNGLE = 0
SEA = 1
ABSTAIN = -1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.eval()
preprocess = EfficientNet_B0_Weights.DEFAULT.transforms(antialias=True)

fasttext.util.download_model('en', if_exists='ignore')
ft = fasttext.load_model('cc.en.300.bin')
v_sea = []
for word in ['sea', 'ocean', 'boat', 'fish', 'beach', 'blue']:
    v_sea.append(ft.get_word_vector(word))
v_sea = np.array(v_sea)
v_jungle = []
for word in ['forest', 'jungle', 'wood', 'bush', 'green']:
    v_jungle.append(ft.get_word_vector(word))
v_jungle = np.array(v_jungle)

LABELING_FUNCS = []


def add_func(f: Callable):
    """
    Decorator to add a labeling function to the list.

    :param f: The labeling function to add.
    :returns: The input function.
    """
    LABELING_FUNCS.append(f)
    return f


@add_func
@labeling_function()
def check_color(x: str | os.PathLike) -> int:
    """
    Labeling function to classify images based on dominant color.

    :param x: Path to the image file.
    :returns: Label indicating SEA, JUNGLE, or ABSTAIN.
    """
    # read image into BGR format
    img = cv2.imread(str(x))
    if len(img.shape) < 3:
        return -1
    color_max = np.argmax(img.mean(axis=(0, 1)))
    match color_max:
        case 0:
            return SEA
        case 1:
            return JUNGLE
        case 2:
            return ABSTAIN


@add_func
@labeling_function()
def check_pixel_color(x: str | os.PathLike) -> int:
    """
    Labeling function to classify images based on mode of pixel colors.

    :param x: Path to the image file.
    :returns: Label indicating SEA, JUNGLE, or ABSTAIN.
    """
    img = cv2.imread(str(x))
    if len(img.shape) < 3:
        return -1
    pixel_max_color = mode(np.argmax(img, axis=2), axis=None).mode
    match pixel_max_color:
        case 0:
            return SEA
        case 1:
            return JUNGLE
        case 2:
            return ABSTAIN


@add_func
@labeling_function()
def check_with_efficientNet(x: str | os.PathLike) -> int:
    """
    Labeling function to classify images using EfficientNet predictions and FastText embeddings.

    :param x: Path to the image file.
    :returns: Label indicating SEA or JUNGLE.
    """
    img = read_image(str(x), mode=ImageReadMode.RGB)
    batch = preprocess(img).unsqueeze(0)

    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    category_name = EfficientNet_B0_Weights.DEFAULT.meta["categories"][class_id]
    v = ft.get_sentence_vector(category_name)

    dist_sea = min(np.sum(np.power(v_sea - v, 2), axis=1))
    dist_jungle = min(np.sum(np.power(v_jungle - v, 2), axis=1))

    return SEA if dist_sea < dist_jungle else JUNGLE
