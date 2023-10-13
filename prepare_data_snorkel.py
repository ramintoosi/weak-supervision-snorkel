"""
This script performs weak supervision labeling using labeling functions with Snorkel.
"""

import os
import pickle
from pathlib import Path

from snorkel.labeling import LFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel

from labeling_funcs import LABELING_FUNCS


def weak_supervision_labeling(root: str = 'data_split_manual', splits=None, root_save: str = 'data_snorkel'):
    """
    Perform weak supervision labeling using labeling functions with snorkel and save label data to pickle files.

    :param root: Root directory containing the input images (root/split/class_name).
    :param splits: List of data splits to process (e.g. ['train', 'val']).
    :param root_save: Root directory to save the labeled data.
    """
    if splits is None:
        splits = ['train', 'val']

    root_new = Path(root_save)
    os.makedirs(root_new, exist_ok=True)

    for split in splits:
        root_split = Path(root / split)
        all_images = list(root_split.glob('**/*.jpg'))
        print(f'{len(all_images)} images found in {split}')
        applier = LFApplier(LABELING_FUNCS)

        L_train = applier.apply(all_images)

        print(LFAnalysis(L=L_train, lfs=LABELING_FUNCS).lf_summary())

        label_model = LabelModel(cardinality=2, verbose=True)
        label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)

        label_split = label_model.predict_proba(L_train)

        os.makedirs(root_new / split, exist_ok=True)
        with open(root_new / split / 'data.pkl', 'wb') as f:
            pickle.dump({"labels": label_split, "images": all_images}, f)


if __name__ == '__main__':
    weak_supervision_labeling()