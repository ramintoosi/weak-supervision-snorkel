"""
This script tests trained CNN models using various metrics on the test dataset.
We assume that there is a test split in the data root
"""

import os
from pathlib import Path

import torch
from torcheval.metrics import MulticlassPrecision, BinaryAccuracy
from tqdm import tqdm

from data import get_data_loader
from model import CNNModel

metrics = [MulticlassPrecision(num_classes=2, average=None),
           BinaryAccuracy()]


def test_model(weight_folder: str, data_root: str, device='cpu') -> None:
    """
    Test the trained CNN models using specified metrics.

    :param weight_folder: Path to the folder containing model weight files (eg. original_best_model.pt).
    :param data_root: Root directory of the dataset.
    :param device: Device to run the evaluation on (default is 'cpu').
    """

    model_wights = [os.path.join(weight_folder, x) for x in os.listdir(weight_folder) if x.endswith('_best_model.pt')]
    results = []
    print(f'Found {len(model_wights)} Models.')
    for model_weight in model_wights:
        model = CNNModel()
        model.load_state_dict(torch.load(model_weight)['model_state_dict'])
        model.eval()
        model.to(device)

        dataloaders = get_data_loader(root=data_root, splits=['test'])
        predictions = []
        labels = []
        pbar = tqdm(dataloaders['test'], desc=f'{" ".join(Path(model_weight).name.split("_")[:-2]):15}', unit=' batch')
        for inputs, label in pbar:
            inputs = inputs.to(device)
            # labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                outputs = outputs.cpu()
                _, preds = torch.max(outputs, 1)

            predictions.extend(preds.numpy().tolist())
            labels.extend(label.numpy().tolist())
        for metric in metrics:
            metric.update(torch.tensor(predictions), torch.tensor(labels))
        results.append(f'{" ".join(Path(model_weight).name.split("_")[:-2]):15} '
                       f'Precision = {metrics[0].compute().numpy()} '
                       f'ACC = {metrics[1].compute():.4f}')

    for res in results:
        print(res)
