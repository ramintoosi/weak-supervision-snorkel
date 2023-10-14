"""
This script provides functions to train and test a CNN model with/without the Snorkel framework.
"""

import torch

from data import get_data_loader_snorkel, get_data_loader
from inference import test_model
from model import CNNModel
from train import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Train():
    """
    Train a CNN model with/without the Snorkel framework.
    """
    model = CNNModel()

    dataloaders = {
        'original': get_data_loader('data_split_manual', splits=['train', 'val']),
        'snorkel_hard': get_data_loader_snorkel('data_snorkel', splits=['train', 'val'], label_type='hard'),
        'snorkel_soft': get_data_loader_snorkel('data_snorkel', splits=['train', 'val'], label_type='soft')
    }
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5)
    for run_name, dataloader in dataloaders.items():
        train(
            model=model, dataloaders=dataloader, optimizer=optimizer, criterion=criterion, scheduler=lr_scheduler,
            device=device,
            run_name=run_name,
            num_epochs=20
        )


def Test():
    """
    Test a trained CNN model using the test dataset.
    """
    test_model(weight_folder='weights', data_root='data_split_manual', device=device)


if __name__ == '__main__':
    Train()
    Test()
