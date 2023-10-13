"""
This script defines the training function for our CNN.
"""
import copy
import os
import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import MulticlassPrecision
from tqdm import tqdm


def train(
        model: nn.Module,
        dataloaders: dict,
        optimizer,
        criterion,
        scheduler,
        device,
        run_name: str,
        writer: SummaryWriter | None = None,
        num_epochs: int = 10):
    """
    Train the CNN model.

    :param model: The neural network model to be trained.
    :param dataloaders: Dictionary containing dataloaders for training and validation.
    :param optimizer: The optimizer for updating model parameters.
    :param criterion: Loss criterion for training.
    :param scheduler: Learning rate scheduler.
    :param device: Device for computation ('cpu' or 'cuda').
    :param run_name: Name of the run for saving model checkpoints.
    :param writer: SummaryWriter for logging training progress (optional).
    :param num_epochs: Number of training epochs.
    :return: Best trained model.
    """

    os.makedirs('./weights', exist_ok=True)

    metric = MulticlassPrecision(num_classes=2, average=None)

    since = time.time()

    dataset_sizes = {phase: len(dl.dataset) for phase, dl in dataloaders.items()}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1_000_000.
    best_acc = 0
    model.to(device)

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_samples = 0

            running_pred_vec = []
            running_target_vec = []

            # Iterate over data.

            pbar = tqdm(dataloaders[phase], desc=f'Epoch {epoch:3}/{num_epochs:3} - {phase:6}', unit=' batch')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                # labels = labels.to(device)
                running_samples += inputs.size(0)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = outputs.cpu()
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                if len(labels.shape) > 1:
                    _, labels = torch.max(labels, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

                running_pred_vec.extend(preds.numpy().tolist())
                running_target_vec.extend(labels.numpy().tolist())

                metric.update(torch.tensor(running_pred_vec), torch.tensor(running_target_vec))
                m = metric.compute()

                pbar.set_postfix(loss=running_loss / running_samples,
                                 accuracy=running_corrects.item() / running_samples * 100,
                                 Precision=[round(m[0].item(), 3), round(m[1].item(), 3)])

            if phase == 'val':
                scheduler.step(running_loss)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if writer is not None:
                writer.add_scalar('LOSS/{}'.format(phase), epoch_loss, epoch)
                writer.add_scalar('ACC/{}'.format(phase), epoch_acc, epoch)
                writer.add_scalar('OPTIM/LR', scheduler.get_lr()[-1], epoch)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'accuracy': best_acc
                }, f'weights/{run_name}_best_model.pt')
            pbar.close()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val accuracy: {best_acc * 100}')

    model.load_state_dict(best_model_wts)
    return model