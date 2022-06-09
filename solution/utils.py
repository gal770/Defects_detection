import os

import torch
from torch import nn
from torchvision import transforms

from wafers_dataset import WafwersDataSet
from models import get_xception_based_model

"""Utility methods and constants used throughout the project."""


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_transform(part, patch_size):
    if part == 'train':
        TRANSFORM = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(0.4,
                                 0.2437)
        ])
    if part == 'val' or part == 'test':
        TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.4,
                                 0.2437),
        ])
    return TRANSFORM


def load_dataset(dataset_name: str, dataset_part: str, patch_size: int) -> \
        torch.utils.data.Dataset:
    """Loads dataset part from dataset name.

    For example, loading the training set of 128 X 128 sized patches.

    Args:
        dataset_name: dataset name. For example, AD_CTW_128.
        dataset_part: dataset part, one of: train, val, test.

    Returns:
        dataset: a torch.utils.dataset.Dataset instance.
    """
    transform = {'train': load_transform('train', patch_size),
                 'val': load_transform('val', patch_size),
                 'test': load_transform('test', patch_size)}[dataset_part]
    dataset = WafwersDataSet(
        root_path=os.path.join(os.path.dirname(os.getcwd()), 'datasets',
                               dataset_name, dataset_part),
        transform=transform)
    return dataset


def load_model() -> nn.Module:
    print(f"Building model XceptionBased...")
    model = get_xception_based_model()
    model.load_state_dict(
        torch.load("/home/eeproj5/Documents/solution_camtek/solution/checkpoints/AD_CTW_X5_16_XceptionBased_Adam.pt"))
    model = model.to(device)
    return model

def load_loss():
    loss = nn.CrossEntropyLoss(weight=torch.tensor([1, 1], dtype=torch.float).to(device))
    return loss




