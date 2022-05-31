import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy

DATA_PATH = '../../data'  # write to this variable when importing this module from different directory context than assumed here
DATASET = 'StanfordCars'  # write to this variable if you wish to use another dataset
# hyperparams/parameters that need defining or tuning
CLASSIFIER_INPUT_SIZE = 0  # IMPORTANT: definitely specify this one.
BATCH_SIZE = 8
NUM_EPOCHS = 15
FEATURE_EXTRACT = True

def dataset_folders():
    return {
        'StanfordCars': os.path.join('StanfordCars', 'pytorch_structured_dataset'),
        'FGVC-Aircrafts': None  # todo: define folder
    }


def data_path():
    return DATA_PATH


def dataset_path():
    return os.path.join(data_path(), dataset_folders()[DATASET])


def transforms():
    # todo if relevant: adapt this per dataset?
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(CLASSIFIER_INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(CLASSIFIER_INPUT_SIZE),
            transforms.CenterCrop(CLASSIFIER_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


def dataset():
    # assumes a path pointing to a set of folders representing classes, and the samples within those
    # classes to be in their respective folders
    tforms = transforms()
    dataset = {}
    for x in ['train', 'val']:
        dataset[x] = datasets.ImageFolder(os.path.join(dataset_path(), x), tforms[x])
    return dataset


def dataloaders():
    ds = dataset()
    dataloaders = {}
    for x in ['train', 'val']:
        dataloaders[x] = torch.utils.data.DataLoader(ds[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    return dataloaders


# def finetune_classifier(classifier, criterion, optimizer, is_inception=False):
#     pass
