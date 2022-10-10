import sys
sys.path.insert(0, './')
import numpy as np


import torch
import torch.nn as nn
from torchvision import datasets, transforms

from .utility import SubsetRandomSampler, SubsetSampler

class IndexedCIFAR10(datasets.CIFAR10):

    def __init__(self, root, train, download, transform):

        super(IndexedCIFAR10, self).__init__(root = root, train = train, transform = transform, download = download)

    def __getitem__(self, index):

        img, target = super(IndexedCIFAR10, self).__getitem__(index)

        return img, target, index

def cifar10(batch_size, valid_ratio = None, shuffle = True, augmentation = True, train_subset = None):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        ]) if augmentation == True else transforms.Compose([
        transforms.ToTensor()
        ])
    transform_valid = transforms.Compose([
        transforms.ToTensor()
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor()
        ])

    trainset = IndexedCIFAR10(root = './data/cifar10', train = True, download = True, transform = transform_train)
    validset = IndexedCIFAR10(root = './data/cifar10', train = True, download = True, transform = transform_valid)
    testset = IndexedCIFAR10(root = './data/cifar10', train = False, download = True, transform = transform_test)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if train_subset is None:
        instance_num = len(trainset)
        indices = np.random.permutation(list(range(instance_num)))
    else:
        indices = np.random.permutation(train_subset)
        instance_num = len(indices)
    print('%d instances are picked from the training set' % instance_num)

    if valid_ratio is not None and valid_ratio > 0.:
        split_pt = int(instance_num * valid_ratio)
        train_idx, valid_idx = indices[split_pt:], indices[:split_pt]
        if shuffle == True:
            train_sampler, valid_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(valid_idx)
        else:
            train_sampler, valid_sampler = SubsetSampler(train_idx), SubsetSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = train_sampler, num_workers = 4, pin_memory = True)
        valid_loader = torch.utils.data.DataLoader(validset, batch_size = batch_size, sampler = valid_sampler, num_workers = 4, pin_memory = True)
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)

    else:
        if shuffle == True:
            train_sampler = SubsetRandomSampler(indices)
        else:
            train_sampler = SubsetSampler(indices)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, sampler = train_sampler, num_workers = 4, pin_memory = True)
        valid_loader = None
        test_loader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)

    return train_loader, valid_loader, test_loader, classes
