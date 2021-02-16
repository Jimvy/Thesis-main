import argparse
import os
import sys
import itertools

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as tvtransforms

from datasets import DEFAULT_DATA_FOLDER

# mean = [0.485, 0.456, 0.406]  # From akamaster
# stddev = [0.229, 0.224, 0.225]

mean = [0.4914, 0.4822, 0.4465]  # From own computations
stddev = [0.2470, 0.2435, 0.2616]  # other values are .2023, .1994, .2010

labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class CIFAR10:
    def __init__(self, data_folder, pin_memory=False):
        self.data_folder = data_folder
        self.pin_memory = pin_memory

    def get_train_loader(self, batch_size, shuffle=True, num_workers=0,
                         use_random_crops=True, use_hflip=True):
        transforms = []
        if use_random_crops:
            transforms.append(tvtransforms.RandomCrop(32, padding=4))
        if use_hflip:
            transforms.append(tvtransforms.RandomHorizontalFlip())
        transforms.append(tvtransforms.ToTensor())
        transforms.append(tvtransforms.Normalize(mean, stddev))

        trainset = torchvision.datasets.CIFAR10(root=self.data_folder, train=True, download=False,
                                                transform=tvtransforms.Compose(transforms))
        trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=shuffle,
                                 num_workers=num_workers, pin_memory=self.pin_memory)
        return trainloader

    def get_test_loader(self, batch_size, shuffle=False, num_workers=0):
        transforms = [tvtransforms.ToTensor()]
        transforms.append(tvtransforms.Normalize(mean, stddev))

        testset = torchvision.datasets.CIFAR10(root=self.data_folder, train=False, download=False,
                                               transform=tvtransforms.Compose(transforms))
        testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=self.pin_memory)
        return testloader

    @staticmethod
    def get_labels():
        return labels


def compute_stats():
    # Compute the statistics for this dataset: mean, standard deviation
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trainset = torchvision.datasets.CIFAR10('~/datasets', train=True, download=False, transform=tvtransforms.ToTensor())
    trainloader = DataLoader(trainset, batch_size=2048, shuffle=False)

    print("First statistics: mean {} and stddev {}".format(
        trainset.data.mean(axis=(0, 1, 2)) / 255,
        trainset.data.std(axis=(0, 1, 2)) / 255
    ))

    sum_pixels = torch.zeros(3, device=device)
    sum_squared_pixels = torch.zeros(3, device=device)
    pixel_counts = 0
    batch_count = 0
    for train_batch in itertools.chain(trainloader):
        batch_sample, batch_label = train_batch[0].to(device), train_batch[1].to(device)
        sum_pixels += torch.sum(batch_sample, (0, 2, 3))
        sum_squared_pixels.add_(torch.sum(torch.square(batch_sample), (0, 2, 3)))
        pixel_counts += torch.numel(batch_sample) / 3
        batch_count += batch_sample.shape[0]
    # Now, let's compute the true statistics
    mean_pixel = sum_pixels / pixel_counts
    variance = (sum_squared_pixels / pixel_counts) - mean_pixel ** 2
    unbiased_variance = variance * (pixel_counts / (pixel_counts - 1))
    stddev_pixel = torch.sqrt(variance)
    unbiased_sttdev = torch.sqrt(unbiased_variance)  # Probably not correct...
    print("Statistics: mean {}, stddev {} (unbiased {}), variance {} (unbiased {})".format(
        mean_pixel, stddev_pixel, unbiased_sttdev, variance, unbiased_variance))
    sum_rems_squared = torch.zeros(3, device=device)
    for train_batch in itertools.chain(trainloader):
        batch_sample = train_batch[0].to(device)
        sum_rems_squared += torch.sum(torch.square(batch_sample - mean_pixel[None, :, None, None]), (0, 2, 3))
    variance2 = sum_rems_squared / pixel_counts
    stddev2 = torch.sqrt(variance2)
    print("New statistics: {} and {}".format(variance2, stddev2))


if __name__ == '__main__':
    print(sys.argv)
    if sys.argv[1] == 'compute_meanstd':
        compute_stats()
