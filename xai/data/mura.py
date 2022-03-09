# -*- coding: utf-8 -*-
""" Pytorch data loader for Mura dataset.
"""
import glob
import os

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode
from torchvision.io import read_image


class MuraImageDataset(Dataset):
    def __init__(self, path: str):
        file_names = glob.glob(path)
        labels = list(map(lambda x: x.split(os.path.sep)[-2], file_names))
        is_train_set = list(map(lambda x: x.split(os.path.sep)[-3] == "train", file_names))

        self.__labels_map = dict()

        for idx, unique_labels in enumerate(np.unique(labels)):
            self.__labels_map[unique_labels] = idx

        self.__file_names = file_names
        self.__labels = list(map(lambda x: self.__labels_map[x], labels))
        self.__is_train_set = is_train_set

    def __getitem__(self, index):
        img_path = self.__file_names[index]
        image = read_image(img_path, ImageReadMode.GRAY)

        label = np.array([0, 0], dtype=np.float32)
        label[self.__labels[index]] = 1

        return image, torch.from_numpy(label)

    def __len__(self):
        return len(self.__file_names)
