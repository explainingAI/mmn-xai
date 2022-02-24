# -*- coding: utf-8 -*-
""" Pytorch data loader for Mura dataset.
"""
import glob
import os

import numpy as np

from torch.utils.data import Dataset
from torchvision.io import read_image


class MuraImageDataset(Dataset):
    def __init__(self, path: str):
        file_names = glob.glob(path)
        labels = list(map(lambda x: x.split(os.path.sep)[-2], self.__file_names))
        is_train_set = list(map(lambda x: x.split(os.path.sep)[-3] == "train", self.__file_names))

        self.__labels_map = dict()

        for idx, unique_labels in enumerate(np.unique(labels)):
            self.__labels_map[unique_labels] = idx

        self.__file_names = file_names
        self.__labels = list(map(lambda x: self.__labels_map[x], labels))
        self.__is_train_set = is_train_set

    def __getitem__(self, index):
        img_path = self.__file_names[index]
        image = read_image(img_path)
        label = self.__labels[index]

        return image, label

    def __len__(self):
        return len(self.__file_names)
