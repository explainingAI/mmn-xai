# -*- coding: utf-8 -*-
""" Pytorch data loader for Mura dataset.
"""
import os

import numpy as np

import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, file_names, get_img_fn, one_hot_encoding: int = -1):
        if one_hot_encoding > 1:
            raise ValueError(f"Selected option for one hot encoding not valid")
        labels = list(map(lambda x: x.split(os.path.sep)[-2], file_names))
        is_train_set = list(map(lambda x: x.split(os.path.sep)[-3] == "train", file_names))

        self.__get_img_fn = get_img_fn
        self.__labels_map = dict()

        for idx, unique_labels in enumerate(np.unique(labels)):
            self.__labels_map[unique_labels] = idx

        self.__file_names = file_names
        self.__labels = list(map(lambda x: self.__labels_map[x], labels))
        self.__is_train_set = is_train_set

        self.__one_hot_encoding = one_hot_encoding

    def __getitem__(self, index):
        img_path = self.__file_names[index]
        image = self.__get_img_fn(img_path)

        if self.__one_hot_encoding < 0:
            label = np.array([0] * len(self.__labels_map), dtype=np.float32)
            label[self.__labels[index]] = 1
        else:
            label = np.array([0], dtype=np.float32)
            label[0] = int(self.__labels[index] == self.__one_hot_encoding)

        return image, torch.from_numpy(label)

    def __len__(self):
        return len(self.__file_names)
