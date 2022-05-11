# -*- coding: utf-8 -*-
""" Suite of tests for the faithfullness metric.

"""
import unittest
import os

import numpy as np

from uib_xai.data import dataset


def creates_dataset(removed_classes, one_hot_encoding=-1):
    return dataset.ImageDataset(
        [os.path.join(".", "un", "1.png"), os.path.join(".", "dos", "2.png"),
         os.path.join(".", "tres", "3.png")],
        lambda x: np.zeros((10, 10)), removed_classes=removed_classes,
        one_hot_encoding=one_hot_encoding)


class TestDataset(unittest.TestCase):
    def test_dataset_without_1_class(self):
        """ Test case: all classes but one. """
        dts = creates_dataset(['dos'])

        for img, _ in dts:
            self.assertAlmostEqual(img.shape[1], 10)
            break

    def test_dataset_without_2_class(self):
        """ Test case: all classes but two. """
        dts = creates_dataset(['dos', 'tres'])

        for img, _ in dts:
            self.assertAlmostEqual(img.shape[1], 10)
            break

    def test_dataset_all_classes(self):
        """ Test case: all classes """
        dts = creates_dataset(None)

        for img, _ in dts:
            self.assertAlmostEqual(img.shape[1], 10)
            break

    def test_dataset_sparce(self):
        dts = creates_dataset(None, 0)