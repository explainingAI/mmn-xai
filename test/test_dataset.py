# -*- coding: utf-8 -*-
""" Suite of tests for the faithfullness metric.

"""
import unittest
import os

import numpy as np

from uib_xai.data import dataset


class TestDataset(unittest.TestCase):
    def test_dataset_without_1_class(self):
        """ Test case: all classes but one. """
        dts = dataset.ImageDataset(
            [os.path.join(".", "un", "1.png"), os.path.join(".", "dos", "2.png"),
             os.path.join(".", "tres", "3.png")],
            lambda x: np.zeros((10, 10)), removed_classes=["dos"])

        for img in dts:
            break

    def test_dataset_without_2_class(self):
        """ Test case: all classes but two. """
        dts = dataset.ImageDataset(
            [os.path.join(".", "un", "1.png"), os.path.join(".", "dos", "2.png"),
             os.path.join(".", "tres", "3.png")],
            lambda x: np.zeros((10, 10)), removed_classes=["dos", "tres"])

        for img in dts:
            break
