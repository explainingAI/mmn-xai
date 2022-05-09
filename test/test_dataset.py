# -*- coding: utf-8 -*-
""" Suite of tests for the faithfullness metric.

"""
import unittest

import numpy as np
import torch

from uib_xai.data import dataset


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        """ Test case: the saliency map is perfectly faithfull. """
        dts = dataset.ImageDataset([".\\un\\1.png", ".\\dos\\2.png", ".\\tres\\3.png"],
                                     lambda x: np.zeros((10, 10)), removed_classes=["dos"])

        for img in dts:
            break