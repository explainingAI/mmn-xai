# -*- coding: utf-8 -*-
""" Suite of tests for the faithfullness metric.

"""
import unittest

import torch

from xai.metrics import faithfullness


class TestFaithfullness(unittest.TestCase):
    def test_perfect_saliency(self):
        """ Test case: the saliency map is perfectly faithfull. """
        sal_map = torch.zeros((2, 2))
        img = torch.zeros((2, 2))

        sal_map[1][1] = 1
        faith = faithfullness.faithfullness(img, sal_map,
                                            lambda x: (x[1][1] == 1).float().reshape((1, -1)),
                                            (1, 1), 1)

        self.assertAlmostEqual(faith, 1)

    def test_inverse_saliency(self):
        """ Test case: the saliency map is just inverse of the real thing"""
        sal_map = torch.ones((2, 2))
        img = torch.zeros((2, 2))

        sal_map[1][1] = 0
        faith = faithfullness.faithfullness(img, sal_map,
                                            lambda x: (x[1][1] == 1).float().reshape((1, -1)),
                                            (1, 1), 1)

        self.assertAlmostEqual(faith, -1)
