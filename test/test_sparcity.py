""" Suite of tests for the sparcity metric module.

Written by: Miquel Miró Nicolau (UIB)
"""
import unittest

import numpy as np

from mmn_xai.metrics import sparcity


class TestSparcity(unittest.TestCase):
    def test_sparcity_zero(self) -> None:
        self.assertAlmostEqual(sparcity.sparcity(np.zeros((10, 10))), 0)

    def test_sparcity_one(self) -> None:
        self.assertAlmostEqual(sparcity.sparcity(np.ones((10, 10))), 1)

    def test_sparcity_half(self) -> None:
        saliency_map = np.zeros((10, 10))
        saliency_map[0:5, 0:5] = 255

        self.assertAlmostEqual(sparcity.sparcity(saliency_map), 4)
