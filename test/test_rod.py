""" Suite of tests for the detector of out-of-distribution behavior.

Writen by: Miquel Miró-Nicolau (UIB), 2022.
"""
import unittest

import numpy as np

from mmn_xai.ood import miro as ood_detector


class TestRodDetector(unittest.TestCase):
    """Suite of tests for the ROD detector."""

    def test_extreme_behavior_verbose(self) -> None:
        """Test case: extreme behavior with verbose."""
        res = ood_detector.detect(
            lambda x: np.array([1, 0]), [np.zeros((2, 2))], verbose=True
        )

        self.assertAlmostEqual(res, 1)

    def test_extreme_behavior(self) -> None:
        """Test case: extreme behavior without verbose."""
        res = ood_detector.detect(
            lambda x: np.array([1, 0]), [np.zeros((2, 2))], verbose=False
        )

        self.assertTrue(res)

    def test_normal_behavior(self) -> None:
        """Test case: normal behavior"""
        res = ood_detector.detect(
            lambda x: np.array([0.5, 0.5]), [np.zeros((2, 2))], verbose=True
        )

        self.assertAlmostEqual(res, 0)
