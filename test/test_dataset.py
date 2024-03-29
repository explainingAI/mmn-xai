""" Suite of tests for the faithfullness metric.

"""
import os
import unittest
from typing import List, Optional

import numpy as np

from mmn_xai.data import dataset


def creates_dataset(
    removed_classes: Optional[List[str]], one_hot_encoding: int = -1
) -> dataset.ImageDataset:
    return dataset.ImageDataset(
        [
            os.path.join(".", "un", "1.png"),
            os.path.join(".", "dos", "2.png"),
            os.path.join(".", "tres", "3.png"),
        ],
        lambda x: np.zeros((10, 10)),
        removed_classes=removed_classes,
        one_hot_encoding=one_hot_encoding,
    )


class TestDataset(unittest.TestCase):
    def test_dataset_without_1_class(self) -> None:
        """Test case: all classes but one."""
        dts = creates_dataset(["dos"])

        for img, _ in dts:
            self.assertAlmostEqual(img.shape[1], 10)
            break

    def test_dataset_without_2_class(self) -> None:
        """Test case: all classes but two."""
        dts = creates_dataset(["dos", "tres"])

        for img, _ in dts:
            self.assertAlmostEqual(img.shape[1], 10)
            break

    def test_dataset_all_classes(self) -> None:
        """Test case: all classes"""
        dts = creates_dataset(None)

        for img, _ in dts:
            self.assertAlmostEqual(img.shape[1], 10)
            break

    def test_dataset_sparce(self) -> None:
        creates_dataset(None, 0)

    def test_combine(self) -> None:
        dts1 = creates_dataset(["dos"])
        dts2 = creates_dataset(["dos", "tres"])

        dts_combine = dts2 + dts1

        for img, _ in dts_combine:
            self.assertAlmostEqual(img.shape[1], 10)
