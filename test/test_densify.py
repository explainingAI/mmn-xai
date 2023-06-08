import unittest

import numpy as np

from mmn_xai.xai import utils


class TestDensify(unittest.TestCase):
    def test_densify_fn(self) -> None:
        """Test case: all classes but one."""
        image = np.zeros((10, 10), dtype=np.uint8)
        image[5:7, 5:7] = 1

        explanation = np.zeros_like(image)
        explanation[5:7, 5:7] = 1

        explanation[6, 6] = 2
        explanation[0:2, 0:2] = 1

        xai_densify = utils.densify(explanation, image, np.max, norm=False)
        summed = int(np.sum(xai_densify[5:7, 5:7]))  # 2x2x2

        self.assertAlmostEqual(summed, (2 * 2 * 2))
