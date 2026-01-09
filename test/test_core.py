import math
import unittest

from utils import setup_fake_environment


setup_fake_environment()

from boundml.core.utils import shifted_geometric_mean


class ShiftedGeometricMeanTests(unittest.TestCase):
    def test_positive_values(self):
        values = [1.0, 2.0, 4.0]
        result = shifted_geometric_mean(values, shift=1.0)
        expected = math.exp(sum(math.log(v + 1.0) for v in values) / len(values)) - 1.0
        self.assertAlmostEqual(result, expected)

    def test_handles_empty_iterable(self):
        self.assertTrue(math.isnan(shifted_geometric_mean([], shift=1.0)))


if __name__ == "__main__":
    unittest.main()
