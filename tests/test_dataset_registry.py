import unittest

from src.datasets.registry import registry


class DatasetRegistryTest(unittest.TestCase):
    def test_paper_vision_datasets_are_registered(self):
        expected = {
            "CIFAR100",
            "Flowers102",
            "PCAM",
            "FER2013",
            "Cars",
            "DTD",
            "GTSRB",
            "RESISC45",
            "SUN397",
            "SVHN",
        }

        self.assertTrue(expected.issubset(registry))


if __name__ == "__main__":
    unittest.main()
