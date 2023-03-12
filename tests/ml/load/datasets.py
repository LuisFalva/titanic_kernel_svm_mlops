import os
import unittest
import pandas as pd
from ml.load.datasets import DatasetLoader


class TestDatasetLoader(unittest.TestCase):
    def setUp(self):
        test_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(test_dir, 'data', 'input', 'test_data.csv')
        self.loader = DatasetLoader(data_path)

    def tearDown(self):
        pass

    def test_load(self):
        df = self.loader.load()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 9)

        self.loader = DatasetLoader('invalid_path.csv')
        with self.assertRaises(FileNotFoundError):
            self.loader.load()


if __name__ == '__main__':
    unittest.main()
