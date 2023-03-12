import unittest
import numpy as np
import pandas as pd
from ml.functions import FeatureEngine


class TestFeatureEngine(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        self.test_df = pd.DataFrame({'A': [4, 5, 6], 'B': [7, 8, 9], 'C': [10, np.nan, 12]})
        self.datasets = [self.train_df, self.test_df]
        self.column = 'C'

    def test_compute_random_num_returns_array_of_integers(self):
        mean = 5
        std = 1
        is_null = 3
        rand = FeatureEngine._compute_random_num(mean, std, is_null)
        self.assertTrue(isinstance(rand, np.ndarray))
        self.assertEqual(rand.shape, (is_null,))

    def test_fill_nan_replaces_nan_values_with_random_integers(self):
        mean = self.train_df[self.column].mean()
        std = self.test_df[self.column].std()
        is_null = self.test_df[self.column].isnull().sum()
        rand = FeatureEngine._compute_random_num(mean, std, is_null)
        FeatureEngine._fill_nan(self.train_df, self.test_df, rand, self.column)
        self.assertFalse(self.test_df[self.column].isnull().any())

    def test_nan_inputer_imputes_missing_values_with_random_integers(self):
        FeatureEngine.nan_inputer(self.datasets, self.column)
        self.assertFalse(self.test_df[self.column].isnull().any())

    def test_common_value_inputer_replaces_missing_values_with_common_value(self):
        col = 'B'
        common_value = 'missing'
        df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, np.nan, np.nan], 'C': [7, 8, 9]})
        FeatureEngine.common_value_inputer(df, col, common_value)
        self.assertFalse(df[col].isnull().any())
        self.assertTrue((df[col] == common_value).all())

    def test_mean_inputer_replaces_missing_values_with_mean_of_column(self):
        col = 'B'
        df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, np.nan], 'C': [7, 8, 9]})
        df = FeatureEngine.mean_inputer(df, col)
        self.assertFalse(df[col].isnull().any())
        self.assertAlmostEqual(df[col][2], 4.0)


if __name__ == '__main__':
    unittest.main()
