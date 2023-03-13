import unittest
import pandas as pd
from ml.functions import SetSplit


class TestSetSplit(unittest.TestCase):
    def setUp(self):
        self.train_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9], 'Y': [0, 1, 0]})
        self.test_df = pd.DataFrame({'A': [4, 5, 6], 'B': [7, 8, 9], 'C': [10, 11, 12], 'Y': [1, 0, 1]})
        self.splitter = SetSplit(self.train_df, self.test_df)

    def test_split_method_splits_train_and_test_sets_correctly(self):
        self.splitter.split(train_col='Y', test_col='Y')
        self.assertListEqual(list(self.splitter.X_train.columns), ['A', 'B', 'C'])
        self.assertListEqual(list(self.splitter.X_test.columns), ['A', 'B', 'C'])
        self.assertListEqual(list(self.splitter.Y_train), [0, 1, 0])


if __name__ == '__main__':
    unittest.main()
