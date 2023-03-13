import unittest
import pandas as pd
from ml.functions import DropPdColumns


class TestDropPdColumns(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        self.columns_to_drop = ['B', 'C']
        self.dropper = DropPdColumns(self.df, self.columns_to_drop)

    def test_drop_method_drops_columns_from_dataframe(self):
        self.dropper.drop()
        self.assertListEqual(list(self.dropper.pandas_df.columns), ['A'])


if __name__ == '__main__':
    unittest.main()
