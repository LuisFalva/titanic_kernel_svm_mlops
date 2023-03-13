import unittest
import pandas as pd
from ml.preprocess.feature import Encoder


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'gender': ['male', 'female', 'male', 'male', 'female'],
            'age': [20, 30, 25, 18, 40],
            'income': [50000, 70000, 60000, 40000, 90000]
        })
        self.encoder = Encoder(cols=['gender'])

    def test_fit_transform_single_column(self):
        transformed_df = self.encoder.fit_transform(self.data)
        expected_df = pd.DataFrame({
            'gender': [1, 0, 1, 1, 0],
            'age': [20, 30, 25, 18, 40],
            'income': [50000, 70000, 60000, 40000, 90000]
        })
        pd.testing.assert_frame_equal(transformed_df, expected_df)

    def test_fit_transform_multiple_columns(self):
        self.encoder = Encoder(cols=['gender', 'age'])
        transformed_df = self.encoder.fit_transform(self.data)
        expected_df = pd.DataFrame({
            'gender': [1, 0, 1, 1, 0],
            'age': [1, 3, 2, 0, 4],
            'income': [50000, 70000, 60000, 40000, 90000]
        })
        pd.testing.assert_frame_equal(transformed_df, expected_df)


if __name__ == '__main__':
    unittest.main()
