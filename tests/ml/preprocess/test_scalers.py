import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml.preprocess.feature import Scalers


class TestScalers(unittest.TestCase):
    def setUp(self):
        self.x, self.y = load_iris(return_X_y=True)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x,
            self.y,
            test_size=0.2,
            random_state=42
        )
        self.scaler = Scalers()

    @staticmethod
    def fake_datasets():
        x_train = np.array([[1, 2], [3, 4], [5, 6]])
        x_test = np.array([[7, 8], [9, 10]])
        return x_train, x_test

    def assert_is_instance(self, transformed_x_train, transformed_x_test):
        self.assertIsInstance(transformed_x_train, np.ndarray)
        self.assertIsInstance(transformed_x_test, np.ndarray)

    @staticmethod
    def assert_array_almost_equal(transformed_x_train, expected_x_train, transformed_x_test, expected_x_test):
        np.testing.assert_array_almost_equal(transformed_x_train, expected_x_train)
        np.testing.assert_array_almost_equal(transformed_x_test, expected_x_test)

    @staticmethod
    def expected_scaler(x_train):
        expected_scaler = StandardScaler()
        expected_scaler.fit(x_train)
        return expected_scaler

    def test_fit_transform(self):
        scaler = Scalers()

        x_train, x_test = self.fake_datasets()

        transformed_x_train, transformed_x_test = scaler.fit_transform(x_train, x_test)
        self.assert_is_instance(transformed_x_train, transformed_x_test)

        expected_scaler = self.expected_scaler(x_train)
        expected_x_train, expected_x_test = expected_scaler.transform(x_train), expected_scaler.transform(x_test)
        self.assert_array_almost_equal(
            transformed_x_train,
            expected_x_train,
            transformed_x_test,
            expected_x_test
        )


if __name__ == '__main__':
    unittest.main()
