import logging
import unittest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ml.models.svm import SVM


class TestSVM(unittest.TestCase):
    def setUp(self):
        self.x, self.y = load_iris(return_X_y=True)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                                random_state=42)
        self.feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        self.svm = SVM(kernel='linear', random_state=0)
        self.logger = logging.getLogger(__name__)

    def test_score(self):
        self.svm.fit_predict(self.x_train, self.x_test, self.y_train)
        y_pred = self.svm.model.predict(self.x_train)
        acc_score = accuracy_score(self.y_train, y_pred)
        self.svm.score(self.x_train, self.y_train)
        self.assertAlmostEqual(acc_score, 0.975, places=3)

    def test_fit_predict(self):
        self.svm.fit_predict(self.x_train, self.x_test, self.y_train)
        y_pred = self.svm.model.predict(self.x_test)
        self.assertIsInstance(y_pred, np.ndarray)

    def test_set_params(self):
        self.svm.set_params(kernel='rbf', C=1.0)
        self.assertEqual(self.svm.model.kernel, 'rbf')
        self.assertAlmostEqual(self.svm.model.C, 1.0, places=3)


if __name__ == '__main__':
    unittest.main()
