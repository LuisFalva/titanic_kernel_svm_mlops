import os
import unittest
import pandas as pd
from pipeline.constants import PASSENGER_ID, NAME, TICKET, CABIN, AGE, EMBARKED, SEX, FARE, SURVIVED
from pipeline.pipeline import TitanicKernelSVMPipeline


class TestTitanicKernelSVMPipeline(unittest.TestCase):

    def setUp(self):
        self.train_ds_path = "pipeline/data/input/train.csv"
        self.test_ds_path = "pipeline/data/input/test.csv"
        self.output_path = "tests/pipeline/data/output"
        self.pipeline = TitanicKernelSVMPipeline(
            self.train_ds_path,
            self.test_ds_path,
            self.output_path
        )
        self._predictions_path = os.path.join(self.output_path, "predictions.csv")

    def test_preprocess_dataset(self):
        train_df = pd.DataFrame({
            PASSENGER_ID: [1, 2, 3],
            NAME: ["John Doe", "Jane Doe", "Bob Smith"],
            TICKET: ["123", "456", "789"],
            CABIN: ["A1", "B2", "C3"],
            AGE: [30, 20, None],
            EMBARKED: ["S", "C", "Q"],
            SEX: ["male", "female", "male"],
            FARE: [10, 20, None],
            SURVIVED: [0, 1, 1]
        })
        test_df = pd.DataFrame({
            PASSENGER_ID: [4, 5, 6],
            NAME: ["Mary Johnson", "Tom Smith", "Sarah Lee"],
            TICKET: ["999", "888", "777"],
            CABIN: ["D4", "E5", "F6"],
            AGE: [25, None, 35],
            EMBARKED: ["S", "C", "S"],
            SEX: ["female", "male", "female"],
            FARE: [30, None, 50]
        })

        train_df, test_df = self.pipeline.preprocess_dataset(train_df, test_df)

        self.assertEqual(train_df.shape, (3, 5))
        self.assertEqual(set(train_df.columns), {AGE, EMBARKED, SEX, FARE, SURVIVED})
        self.assertEqual(test_df.shape, (3, 5))
        self.assertEqual(set(test_df.columns), {PASSENGER_ID, AGE, EMBARKED, SEX, FARE})

    def test_persist(self):
        test_df = pd.DataFrame({
            PASSENGER_ID: [4, 5, 6],
            NAME: ["Mary Johnson", "Tom Smith", "Sarah Lee"],
            TICKET: ["999", "888", "777"],
            CABIN: ["D4", "E5", "F6"],
            AGE: [25, None, 35],
            EMBARKED: ["S", "C", "S"],
            SEX: ["female", "male", "female"],
            FARE: [30, None, 50]
        })
        self.pipeline._y_pred = [0, 1, 1]

        self.pipeline.persist(test_df, self.output_path)

        self.assertTrue(os.path.exists(self._predictions_path))

        predictions = pd.read_csv(self._predictions_path)
        self.assertEqual(set(predictions.columns), {PASSENGER_ID, SURVIVED})
        self.assertEqual(predictions.shape[0], test_df.shape[0])

        expected_predictions = pd.DataFrame({
            PASSENGER_ID: [4, 5, 6],
            SURVIVED: [0, 1, 1]
        })
        pd.testing.assert_frame_equal(predictions, expected_predictions, check_dtype=False)

    def test_process(self):
        test_df = pd.DataFrame({
            "PassengerId": [4, 5, 6],
            "Survived": [0, 0, 1]
        })

        self.pipeline.process()

        self.assertTrue(os.path.exists(self._predictions_path))

        predictions = pd.read_csv(self._predictions_path)
        self.assertEqual(set(predictions.columns), {PASSENGER_ID, SURVIVED})
        self.assertEqual(predictions.shape[1], test_df.shape[1])


if __name__ == '__main__':
    unittest.main()
