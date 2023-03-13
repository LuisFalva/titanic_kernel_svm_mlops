import unittest
import pandas as pd
from pipeline.pipeline import TitanicKernelSVMPipeline
from pipeline.constants import PASSENGER_ID, SURVIVED


class TestTitanicKernelSVMPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.train_ds_path = "data/input/train.csv"
        self.test_ds_path = "data/input/test.csv"
        self.output_path = "data/output/predictions.csv"

    def test_pipeline_integration(self):
        # create a pipeline instance
        pipeline = TitanicKernelSVMPipeline(
            self.train_ds_path,
            self.test_ds_path,
            self.output_path
        )

        # run the pipeline
        pipeline.process()

        # load the predictions file
        predictions_df = pd.read_csv(self.output_path)

        # check that the predictions file has the expected format and contains valid predictions
        self.assertEqual(set(predictions_df.columns), {PASSENGER_ID, SURVIVED})
        self.assertTrue(predictions_df[SURVIVED].isin([0, 1]).all())


if __name__ == '__main__':
    unittest.main()
