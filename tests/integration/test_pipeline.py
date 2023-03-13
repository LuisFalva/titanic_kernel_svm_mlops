import os
import unittest
import pandas as pd
from main import TitanicKernelSVMMain
from pipeline.constants import PASSENGER_ID, SURVIVED


class TestTitanicKernelSVMPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.output_path = "tests/integration/data/output"

    def test_pipeline_integration(self):
        # create a pipeline instance
        main = TitanicKernelSVMMain(self.output_path)

        # run the pipeline
        main.start()

        # load the predictions file
        predictions_path = os.path.join(self.output_path, "predictions.csv")
        predictions_df = pd.read_csv(predictions_path)

        # check that the predictions file has the expected format and contains valid predictions
        self.assertEqual(set(predictions_df.columns), {PASSENGER_ID, SURVIVED})
        self.assertTrue(predictions_df[SURVIVED].isin([0, 1]).all())


if __name__ == '__main__':
    unittest.main()
