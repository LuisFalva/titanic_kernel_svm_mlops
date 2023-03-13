import os
import logging
import pandas as pd
from ml.models.svm import SVM
from ml.load.datasets import DatasetLoader
from ml.preprocess.feature import Encoder, Scalers
from ml.functions import DropPdColumns, FeatureEngine, SetSplit, PandasProfiler
from pipeline.constants import PASSENGER_ID, NAME, TICKET, CABIN, AGE, EMBARKED, SEX, FARE, SURVIVED

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class TitanicKernelSVMPipeline:

    def __init__(self, train_ds_path, test_ds_path, output_path):
        self.train_ds_path = train_ds_path
        self.test_ds_path = test_ds_path
        self.output_path = output_path

        self._logger = logging.getLogger(__name__)
        self._feat = FeatureEngine()
        self._slrs = Scalers()
        self._svm = SVM()
        self._y_pred = None

    def _if_dir_exists(self, output_dir):
        if not os.path.exists(output_dir):
            self._logger.info(f"Creating '{output_dir}' local directory.")
            os.makedirs(output_dir)

    def preprocess_dataset(self, train_df, test_df):
        try:
            cols_to_drop_train = [
                PASSENGER_ID,
                NAME,
                TICKET,
                CABIN
            ]
            drop_train_cols = DropPdColumns(train_df, cols_to_drop_train)
            drop_train_cols.drop()
            cols_to_drop_test = [
                NAME,
                TICKET,
                CABIN
            ]
            drop_test_cols = DropPdColumns(test_df, cols_to_drop_test)
            drop_test_cols.drop()
            self._feat.nan_inputer([train_df, test_df], AGE)
            self._feat.common_value_inputer(train_df, EMBARKED, "S")
            test_df = self._feat.mean_inputer(test_df, FARE)
            train_df = Encoder([SEX, EMBARKED]).fit_transform(train_df)
            test_df = Encoder([SEX, EMBARKED]).fit_transform(test_df)
            return train_df, test_df
        except Exception as e:
            raise self._logger.error(f"Fatal Error On 'preprocess_dataset' Step. Trace: {e}")

    def persist(self, test_df, path):
        try:
            output_dir = os.path.dirname(path)
            self._if_dir_exists(output_dir)
            submission = pd.DataFrame({
                PASSENGER_ID: test_df[PASSENGER_ID],
                SURVIVED: self._y_pred
            })
            submission.to_csv(path, index=False)
        except Exception as e:
            raise self._logger.error(f"Fatal Error On 'persist' Step. Trace: {e}")

    def save_profiler_report(self, pandas_df, dataset_type: str):
        profiler = PandasProfiler(
            pandas_df=pandas_df,
            title=f"Pandas Profiler {dataset_type} Dataset"
        )
        base_path = "data/output/report"
        self._if_dir_exists(base_path)
        report_path = f"{base_path}/{dataset_type}-report.html"
        profiler.save_report(report_path)
        self._logger.info(f"Profiler Report Generated Successfully On: '{report_path}'.")

    def process(self):
        self._logger.info(f'Load Train Dataset: {self.train_ds_path}')
        train_loader = DatasetLoader(self.train_ds_path)
        train_df = train_loader.load()
        self._logger.info(f'Successful Load.')
        self._feat.check_nulls(test_df=train_df)

        self._logger.info(f'Generate Profiler Report for Train Dataset.')
        self.save_profiler_report(train_df, "train")

        self._logger.info(f'Load Test Dataset: {self.test_ds_path}')
        test_loader = DatasetLoader(self.test_ds_path)
        test_df = test_loader.load()
        self._logger.info(f'Successful Load.')
        self._feat.check_nulls(test_df=test_df)

        self._logger.info(f'Generate Profiler Report for Test Dataset.')
        self.save_profiler_report(test_df, "test")

        self._logger.info('Start Train & Test Datasets Preprocess')
        train_df, test_df = self.preprocess_dataset(train_df, test_df)
        self._logger.info('End Train & Test Datasets Preprocess')

        self._logger.info('Start Split Train & Test Datasets')
        s = SetSplit(train_df, test_df)
        s.split(train_col=SURVIVED, test_col=PASSENGER_ID)
        self._logger.info('End Split Train & Test Datasets')

        self._logger.info('Start Standard Scaler Transform')
        x_train, x_test = self._slrs.fit_transform(s.X_train, s.X_test)
        self._logger.info('End Standard Scaler Fit')

        self._logger.info('Start SVM Model Predict')
        self._y_pred = self._svm.fit_predict(x_train, x_test, s.Y_train)
        self._logger.info('End SVM Model Predict')

        self._svm.score(x_train, s.Y_train)
        self._logger.info('SVM  Score Model Predict')

        self.persist(test_df, self.output_path)
        self._logger.info(f'Successfully Persisted SVM Model Predictions: {self.output_path}')
