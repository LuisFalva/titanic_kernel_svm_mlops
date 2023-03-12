import unittest
import pandas as pd
import ydata_profiling as yp
from ml.functions import PandasProfiler


class TestPandasProfiler(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        self.title = 'Test Profile Report'
        self.profiler = PandasProfiler(self.df, self.title)

    def test_profiler_method_returns_profile_report(self):
        report = self.profiler.profiler()
        self.assertTrue(isinstance(report, yp.profile_report.ProfileReport))


if __name__ == '__main__':
    unittest.main()
