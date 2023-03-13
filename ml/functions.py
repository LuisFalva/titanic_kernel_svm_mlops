"""
This module contains classes and methods for performing various data preparation and analysis tasks
using pandas, numpy, and other data science libraries.

Classes:
    PandasProfiler: A class for generating a pandas-profiling report for a pandas DataFrame.
    DropPdColumns: A class for dropping columns from a pandas DataFrame.
    FeatureEngine: A class for performing various feature engineering tasks on a pandas DataFrame.
    SetSplit: A class for splitting a dataset into training and test sets.

Functions:
    None.

Module Constants:
    None.

Example:
    # import the PandasProfiler and DropPdColumns classes from the module
    from my_module import PandasProfiler, DropPdColumns

    # create a DataFrame with columns 'A', 'B', 'C'
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

    # create a PandasProfiler object to profile the DataFrame
    profiler = PandasProfiler(df, 'My Profile Report')

    # generate the report and display it
    report = profiler.profiler()
    report.to_widgets()

    # create a DropPdColumns object to drop columns 'B' and 'C'
    dropper = DropPdColumns(df, ['B', 'C'])

    # drop the columns and print the resulting DataFrame
    dropper.drop()
    print(dropper.df)

    Output:
    A pandas-profiling report with an overview of the DataFrame,
    including statistics, distributions, and correlations.

    Cols droped from   A  B  C
    0  1  4  7
    1  2  5  8
    2  3  6  9 , cols: ['B', 'C']
       A
    0  1
    1  2
    2  3
"""

import logging
import numpy as np
import pandas as pd
import ydata_profiling as yp


class PandasProfiler:
    """
    A class for generating a pandas-profiling report for a pandas DataFrame.

    Attributes:
    df (pd.DataFrame): The DataFrame to profile.
    title (str): The title to display in the report.

    Methods:
    profiler(): Generate a pandas-profiling report for the DataFrame.
    save_report(file_path): Save the pandas-profiling report to a file.

    Example:
    # create a DataFrame with columns 'A', 'B', 'C'
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

    # create a PandasProfiler object to profile the DataFrame
    profiler = PandasProfiler(df, 'My Profile Report')

    # generate the report and display it
    report = profiler.profiler()
    report.to_widgets()

    # save the report to a file
    profiler.save_report('report.html')

    Output:
    A pandas-profiling report with an overview of the DataFrame,
    including statistics, distributions, and correlations. The report is also
    saved to a file in HTML format.
    """
    def __init__(self, pandas_df, title):
        self.pandas_df = pandas_df
        self.title = title

    def profiler(self):
        """Generate a profile report for the Pandas dataframe.

        Returns:
            A profile report (yp.ProfileReport object) of the Pandas dataframe.

        Raises:
            NameError: If the Pandas dataframe is not defined.

        Example:
            >> profiler = PandasProfiler(pandas_df=my_df, title="My Data")
            >> profile_report = profiler.profiler()
        """
        try:
            return yp.ProfileReport(self.pandas_df, title=self.title)
        except NameError as name_error:
            raise NameError(f"name '{self.pandas_df}' is not defined. {name_error}") from name_error

    def save_report(self, file_path):
        """Save the pandas-profiling report to a file.

        Args:
            file_path (str): The file path to save the report.

        Returns:
            None. The report is saved to the specified file path.

        Example:
            >> profiler = PandasProfiler(pandas_df=my_df, title="My Data")
            >> profile_report = profiler.profiler()
            >> profiler.save_report('report.html')
        """
        profile_report = self.profiler()
        profile_report.to_file(file_path)


class DropPdColumns:
    """
    A class for dropping columns from a pandas DataFrame.

    Attributes:
    df (pd.DataFrame): The DataFrame to drop columns from.
    columns_to_drop (list): A list of column names to drop from the DataFrame.
    axis (int): The axis along which to drop columns (0 for rows, 1 for columns).
    inplace (bool): Whether to modify the DataFrame in-place or return a copy.

    Methods:
    drop(): Drop the specified columns from the DataFrame.

    Example:
    # create a DataFrame with columns 'A', 'B', 'C'
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

    # create a DropPdColumns object to drop columns 'B' and 'C'
    dropper = DropPdColumns(df, ['B', 'C'])

    # drop the columns and print the resulting DataFrame
    dropper.drop()
    print(dropper.df)

    Output:
    Cols droped from   A  B  C
    0  1  4  7
    1  2  5  8
    2  3  6  9 , cols: ['B', 'C']
       A
    0  1
    1  2
    2  3
    """

    def __init__(self, pandas_df, columns_to_drop, axis=1, inplace=True):
        self.pandas_df = pandas_df
        self.columns_to_drop = columns_to_drop
        self._axis = axis
        self._inplace = inplace
        self._logger = logging.getLogger(__name__)

    def drop(self):
        """Drop specified columns from the Pandas dataframe.

        Returns:
            None. The specified columns are dropped from the Pandas dataframe.

        Raises:
            NameError: If the Pandas dataframe is not defined.

        Example:
            >> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            >> d = Drop(df=df, columns_to_drop=['B'], axis=1, inplace=True)
            >> d.drop()
            None
            >> df
               A
            0  1
            1  2
        """
        try:
            self.pandas_df.drop(self.columns_to_drop, axis=self._axis, inplace=self._inplace)
            self._logger.info("Cols dropped: %s", self.columns_to_drop)
        except NameError as name_error:
            raise NameError(f"name '{self.pandas_df}' is not defined. {name_error}") from name_error


class FeatureEngine:

    _logger = logging.getLogger(__name__)

    @staticmethod
    def _compute_random_num(mean, std, is_null):
        """
        Compute an array of random integers with a specified mean,
        standard deviation, and number of elements.

        Parameters:
        mean (int): The mean value of the random numbers.
        std (int): The standard deviation of the random numbers.
        is_null (int): The number of elements in the array to generate.

        Returns:
        rand (np.ndarray): An array of random integers with shape (is_null,).
        """
        rand = np.random.randint(mean - std, mean + std, size=is_null)
        return rand

    @staticmethod
    def _fill_nan(train_df, dataset, rand, column):
        """
        Replace NaN values in a column of a DataFrame with random integers.

        Parameters:
        train_df (pd.DataFrame): The training dataset used to generate random numbers.
        dataset (pd.DataFrame): The dataset to impute NaN values in.
        rand (np.ndarray): An array of random integers to fill NaN values with.
        column (str): The name of the column to impute NaN values in.

        Returns:
        None
        """
        slice_df = dataset[column].copy()
        slice_df[np.isnan(slice_df)] = rand
        dataset[column] = slice_df
        dataset[column] = train_df[column].astype(int)

    @staticmethod
    def check_nulls(env=None, **kwargs):
        """
        Check the number of null values in each column of each
        DataFrame in a dictionary of DataFrames.

        Parameters:
        **kwargs (dict): A dictionary of DataFrames to check null values in.

        Returns:
        print description of null counts marked with '**'
        """
        for df_name, df in kwargs.items():
            cols_to_check = df.columns
            for c in cols_to_check:
                chk_null = df[c].isnull().sum()
                if chk_null >= 1:
                    chk_null = f"{chk_null} **"
                if env == "vm":
                    print(f"no. of nulls for col '{c}' in set '{df_name}': {chk_null}")
                else:
                    FeatureEngine._logger.info(
                        f"no. of nulls for col '{c}' in set '{df_name}': {chk_null}"
                    )

    @staticmethod
    def nan_inputer(datasets: list, column: str):
        """
        Impute missing values in a column of multiple
        DataFrames by filling NaN values with random integers.

        Parameters:
        datasets (list): A list of DataFrames to impute missing values in.
        column (str): The name of the column to impute missing values in.

        Returns:
        None
        """
        train_df = datasets[0]
        test_df = datasets[1]
        for dataset in datasets:
            mean = train_df[column].mean()
            std = test_df[column].std()
            is_null = dataset[column].isnull().sum()
            rand = FeatureEngine._compute_random_num(mean, std, is_null)
            FeatureEngine._fill_nan(train_df, dataset, rand, column)

    @staticmethod
    def common_value_inputer(df, col: str, common_value: str):
        """
        Replace missing values in a column of a DataFrame with a specified common value.

        Parameters:
        df (pd.DataFrame): The DataFrame to impute missing values in.
        col (str): The name of the column to impute missing values in.
        common_value (str): The common value to use to fill missing values.

        Returns:
        None
        """
        df[col] = df[col].fillna(common_value)

    @staticmethod
    def mean_inputer(df, col: str):
        """
        Replace missing values in a column of a DataFrame with the mean value of the column.

        Parameters:
        df (pd.DataFrame): The DataFrame to impute missing values in.
        col (str): The name of the column to impute missing values in.

        Returns:
        df (pd.DataFrame): The modified DataFrame with missing values filled.
        """
        df = df.fillna(df[col].mean())
        return df


class SetSplit:

    def __init__(self, train, test, pd_module=pd):
        self._train = train
        self._test = test
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self._pd = pd_module

    def split(self, **kwargs):
        self.X_train = self._train.drop(kwargs["train_col"], axis=1)
        self.Y_train = self._train[kwargs["train_col"]]
        self.X_test = self._test.drop(kwargs["test_col"], axis=1).copy()
