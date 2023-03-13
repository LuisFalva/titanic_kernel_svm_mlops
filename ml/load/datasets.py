import pandas as pd


class DatasetLoader:
    """
    A class for loading a CSV file into a pandas DataFrame.

    Attributes:
    file_path (str): The path to the CSV file to load.

    Methods:
    load(): Load the CSV file into a pandas DataFrame.

    Example:
    # create a DatasetLoader object for a file called 'my_data.csv'
    loader = DatasetLoader('my_data.csv')

    # load the file into a pandas DataFrame and display the first 5 rows
    df = loader.load()
    print(df.head())

    Output:
       Column 1  Column 2  Column 3
    0         1         4         7
    1         2         5         8
    2         3         6         9
    """
    
    def __init__(self, file_path):
        self._file_path = file_path
        
    def load(self):
        try:
            load_df = pd.read_csv(self._file_path)
            return load_df
        except FileNotFoundError as fnf:
            raise FileNotFoundError(f"File from 'file_path' does not exists. Provide a valid path. {fnf}") from fnf
