"""
This module contains the implementation of a program for analyzing Titanic survival data.

The program uses a support vector machine (SVM)
model to predict survival outcomes based on various
features such as age, gender, and ticket class.
The program reads in a dataset of passenger information in CSV format,
trains the SVM model on a portion of the dataset, and evaluates the model's performance
on the remaining portion of the dataset.
The program also generates various visualizations of the data and model results.

The main entry point for the program is the `TitanicKernelSVMMain` class in this module. This class
provides methods for loading the data, training the model, and generating visualizations.
"""

import yaml
from pipeline.pipeline import TitanicKernelSVMPipeline


class TitanicKernelSVMMain:
    """
    A class for running the TitanicKernelSVMPipeline using a YAML configuration file.

    Attributes:
        train_ds_path (str): The path to the training dataset.
        test_ds_path (str): The path to the test dataset.
        output_path (str): The path to the output directory.

    Methods:
        yaml_loader: Loads a YAML configuration file and extracts the dataset and output paths.
        start: Starts the TitanicKernelSVMPipeline with the loaded dataset and output paths.
    """
    def __init__(self):
        """
        Initializes a new instance of the TitanicKernelSVMMain class with default attribute values.
        """
        self.train_ds_path = None
        self.test_ds_path = None
        self.output_path = None

    def yaml_loader(self):
        """
        Loads a YAML configuration file containing model properties, and extracts the train
        dataset path, test dataset path, and output directory path from it.

        Args:
            self: An instance of the class that contains the yaml_loader method.

        Returns:
            None.

        Raises:
            FileNotFoundError: If the specified YAML file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        with open('conf/model-properties.yaml', 'r', encoding='utf-8') as model_properties_yaml:
            config = yaml.load(model_properties_yaml, Loader=yaml.FullLoader)

        model_args = config['environment']['model-arguments']
        self.train_ds_path = str(model_args['train_ds_path'])
        self.test_ds_path = str(model_args['test_ds_path'])
        self.output_path = str(model_args['output_path'])

    def start(self):
        """
        Starts the TitanicKernelSVMPipeline by loading the required YAML configuration
        file and initializing the pipeline object with the specified train and test
        dataset paths and output directory path.

        Args:
            self: An instance of the class that contains the start method.

        Returns:
            None.

        Raises:
            Any exceptions raised by the TitanicKernelSVMPipeline.process method.
        """
        self.yaml_loader()
        TitanicKernelSVMPipeline(
            train_ds_path=self.train_ds_path,
            test_ds_path=self.test_ds_path,
            output_path=self.output_path
        ).process()


if __name__ == '__main__':
    TitanicKernelSVMMain().start()
