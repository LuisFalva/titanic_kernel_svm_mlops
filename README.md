# Kernel Titanic 🚢
This project is a Python package with a CLI to train and evaluate a machine learning model for the famous Kaggle's Titanic competition. The package includes functions for data cleaning, feature engineering, model training, and evaluation.

## Setup 🚀

Before running the package, make sure you have Python 3.6 or later installed on your system.

To set up the project, run the following command in your terminal:

```sh   
make setup
```

> **Note**
> This will create a virtual environment and install all the required dependencies.

## Usage 💻
To train and evaluate the model, run the following command:

```sh   
make run
```

> **Note**
> This will execute the run-model.sh script, which activates the virtual environment and runs the main script to train and evaluate the model.

The output will be similar to the following:

```bash   
█▄▀ █▀▀ █▀█ █▄░█ █▀▀ █░░   ▀█▀ █ ▀█▀ ▄▀█ █▄░█ █ █▀▀
█░█ ██▄ █▀▄ █░▀█ ██▄ █▄▄   ░█░ █ ░█░ █▀█ █░▀█ █ █▄▄

IPython could not be loaded!
2023-03-12 23:28:09,514 - pipeline.pipeline - INFO - Load Train Dataset: data/input/train.csv
2023-03-12 23:28:09,518 - pipeline.pipeline - INFO - Successful Load.
2023-03-12 23:28:09,518 - ml.functions - INFO - no. of nulls for col 'PassengerId' in set 'test_df': 0
2023-03-12 23:28:09,519 - ml.functions - INFO - no. of nulls for col 'Survived' in set 'test_df': 0
2023-03-12 23:28:09,519 - ml.functions - INFO - no. of nulls for col 'Pclass' in set 'test_df': 0
2023-03-12 23:28:09,519 - ml.functions - INFO - no. of nulls for col 'Name' in set 'test_df': 0
2023-03-12 23:28:09,519 - ml.functions - INFO - no. of nulls for col 'Sex' in set 'test_df': 0
2023-03-12 23:28:09,520 - ml.functions - INFO - no. of nulls for col 'Age' in set 'test_df': 177 **
2023-03-12 23:28:09,520 - ml.functions - INFO - no. of nulls for col 'SibSp' in set 'test_df': 0
2023-03-12 23:28:09,520 - ml.functions - INFO - no. of nulls for col 'Parch' in set 'test_df': 0
2023-03-12 23:28:09,520 - ml.functions - INFO - no. of nulls for col 'Ticket' in set 'test_df': 0
2023-03-12 23:28:09,520 - ml.functions - INFO - no. of nulls for
```
### Want to contribute? 🤔

Bring it on! If you have an idea or want to ask anything, or there is a bug you want fixed, you may open an [issue ticket](https://github.com/LuisFalva/titanic_kernel_svm_mlops/issues). You will find the guidelines to make an issue request there. Also, you can get a glimpse of [Open Source Contribution Guide best practices here](https://opensource.guide/).
Cheers 🍻!

### Support or Contact 📠

Having troubles? You can send a [DM](https://mail.google.com/mail/u/0/?tab=rm&ogbl#inbox?compose=CllgCJZZQVJHBJKmdjtXgzlrRcRktFLwFQsvWKqcTRtvQTVcHvgTNSxVzjZqjvDFhZlVJlPKqtg) and I’ll help you sort it out.

### License 📃

Released under the Apache License, version 2.0.
