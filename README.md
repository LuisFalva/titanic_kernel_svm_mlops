# Kernel Titanic 🚢
This project is a Python package with a CLI to train and evaluate a machine learning model for the famous [Kaggle's Titanic competition](https://www.kaggle.com/competitions/titanic). 
The package includes functions for data cleaning, feature engineering, model training, and evaluation.

## Setup 🚀

Before running the package, make sure you have Python 3.6 or later installed on your system.

To set up the project, run the following command in your terminal:

```sh   
make setup
```

> **Note**:
> This will create a virtual environment and install all the required dependencies.

## Usage 💻
To train and evaluate the model, run the following command:

```sh   
make run
```

> **Note**:
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

## Local Dev & Test 🛠️
Developers can run tests on local environment before pushing any new feature branch into remote. 
This can be useful for local development purposes, as it allows developers to test their changes in a containerized environment that closely matches the production environment.

```sh   
make test-all
```

For build your test docker image run:

```sh   
make docker-build tag=<your-dev-tag-name>
```

> **Warning**:
> Make sure to use the 'tag' argument to create a new docker image with your changes.

> **Note**
> Is stronlgy suggested to use the [EDA](https://github.com/LuisFalva/titanic_kernel_svm_mlops/blob/develop/notebooks/EDA.ipynb) jupyter noteboook with all the pipeline steps devided by each cell and quick explanation.

### Want to contribute? 🤔

Bring it on! If you have an idea or want to ask anything, or there is a bug you want fixed, you may open an [issue ticket](https://github.com/LuisFalva/titanic_kernel_svm_mlops/issues). You will find the guidelines to make an issue request there. Also, you can get a glimpse of [Open Source Contribution Guide best practices here](https://opensource.guide/).
Cheers 🍻!

### Support or Contact 📠

Having troubles? You can send a [DM](https://mail.google.com/mail/u/0/?tab=rm&ogbl#inbox?compose=CllgCJZZQVJHBJKmdjtXgzlrRcRktFLwFQsvWKqcTRtvQTVcHvgTNSxVzjZqjvDFhZlVJlPKqtg) and I’ll help you sort it out.

### References 📚
- [Titanic Solution (top 8%)](https://www.kaggle.com/code/akhileshthite/titanic-solution-top-8)
