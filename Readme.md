
# Credit Card Fraud Detector

## Table of Contents
- [Project Structure](#project-structure)
- [Features and Dataset](#features-and-dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Sampling Techniques](#sampling-techniques)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Configuration and Results](#configuration-and-results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

## Project Structure 🗂️

The project is structured as follows:

```
.
├── 📁 configs_and_results/                     # Stores model configurations and evaluation results in JSON format
│   ├── 📂 LogisticRegression/                  # Subfolder for Logistic Regression results
│   ├── 📂 RandomForestClassifier/              # Subfolder for Random Forest results
│   ├── 📂 MLPClassifier/                       # Subfolder for MLP Classifier results
│   └── 📂 VotingClassifier/                    # Subfolder for Voting Classifier results
├── 📓 Credit Card Fraud Detection EDA.ipynb    # Exploratory Data Analysis (EDA) notebook
├── 📊 credit_fraud_utils_data.py               # Functions for loading data and handling sampling
├── 📊 credit_fraud_utils_Eval.py               # Functions for evaluating model performance
├── ⚙️ credit_fraud_utils_modeling.py           # Model initialization and saving configurations
├── 🚀 main.py                                  # Main script to run the model pipeline
├── 🗂️ test.csv                                 # Test dataset
├── 🗂️ train.csv                                # Training dataset
└── 🗂️ val.csv                                  # Validation dataset
```
## Exploratory Data Analysis (EDA) 📊

The EDA notebook (`Credit Card Fraud Detection EDA.ipynb`) provides a comprehensive analysis of the dataset. Key components include:

- **Data Visualization**: Insights into the distribution of features, correlation matrices.
- **Descriptive Statistics**: Summary statistics that highlight key characteristics of the data.
- **Class Distribution**: Analyzes the proportion of legitimate vs. fraudulent transactions to assess class imbalance.


## Sampling Techniques ⚖️

This project supports multiple resampling methods to handle class imbalance:
1. **OverSampling**: Duplicates minority class samples.
2. **UnderSampling**: Reduces the number of majority class samples.
3. **SMOTE**: Generates synthetic data for the minority class.
4. **Combination**: Combines UnderSampling with OverSampling or SMOTE.

## Preprocessing 🔄

Two preprocessing options are available:
1. **MinMaxScaler**: Scales features within a given range.
2. **StandardScaler**: Standardizes features by removing the mean and scaling to unit variance.

## Modeling 🧠

1. **Logistic Regression**: A linear model for classification.
2. **Random Forest Classifier**: An ensemble of decision trees.
3. **MLP Classifier**: A neural network classifier.
4. **Voting Classifier**: Combines predictions from multiple models for better performance.

## Evaluation 📈

Evaluation metrics include:
- **Precision-Recall Curve**: AUC-PR score to evaluate the model’s performance on imbalanced data.
- **F1 Score**: A balanced measure between precision and recall.
- **Optimal Threshold**: Based on the Precision-Recall curve to improve classification.

## Configuration and Results 💾

All configurations, models, and results are stored in JSON format under the `configs_and_results/` folder. Each JSON file logs:
- The sampling method used and class ratios
- Preprocessing steps
- Model hyperparameters
- Evaluation metrics for both training and testing

## How to Run 🏃‍♂️

### Run the Pipeline

You can run the entire pipeline using the `main.py` script. This script supports various command-line arguments that allow you to configure the experiment dynamically.

#### Example Command:
```bash
python main.py --dataset_path=train.csv --sampler=1 --ratio=0.002,0.004 --preprocessor=2 --model=4 --hyperparameters='[{"n_estimators": 100, "max_depth": 10}]'
```

This command:
- Loads the `train.csv` dataset
- Applies **OverSampling** with ratios of 0.002 and 0.004
- Uses the **StandardScaler** for feature scaling
- Trains a **Voting Classifier** using hyperparameters `n_estimators=100` and `max_depth=10`

### Command-Line Arguments

| Argument           | Description                                                                                      | Example                                                                                 |
|--------------------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|
| `--dataset_path`    | Path to the dataset (required).                                                                  | `--dataset_path=train.csv`                                                              |
| `--sampler`         | Sampling method: `1` for OverSampling, `2` for UnderSampling, `3` for SMOTE, etc.                | `--sampler=1`                                                                           |
| `--ratio`           | Sampling ratios for different methods (comma-separated).                                          | `--ratio=0.002,0.004`                                                                   |
| `--preprocessor`    | Preprocessing method: `1` for MinMaxScaler, `2` for StandardScaler (required).                    | `--preprocessor=2`                                                                      |
| `--model`           | Model to train: `1` for Logistic Regression, `2` for Random Forest, `3` for MLP, `4` for Voting. | `--model=4`                                                                             |
| `--hyperparameters` | A JSON string containing hyperparameters for the selected model.                                  | `--hyperparameters='[{"n_estimators": 100, "max_depth": 10}]'`                          |

## Dependencies 📦
- Python
- pandas
- scikit-learn
- imbalanced-learn
