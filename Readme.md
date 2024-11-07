# Credit Card Fraud Detector
## Project Overview 
The **Credit Card Fraud Detector** project aims to develop a robust machine learning model to identify fraudulent transactions in credit card datasets. By leveraging various sampling techniques, preprocessing methods, and classification algorithms, the project seeks to improve the accuracy and reliability of fraud detection. The insights gained from Exploratory Data Analysis (EDA) inform the modeling strategies .
## Table of Contents
- [Key Insights](#key-insights) 
- [Sampling Techniques](#sampling-techniques)
- [Modeling](#modeling)
- [Results](#Results)



## Key Insights 

These insights are derived from the Exploratory Data Analysis (EDA) conducted on the dataset:

1. **Class Imbalance**: The dataset reveals a significant imbalance between fraudulent and non-fraudulent transactions, necessitating techniques like oversampling or undersampling to improve model performance.

2. **Feature Distributions**: Several numerical features, particularly `Amount`, exhibit skewness and outliers, which could impact predictive modeling.

3. **Correlations**: The heatmap indicates that certain features have strong correlations with the target variable, highlighting potential predictors of fraud that should be prioritized in model training.

4. **Fraud Patterns**: Visualizations suggest identifiable patterns in fraudulent transactions, such as specific transaction amounts, which can inform targeted fraud detection strategies.

## Sampling Techniques 

1. **OverSampling**: Duplicates minority class samples.
2. **UnderSampling**: Reduces the number of majority class samples.
3. **SMOTE**: Generates synthetic data for the minority class.
4. **Combination**: Combines UnderSampling with OverSampling or SMOTE.

## Modeling 

1. **Logistic Regression**: A linear model for classification.
2. **Random Forest Classifier**: An ensemble of decision trees.
3. **MLP Classifier**: A neural network classifier.
4. **Voting Classifier**: Combines predictions from multiple models for better performance.

## Results 💾

| Model                     | Training F1 Score | Testing F1 Score | Training AUC-PR | Testing AUC-PR |
|---------------------------|--------------------|-------------------|------------------|------------------|
| Logistic Regression       | 0.809              | 0.802             | 0.655            | 0.646            |
| Random Forest Classifier  | 0.998              | 0.835             | 0.997            | 0.698            |
| MLP Classifier            | 0.902              | 0.794             | 0.814            | 0.634            |
| Voting Classifier         | 0.963              | 0.800             | 0.929            | 0.644            |

All configurations and results are stored in JSON format under the `configs_and_results/` folder.
