import argparse
import json
import os
from credit_fraud_utils_data import Load_data, Get_Sampler, Get_preprocessor
from credit_fraud_utils_modeling import Get_model, save_config_and_results_json,find_and_save_best_config
from credit_fraud_utils_Eval import evalute_test, evalute_train, create_config_and_metrics
import warnings

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detector')

    parser.add_argument('--dataset_path', type=str, default=r"E:\ML homework\Projects\project 2 Credit Card Fraud Detection\Credit-Card-Fraud-Detector\Code\Data\train.csv", help='dataset path')

    parser.add_argument('--validation_path', type=str, default=r"E:\ML homework\Projects\project 2 Credit Card Fraud Detection\Credit-Card-Fraud-Detector\Code\Data\test.csv", help='Validation dataset path')

    parser.add_argument('--sampler', type=int, default=1, help=
    '''
    1 for OverSampling 
    2 for UnderSampling 
    3 for OverSampling using SMOTE  
    4 for UnderSampling followed by OverSampling 
    5 for UnderSampling followed by SMOTE
    ''')

    parser.add_argument('--ratio', type=str, default='0.002', help='Comma-separated list of ratios for sampling')

    parser.add_argument('--preprocessor', type=int, default=2, help=
    '''
    1 for MinMaxScaler
    2 for StandardScaler
    ''')

    parser.add_argument('--model', type=int, default=4, help=
    '''
    1 for LogisticRegression
    2 for RandomForestClassifier
    3 for MLP Classifier
    4 for Voting Classifier
    ''')

    parser.add_argument('--hyperparameters', type=str, default='[]', help='List of hyperparameter dictionaries')


    args = parser.parse_args()

    # Load hyperparameters from JSON string
    hyperparameters_list = json.loads(args.hyperparameters)


    # Directory to save results
    base_path = r"E:\ML homework\Projects\project 2 Credit Card Fraud Detection\Credit-Card-Fraud-Detector\configs_and_results\4 Voting Classifier"


    sampling_ratios= list(map(float, args.ratio.split(',')))

    experiment_number=1

    # Iterate over hyperparameter combinations
    for ratio in sampling_ratios:
        args.ratio=ratio
        for i, hyperparameters in enumerate(hyperparameters_list, start=1):
            config_and_metrics = create_config_and_metrics(args, hyperparameters)

            # Step 1: Load data
            X_train, T_train = Load_data(args.dataset_path)

            # Step 2: Preprocessing
            preprocessor = Get_preprocessor(args.preprocessor)
            X_train = preprocessor.fit_transform(X_train)

            # Step 3: Sampling
            size=len(T_train[T_train==0])
            Sampler = Get_Sampler(args.sampler, args.ratio,size)
            X_train_resampled, T_train_resampled = Sampler.fit_resample(X_train, T_train)

            # Step 4: Modeling
            model = Get_model(args.model, hyperparameters)
            model.fit(X_train_resampled, T_train_resampled)
            best_threshold, training_metrics = evalute_train(model, X_train_resampled, T_train_resampled)
            config_and_metrics["metrics"]["training"].update(training_metrics)

            # Step 5: Evaluation
            X_test, T_test = Load_data(args.validation_path)
            X_test = preprocessor.transform(X_test)
            testing_metrics = evalute_test(model, X_test, T_test, best_threshold)
            config_and_metrics["metrics"]["testing"].update(testing_metrics)

            # Step 6: Saving configs and results
            json_file_name = f"experiment_number_{experiment_number}.json"
            json_path = os.path.join(base_path, json_file_name)
            save_config_and_results_json(config_and_metrics, json_path=json_path)

            experiment_number+=1

    # Step 7: Find and save the best configuration
    find_and_save_best_config(base_path)
