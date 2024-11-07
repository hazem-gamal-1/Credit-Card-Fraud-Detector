from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.model_selection import GridSearchCV
import json
import os
def Get_model(option, hyperparams):
    if option == 1:
        model = LogisticRegression(**hyperparams)
    elif option == 2:
        model = RandomForestClassifier(**hyperparams)
    elif option == 3:
        model = MLPClassifier(**hyperparams)
    elif option == 4:

        logistic_model = LogisticRegression(**hyperparams.get('logistic_params', {}))
        rf_model = RandomForestClassifier(**hyperparams.get('rf_params', {}))
        mlp_model = MLPClassifier(**hyperparams.get('mlp_params', {}))


        model = VotingClassifier(estimators=[
            ('lr', logistic_model),
            ('rf', rf_model),
            ('mlp', mlp_model)
        ], voting='soft')
    else:
        raise ValueError("Invalid model option")
    return model

def save_config_and_results_json(config, json_path):
    with open(json_path, "w") as file:
        json.dump(config, file, indent=2)


def find_and_save_best_config(directory, output_file_name="best_configuration.json"):
    best_f1_score = -1
    best_config = None

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                    config = json.load(file)


                    f1_score = config.get("metrics", {}).get("testing", {}).get("f1_score", None)


                    if f1_score is not None and f1_score > best_f1_score:
                        best_f1_score = f1_score
                        best_config = config



    output_path = os.path.join(directory, output_file_name)
    with open(output_path, 'w') as output_file:
        json.dump(best_config, output_file, indent=4)

