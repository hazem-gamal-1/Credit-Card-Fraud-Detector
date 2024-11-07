from sklearn.metrics import precision_recall_curve,average_precision_score,classification_report,balanced_accuracy_score,accuracy_score,f1_score
import  numpy as np
def evalute_train(model,X,y_true):
    precision,recall,thresholds=precision_recall_curve(y_true,model.predict_proba(X)[:,1])
    f1_scores=(2*precision*recall)/(precision+recall)
    best_threshold=np.argmax(f1_scores)
    y_pred=(model.predict_proba(X)[:,1]>=thresholds[best_threshold]).astype(int)
    auc_pr=average_precision_score(y_true, y_pred)
    training_metrics = {
        "best_threshold": float(thresholds[best_threshold]),
        "f1_score": float(f1_scores[best_threshold]),
        "auc_pr": float(auc_pr),
    }

    print()
    print("train Evalution")
    print(f"Best threshold {thresholds[best_threshold]}")
    print(f"f1_score {f1_scores[best_threshold]}")
    return thresholds[best_threshold],training_metrics








def evalute_test(model,X,y_true,best_threshold):
    y_pred = (model.predict_proba(X)[:, 1] >= best_threshold).astype(int)
    auc_pr=average_precision_score(y_true, y_pred)
    f1=f1_score(y_true,y_pred)
    testing_metrics = {
        "f1_score": float(f1),
        "auc_pr": float(auc_pr),
    }

    print()
    print("Test Evalution")
    print(f"f1_score: {f1}")
    return testing_metrics



def create_config_and_metrics(args, hyperparameters):
    sampler_types = [
        "OverSampling",
        "UnderSampling",
        "SMOTE",
        "UnderSampling followed by OverSampling",
        "UnderSampling followed by SMOTE"
    ]

    config_and_metrics = {
        "model_configuration": {
            "sampler": {
                "type": sampler_types[args.sampler - 1],
                "ratio of the positive class": args.ratio
            },
            "preprocessing": ["MinMaxScaler", "StandardScaler"][args.preprocessor - 1],
            "model": ["LogisticRegression", "RandomForestClassifier", "MLPClassifier", "VotingClassifier"][args.model - 1],
            "hyperparameters": hyperparameters,
        },
        "metrics": {
            "training": {},
            "testing": {}
        }
    }
    return config_and_metrics
