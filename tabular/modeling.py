from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import dill
import os
import pandas as pd


def define_model_parameters():
    """
    Docstring
    """


    models = {
        "RandomForestClassifier": RandomForestClassifier(
            random_state=42,
            n_jobs=-1),

        #"GradientBoostingClassifier": GradientBoostingClassifier(
        #    random_state=42),

        "XGBClassifier": xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42),

        "LogisticRegression": LogisticRegression(
            penalty="l2",
            solver="lbfgs",
            max_iter=5000,
            random_state=42)
    }

    param_grid = {
        "RandomForestClassifier": {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },

        #"GradientBoostingClassifier": {
        #    "n_estimators": [100, 200, 400],
        #    "learning_rate": [0.01, 0.05, 0.1, 0.5],
        #    "max_depth": [None, 2, 5],
        #},

        "XGBClassifier": {
            "n_estimators": [100, 200, 400],
            "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.5],
            "max_depth": [None, 2, 5, 10],
            "subsample": [0.5, 0.8, 1.0],  ## re-check documentation
        },

        "LogisticRegression": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
        }
    }

    return models, param_grid

def clf_optimization_with_gridsearch(X_train, X_test, y_train, y_test):
    results = []
    fitted_models = {}  # to store best estimators for all (!) models

    models, param_grid = define_model_parameters()

    for name, model in models.items():
        print(f"\n============================")
        print(f"Tuning + Training {name}...")
        print(f"============================")

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grid[name],
            cv=3,
            scoring="f1",         # best metric for balanced classes
            n_jobs=-1,
            verbose=2
        )

        grid.fit(X_train, y_train)

        # Store each (!) best model by name instead of overwriting
        fitted_models[name] = grid.best_estimator_

        # pass current best model (clf being grid searched in this iteration) to 'best_model'
        best_model = fitted_models[name]
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Best_Params": grid.best_params_,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_prob),
            "Pipeline": best_model,
        }

        results.append(metrics)

    df_results = pd.DataFrame(results).drop(columns=["Pipeline"])
    return df_results, fitted_models


def save_models(fitted_models) -> None:
    """
    Save each model in a dictionary to disk using dill.

    Parameters
    ----------
    model_dict : dict
        Dictionary where keys are model names (str)
        and values are model objects.
    """

    #os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(os.path.expanduser('~'), "code", "Lucia-Cordero", "ReefSight-Project", "tabular_training")

    for name, model in fitted_models.items():
        file_path = os.path.join(model_save_path, f"{name}.dill")
        with open(file_path, "wb") as f:
            dill.dump(model, f)
        print(f"Saved model '{name}' to {file_path}")



def load_model(model_name):
    """
    Load a single model by name using dill.

    Parameters
    ----------
    model_name : str
        Name of the model to load (without file extension).
    Returns
    -------
    The loaded model object.
    """

    model_path = os.path.join(os.path.expanduser('~'), "code", "Lucia-Cordero", "ReefSight-Project", "tabular_training")

    file_path = os.path.join(model_path, f"{model_name}.dill")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model '{model_name}' not found in {model_path}")

    with open(file_path, "rb") as f:
        model = dill.load(f)

    print(f"Loaded model '{model_name}' from {file_path}")
    return model
