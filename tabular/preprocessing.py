import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import dill
import os


def data_load(data_raw_path):

    """
    Docstring for data_load: loading the training data
    """


    df = pd.read_csv(data_raw_path,
                     low_memory=False,
                     na_values=["nd", "ND", "Nd", "nD"]
                     )

    return df

def data_cleanup(df):
    """
    Docstring for data_cleanup:

    param df: input dataframes
    return: clean_df
    """

    # drop duplicate entries
    df = df.drop_duplicates()

    # remove empty target entries
    target_col = 'Percent_Bleaching'
    df = df.dropna(subset=[target_col])

    # retireve numerical data columns
    df = df.apply(pd.to_numeric, errors="ignore")

    # remove features w/ 30%+ missing values
    missing_frac = df.isna().mean().sort_values(ascending=False)
    cols_2_drop = missing_frac[missing_frac >= 0.30].index.tolist()
    df = df.drop(columns=cols_2_drop)

    return df

def feature_selection(df):
    """
    Docstring for feature_selection:

    param df: cleaned df
    return: feature_selected X, target y
    """

    # store target separately, define numerical and categorical feature lists
    target_col = 'Percent_Bleaching'
    y = df[target_col]
    df = df.drop(columns=target_col)

    # numerical features to keep, hard-coded (for now)

    num_features = ['Latitude_Degrees', 'Longitude_Degrees',
                    'Date_Year', 'Date_Month',
                    'Distance_to_Shore', 'Turbidity', 'Cyclone_Frequency',
                    'Depth_m', 'ClimSST',
                    'Temperature_Kelvin', 'Windspeed', 'SSTA', 'SSTA_DHW',
                    'TSA', 'TSA_DHW']

    # categorical features to keep, hard-coded (for now)

    cat_features = ['Exposure']

    # splitting numerical and categorical

    num_cols_df = df[num_features]
    cat_cols_df = df[cat_features]

    # merging X

    X = pd.concat([num_cols_df, cat_cols_df], axis=1)

    return X, y


def time_encoder(X):
    """
    Docstring for time_encoder

    :param X: Description
    """

    if "Date_Month" in X.columns:
        X["month_sin"] = np.sin(2 * np.pi * X["Date_Month"]/12)
        X["month_cos"] = np.cos(2 * np.pi * X["Date_Month"]/12)

    if "Date_Year" in X.columns:
        X["year_norm"] = (X["Date_Year"] - X["Date_Year"].mean()) / X["Date_Year"].std()

    X = X.drop(['Date_Year', 'Date_Month'], axis=1)

    return X


def target_encoder(y):
    """
    Docstring
    """

    y_cat = ['healthy' if y == 0 else 'bleached' for y in y]
    mapping = {'healthy': 0, 'bleached': 1}
    y_encoded = [mapping[i] for i in y_cat]

    return y_encoded


def preproc(X):
    """
    Docstring for preprocessor

    :param X: Description
    """

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    ## introduce ctrl via if not or assert that no preprocessor was loaded before defining from scratch
    ## implement preprocessor loading from .dill for users

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("time_encoder", FunctionTransformer(time_encoder)),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler())
            ]), num_cols),

            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),

        ],
        remainder="drop"
        #remainder="passthrough"
    )

    X_preproc = preprocessor.fit_transform(X)

    # Enables serializing external/nested functions
    dill.settings['recurse'] = True

    preproc_save_path = os.path.join(os.path.expanduser('~'), "code", "Lucia-Cordero", "ReefSight-Project", "tabular_training")

    file_path = os.path.join(preproc_save_path, "preproc.dill")
    with open(file_path, "wb") as f:
        dill.dump(preprocessor, f)
        print(f"Saved Preprocessor to {file_path}")
        print("\n Warning: Load this preprocessor to process X_pred for any models trained on its output!")

    return X_preproc

def load_preproc():
    """
    Load preprocessor using dill.

    """

    preproc_path = os.path.join(os.path.expanduser('~'), "code", "Lucia-Cordero", "ReefSight-Project", "tabular_training")

    file_path = os.path.join(preproc_path, "preproc.dill")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessor not found in {preproc_path}")

    with open(file_path, "rb") as f:
        preproc = dill.load(f)

    print(f"Loaded Preprocessor from {file_path}")
    return preproc
