import os
import pandas as pd
import numpy as np


def preprocess() -> None:
    """
    - load training data
    - clean data
    - load preprocessor
    - preprocess training data
    """

    from preprocessing import data_load, data_cleanup, feature_selection, preproc, target_encoder

    # Retrieve `query` data as dataframe iterable
    data_raw_path = os.path.join(os.path.expanduser('~'), "code", "Lucia-Cordero", "ReefSight-Project", "tabular_data", "raw", "global_bleaching_environmental_raw.csv")
    data_processed_path = os.path.join(os.path.expanduser('~'), "code", "Lucia-Cordero", "ReefSight-Project", "tabular_data", "processed", "global_bleaching_environmental_processed.csv")

    data_processed_exists = os.path.isfile(data_processed_path)
    if data_processed_exists:
        print("Loading already processed training data from local CSV...")
        data_processed = pd.read_csv(data_processed_path)


        print(f"processed training data loaded from {data_processed_path}")
        print("preprocess() done")

    else:

        print("Loading raw data..")

        data = data_load(data_raw_path)

        print("Loading raw data..done")
        print("\n Starting raw data processing..")

        clean_data = data_cleanup(data)

        X, y = feature_selection(clean_data)

        X_preproc = preproc(X)

        y_encoded = target_encoder(y)

        y_encoded = np.array(y_encoded).reshape(-1, 1)

        data_processed = pd.DataFrame(np.concatenate((X_preproc, y_encoded), axis=1))

        data_processed.to_csv(data_processed_path,
                              mode='w',
                              header=True,
                              index=False)

        print(f"processed training data saved as {data_processed_path}")
        print("preprocess() done")



def train() -> None:
    """
     - load processed data
     - train-test split data
     - train logistic regresssors and tree-based classifier via GridSearch for hyperparameter optimization
     - extract performance metrics and save separately on local disk
     - save 'best_model' (for each classifier?)
    """

    from modeling import clf_optimization_with_gridsearch, save_models
    from sklearn.model_selection import train_test_split

    data_processed_path = os.path.join(os.path.expanduser('~'), "code", "Lucia-Cordero", "ReefSight-Project", "tabular_data", "processed", "global_bleaching_environmental_processed.csv")

    print(f"Loading preprocessed training data from  {data_processed_path}")

    data_processed_exists = os.path.isfile(data_processed_path)
    if not data_processed_exists:
        print("No preprocessed data available. Please run 'preprocess()' and try again.")


    else:
        data_processed = pd.read_csv(data_processed_path)
        print("Loading preprocessed training data done")

        X_preproc = data_processed.iloc[:, :-1]
        y_encoded = data_processed.iloc[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(
            X_preproc, y_encoded, test_size=0.2, random_state=42
            )

        print("Starting training classification models")
        print("This may take a while...")

        df_results, fitted_models = clf_optimization_with_gridsearch(X_train, X_test, y_train, y_test)

        print("Training done! Here are your performance metrics:")
        print(df_results)

        data_training_path = os.path.join(os.path.expanduser('~'), "code", "Lucia-Cordero", "ReefSight-Project", "tabular_training")
        df_results.to_csv(os.path.join(data_training_path, "performance_metrics.csv"),
                          mode='w',
                          header=True,
                          index=False)

        save_models(fitted_models)



def predict(lat, lon, dt, X_pred: pd.DataFrame = None) -> np.ndarray:

    print("\nUse case: predicting coral bleaching risk")
    print("\nImporting modules...")

    from preprocessing import load_preproc
    from modeling import load_model
    from predicting_v2 import erddap_extract, fetch_sst_range, compute_weekly_clim_max_parallel, fetch_environmental_variables, haversine, distance_to_shore, depth_from_opentopo, infer_region, _endpoint, _fetch_direction, _compute_fetch, _classify, classify_exposure, cyclone_frequency, turbidity, windspeed



    from functools import lru_cache
    import numpy as np
    import pandas as pd
    import requests
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    from shapely.ops import nearest_points
    from datetime import datetime, timedelta
    #from math import radians, sin, cos, asin, sqrt, atan2
    from functools import lru_cache
    from io import StringIO
    import math
    import xarray as xr
    from pyproj import Geod

    coast = gpd.read_file("/home/nico_kas/code/Lucia-Cordero/ReefSight-Project/tabular/gshhg-shp-2.3.7/GSHHS_h_L1.shp")
    coast = coast.to_crs("EPSG:4326")

    if isinstance(dt, str):
        dt = datetime.strptime(dt, "%Y-%m-%d")

    print("\nFetching environmental data...")

    fetch_errors = {}
    results = {}

    # --- Individual fetches with per-variable error capture ---

    try:
        env = fetch_environmental_variables(lat, lon, dt)
        results["env"] = env
        print("  ✓ SST/SSTA/DHW/TSA variables")
    except Exception as e:
        fetch_errors["env"] = str(e)
        print(f"  ✗ SST/SSTA/DHW/TSA variables: {e}")

    try:
        dist = distance_to_shore(lat, lon, coast)
        results["dist"] = dist
        print("  ✓ Distance to shore")
    except Exception as e:
        fetch_errors["dist"] = str(e)
        print(f"  ✗ Distance to shore: {e}")

    try:
        depth = depth_from_opentopo(lat, lon)
        results["depth"] = depth
        print("  ✓ Depth")
    except Exception as e:
        fetch_errors["depth"] = str(e)
        print(f"  ✗ Depth: {e}")

    try:
        exp = classify_exposure(lat, lon, coast)
        results["exp"] = exp
        print("  ✓ Exposure")
    except Exception as e:
        fetch_errors["exp"] = str(e)
        print(f"  ✗ Exposure: {e}")

    try:
        turb = turbidity(lat, lon)
        results["turb"] = turb
        print("  ✓ Turbidity")
    except Exception as e:
        fetch_errors["turb"] = str(e)
        print(f"  ✗ Turbidity: {e}")

    try:
        cyc = cyclone_frequency(lat, lon)
        results["cyc"] = cyc
        print("  ✓ Cyclone frequency")
    except Exception as e:
        fetch_errors["cyc"] = str(e)
        print(f"  ✗ Cyclone frequency: {e}")

    try:
        wind = windspeed(lat, lon, dt)
        results["wind"] = wind
        print("  ✓ Windspeed")
    except Exception as e:
        fetch_errors["wind"] = str(e)
        print(f"  ✗ Windspeed: {e}")

    # --- Abort if any critical variable failed ---
    if fetch_errors:
        print("\n⚠ Prediction aborted. The following variables could not be fetched:")
        for var, err in fetch_errors.items():
            print(f"  - {var}: {err}")
        print("\nPossible causes: ERDDAP server timeout, date out of range, or coordinates outside dataset coverage.")
        return None

    print("\nAll environmental data fetched successfully.")

# --- Build prediction dataframe ---
    if X_pred is None:
        env = results["env"]
        X_pred = pd.DataFrame(dict(
            Latitude_Degrees=[lat],
            Longitude_Degrees=[lon],
            Date_Year=[dt.year],
            Date_Month=[dt.month],
            Distance_to_Shore=[results["dist"]],
            Turbidity=[results["turb"]],
            Cyclone_Frequency=[results["cyc"]],
            Depth_m=[results["depth"]],
            Exposure=[results["exp"]],
            ClimSST=[env["ClimSST"]],
            Temperature_Kelvin=[env["Temperature_Kelvin"]],
            Windspeed=[results["wind"]],
            SSTA=[env["SSTA"]],
            SSTA_DHW=[env["SSTA_DHW"]],
            TSA=[env["TSA"]],
            TSA_DHW=[env["TSA_DHW"]]
        ))


    # --- Model inference ---

    preproc = load_preproc()
    #pass name of clf?
    model = load_model('RandomForestClassifier')
    #feature_names = X_pred.columns.tolist()
    X_processed = preproc.transform(X_pred)
    #X_processed = pd.DataFrame(X_processed, columns=feature_names)
    y_pred = model.predict(X_processed)
    y_proba = model.predict_proba(X_processed)[0]
    print(f"Probability class 0: {y_proba[0]:.3f}")
    print(f"Probability class 1: {y_proba[1]:.3f}")
    print(f"Predicted class: {int(y_pred[0])}")

    if int(y_pred[0]) == 0:
        print("The corals at your diving site were healthy at the time of your visit.")
    else:
        print("The corals at your diving site were suffering from bleaching at the time of your visit.")


    print(f"pred() done")

    return y_pred


#if __name__ == '__main__':
#    try:
#        # preprocess()
#        # train()
#        # pred()
#    except:
#        import sys
#        import traceback
#
#        import ipdb
#        extype, value, tb = sys.exc_info()
#        traceback.print_exc()
#        ipdb.post_mortem(tb)
