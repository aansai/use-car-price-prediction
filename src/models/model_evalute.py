import numpy as np
import pandas as pd
import os
import mlflow
import mlflow.sklearn
import logging
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error


logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def data_load(url):
    logging.info("Data Loading Start")
    df = pd.read_csv(url)
    logger.info("Data Loading Successfully")
    return df

cat_columns = ['brand', 'model', 'fuel_type', 'ext_col', 'int_col', 'accident',
       'transmission_clean']
num_columns = ['model_year', 'milage', 'horse_power', 'displacement_L','cylinder']

def split_data(df):
    X = df.drop(columns=['price'])
    y = np.log1p(df['price'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    logger.info("Train Size: %d | Test Size : %d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test

def build_proccesor(cat_columns, num_columns):
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])
    proccessor = ColumnTransformer([
        ('cat_cols', cat_pipe, cat_columns),
        ('num_cols', num_pipe, num_columns)
    ], remainder='drop')
    return proccessor

PARAMS = {
    "n_estimators"     : 1000,
    "learning_rate"    : 0.05,
    "max_depth"        : 6,
    "num_leaves"       : 50,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 1.5,
    "min_child_samples": 20,
    "random_state"     : 42,
    "test_size"        : 0.33,
}

def build_model(proccessor):
    pipe = Pipeline([
        ('preprocessor', proccessor),
        ('model', lgb.LGBMRegressor(
            n_estimators      = PARAMS["n_estimators"],
            learning_rate     = PARAMS["learning_rate"],
            max_depth         = PARAMS["max_depth"],
            num_leaves        = PARAMS["num_leaves"],
            subsample         = PARAMS["subsample"],
            colsample_bytree  = PARAMS["colsample_bytree"],
            reg_alpha         = PARAMS["reg_alpha"],
            reg_lambda        = PARAMS["reg_lambda"],
            min_child_samples = PARAMS["min_child_samples"],
            random_state      = PARAMS["random_state"],
            n_jobs=-1, verbose=-1
        ))
    ])
    return pipe

def model_evalute(pipe, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)

    r2_train   = r2_score(y_train, y_pred_train)
    r2_test    = r2_score(y_test,  y_pred_test)
    mse_train  = mean_squared_error(y_train, y_pred_train)
    mse_test   = mean_squared_error(y_test,  y_pred_test)
    rmse_train = np.sqrt(mse_train)
    rmse_test  = np.sqrt(mse_test)
    mape       = mean_absolute_percentage_error(y_test, y_pred_test)

    metrics = {
        "r2_train"  : round(r2_train,   4),
        "r2_test"   : round(r2_test,    4),
        "mse_train" : round(mse_train,  4),   
        "mse_test"  : round(mse_test,   4),
        "rmse_train": round(rmse_train, 4),
        "rmse_test" : round(rmse_test,  4),
        "mape"      : round(mape,       4),
    }

    results = pd.DataFrame([{
        'Model'     : 'LightGBM',
        'R2 Train'  : metrics["r2_train"],
        'R2 Test'   : metrics["r2_test"],
        'MSE Train' : metrics["mse_train"],
        'MSE Test'  : metrics["mse_test"],
        'RMSE Train': metrics["rmse_train"],
        'RMSE Test' : metrics["rmse_test"],
        'MAPE'      : metrics["mape"],
    }])

    return results, metrics, pipe

def save_data(data_path, X_train, X_test, y_train, y_test):
    logger.info("Saving data Path")
    save_path = os.path.join(data_path, "processed")
    os.makedirs(save_path, exist_ok=True)
    X_train.to_csv(os.path.join(save_path, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(save_path, "X_test.csv"),  index=False)
    y_train.to_csv(os.path.join(save_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(save_path, "y_test.csv"),  index=False)
    logger.info("Data Saved Successfully")

def main():
    mlflow.set_experiment("car_price_prediction")        
    with mlflow.start_run(run_name="LightGBM_baseline"):
        logger.info("MLflow run started")

        df = data_load(r'data\interim\df.csv')
        X_train, X_test, y_train, y_test = split_data(df)

        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples",  len(X_test))
        mlflow.log_param("n_features",    X_train.shape[1])

        mlflow.log_params(PARAMS)
        proccessor = build_proccesor(cat_columns, num_columns)
        pipe       = build_model(proccessor)
        logger.info("Training Started")

        results, metrics, trained_pipe = model_evalute(pipe, X_train, X_test, y_train, y_test)
        print(results.to_string(index=False))
        mlflow.log_metrics(metrics)
        logger.info("Metrics logged: %s", metrics)

        mlflow.sklearn.log_model(
            sk_model        = trained_pipe,
            artifact_path   = "lgbm_pipeline",
            registered_model_name = "CarPriceLightGBM"   
        )
        logger.info("Model logged to MLflow")

        save_data("data", X_train, X_test, y_train, y_test)
        mlflow.log_artifacts("data/processed", artifact_path="processed_data")

        results_path = "data/processed/results.csv"
        results.to_csv(results_path, index=False)
        mlflow.log_artifact(results_path, artifact_path="reports")

        logger.info("MLflow run finished. Run ID: %s", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()