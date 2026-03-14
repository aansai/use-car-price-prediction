import numpy as np
import pandas as pd
import os
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
from sklearn.metrics import mean_absolute_error

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
    logger.info("Train Size: %d | Test Size : %d",len(X_train),len(X_test))
    return X_train, X_test, y_train, y_test

def build_proccesor(cat_columns,num_columns):
    cat_pipe = Pipeline(
    [
        ('imputer',SimpleImputer(strategy='most_frequent')),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', 
                           unknown_value=-1))
    ]
)
    num_pipe = Pipeline(
    [
        ('imputer',SimpleImputer(strategy='median')),
        ('scale',StandardScaler())
    ]
)
    proccessor = ColumnTransformer(
    [
        ('cat_cols',cat_pipe,cat_columns),
        ('num_cols',num_pipe,num_columns)
    ],remainder='drop'
)
    return proccessor

def build_model(proccessor):
    pipe = Pipeline([
    ('preprocessor', proccessor),
    ('model', lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, num_leaves=50,
                                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.5,
                                min_child_samples=20, random_state=42, n_jobs=-1,verbose=-1))
])
    return pipe

def model_evalute(pipe, X_train, X_test, y_train, y_test):
    pipe.fit(X_train, y_train)

    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)

    r2_train   = r2_score(y_train, y_pred_train)
    r2_test    = r2_score(y_test,  y_pred_test)
    mae_train  = mean_squared_error(y_train, y_pred_train)
    mae_test   = mean_squared_error(y_test,  y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
    mape       = mean_absolute_percentage_error(y_test, y_pred_test)

    results = pd.DataFrame([{
        'Model'     : 'LightGBM',
        'R2 Train'  : round(r2_train,  4),
        'R2 Test'   : round(r2_test,   4),
        'MAE Train' : round(mae_train,  2),
        'MAE Test'  : round(mae_test,   2),
        'RMSE Train': round(rmse_train, 2),
        'RMSE Test' : round(rmse_test,  2),
        'MAPE'      : round(mape,       4)
    }])

    return results

def save_data(data_path,X_train, X_test, y_train, y_test):
    logger.info("Saving data Path")
    save_path = os.path.join(data_path, "processed")
    os.makedirs(save_path, exist_ok=True)
    X_train.to_csv(os.path.join(save_path,"X_train.csv"),index=False)
    X_test.to_csv(os.path.join(save_path,"X_test.csv"),index=False)
    y_train.to_csv(os.path.join(save_path,"y_train.csv"),index=False)
    y_test.to_csv(os.path.join(save_path,"y_test.csv"),index=False)
    logger.info("Data Saved Successfully")

def main():
    df = data_load(r'data\interim\df.csv')
    X_train, X_test, y_train, y_test = split_data(df)
    proccessor = build_proccesor(cat_columns,num_columns)
    pipe = build_model(proccessor)
    logger.info("Training Started")
    results = model_evalute(pipe, X_train, X_test, y_train, y_test)  
    print(results)                                                  
    save_data("data", X_train, X_test, y_train, y_test)

if __name__ == "__main__":
     main()


