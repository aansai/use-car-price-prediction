import numpy as np
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def data_load(url):
    logging.info("Data Loading Start")
    df = pd.read_csv(url)
    logger.info("Data Loading Successfully")
    return df

def save_data(data_path,df):
    logger.info("Saving data Path")
    save_path = os.path.join(data_path,"raw")
    os.makedirs(save_path,exist_ok=True)
    df.to_csv(os.path.join(save_path,"df.csv"),index=False)
    logger.info("Save Data Path Successfully")
    return df

def main():
    df = data_load('used_cars.csv')
    save_data("data",df)

if __name__ == "__main__":
    main()

