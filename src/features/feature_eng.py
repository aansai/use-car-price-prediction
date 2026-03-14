import numpy as np
import pandas as pd
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def data_load(url):
    logger.info("Data Loading Start")
    df = pd.read_csv(url)
    logger.info("Data Loading Successfully")
    return df

def converging_brand(df):
    if 'brand' in df.columns:
        df['brand'] = df['brand'].str.lower()
        logger.info("Conversion of Brand Successfully")
    else:
        logger.warning("'brand' column not found in the dataframe")
    return df

def converging_model(df):
    if 'model' in df.columns:
        df['model'] = df['model'].str.strip()
        df['model'] = df['model'].apply(lambda x: ' '.join(dict.fromkeys(x.split())))
        df['model'] = df['model'].str.title()
        df['model'] = df['model'].apply(lambda x: ' '.join(x.split()[:2]))
        logger.info("Conversion of Model Successfully")
    else:
        logger.warning("'model' column not found in the dataframe")
    return df

def converging_milage(df):
    df['milage'] = pd.to_numeric(
        df['milage'].str.replace(',', '', regex=False).str.replace('mi.', '',
        regex=False).str.strip(), errors='coerce')
    logger.info("Conversion of Milage Successfully")
    return df

def converging_fuel(df):
    df['fuel_type'] = df['fuel_type'].replace('–', 'unknown')
    df['fuel_type'] = df['fuel_type'].str.lower()
    df['fuel_type'] = df['fuel_type'].fillna('unknown')
    logger.info("Conversion Fuel Type Successfully")
    return df

def converging_engine(df):
    df['engine'] = df['engine'].replace('–', np.nan)

    horse_power = pd.to_numeric(
        df['engine'].str.extract(r'(\d+\.?\d*)\s*HP', expand=False), errors='coerce')
    df['horse_power'] = horse_power.fillna(horse_power.median())

    displacement_L = pd.to_numeric(
        df['engine'].str.extract(r'(\d+\.?\d*)\s*L(?:iter)?', expand=False),
        errors='coerce')
    df['displacement_L'] = displacement_L.fillna(displacement_L.median())

    cylinder = pd.to_numeric(
        df['engine'].str.extract(r'(\d+)\s*Cylinder', expand=False),
        errors='coerce')
    df['cylinder'] = cylinder.fillna(cylinder.median())

    df.drop(columns=['engine'], inplace=True)
    logger.info("Conversion of Engine Successfully")
    return df

def converging_transmission(df):
    df['transmission'] = df['transmission'].str.strip().str.lower()

    def standardize_transmission(val):
        if pd.isna(val):
            return np.nan
        val = val.lower().strip()
        if any(x in val for x in ['cvt', 'variable', 'cvt-f']):
            return 'CVT'
        elif any(x in val for x in ['manual', 'm/t', 'm/spd', 'mt', '6 speed mt']):
            return 'Manual'
        elif any(x in val for x in ['automatic', 'a/t', 'auto', 'at', 'dct',
                                    'dual-clutch', 'single-speed', 'fixed gear']):
            return 'Automatic'
        else:
            return np.nan

    df['transmission_clean'] = df['transmission'].apply(standardize_transmission)
    df['transmission_clean'] = df['transmission_clean'].fillna('unknown')
    df.drop(columns=['transmission'], inplace=True)
    logger.info('Conversion of Transmission Successfully')
    return df

def converging_accident(df):
    df['accident'] = df['accident'].str.replace(
        'At least 1 accident or damage reported', 'Accident Reported'
    )
    df['accident'] = df['accident'].fillna('None reported')
    logger.info("Conversion of Accident Successfully")
    return df

def drop_cleantxt(df):
    if 'clean_title' in df.columns:
        df.drop(columns=['clean_title'], inplace=True)
    logger.info("Dropping of Clean Text Columns Successfully")
    return df

def converging_price(df):
    df['price'] = pd.to_numeric(
        df['price'].str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip(),
        errors='coerce')
    df = df.dropna(subset=['price'])
    Q1, Q3 = df['price'].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]
    logger.info('Conversion of Price Successfully')
    return df

def save_data(data_path, df):
    logger.info("Saving data Path")
    save_path = os.path.join(data_path, "interim")
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, "df.csv"), index=False)
    logger.info("Save Data Path Successfully")
    return df

def main():
    df = data_load(r'data\raw\df.csv')
    df = converging_brand(df)
    df = converging_model(df)
    df = converging_milage(df)
    df = converging_fuel(df)
    df = converging_engine(df)
    df = converging_transmission(df)
    df = converging_accident(df)
    df = drop_cleantxt(df)
    df = converging_price(df)
    save_data("data", df)


if __name__ == "__main__":
    main()