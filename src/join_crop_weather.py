import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA = BASE_DIR / 'data'
RAW = DATA / 'raw'
JOIN_KEYS = ['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Crop']

def clean_text_columns(df):
    df = df.copy()
    for col in ['State_Name', 'District_Name', 'Season', 'Crop']:
        if col in df.columns:
            df[col] = df[col].astype('string').str.strip().str.replace(r'\s+', ' ', regex=True)
    return df

crop = pd.read_csv(RAW / 'Indian_crop_production_yield_dataset.csv', low_memory=False)
weather = pd.read_csv(DATA / 'weather_data_final.csv', low_memory=False)
crop = clean_text_columns(crop)
weather = clean_text_columns(weather)
weather = weather.drop_duplicates(subset=JOIN_KEYS)
joined = crop.merge(weather, on=JOIN_KEYS, how='left', validate='many_to_one')
joined.to_csv(DATA / 'crop_weather_joined.csv', index=False)
print('Crop:', crop.shape)
print('Weather:', weather.shape)
print('Joined:', joined.shape)
print('Missing rain_sum_mm ratio:', joined['rain_sum_mm'].isna().mean())
