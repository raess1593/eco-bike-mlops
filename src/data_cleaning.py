import numpy as np
import pandas as pd
from pathlib import Path

def clean_data():
    print("Cleaning data...")
    root_path = Path(__file__).parent.parent
    data_path = root_path / 'data' / 'raw_data.csv'
    cleaned_data_path = root_path / 'data' / 'cleaned_data.csv'

    df = pd.read_csv(data_path)

    for c in ['temp', 'humidity']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

        if c == 'humidity':
            invalid_condition = (df[c] < 0) | (df[c] > 100)
            df[c] = np.where(invalid_condition, np.nan, df[c])

        median = df[c].median(skipna=True)
        df[c] = df[c].fillna(median)

    df['holiday'] = pd.to_numeric(df['holiday'], errors='coerce')
    valid_condition = df['holiday'].isin([0, 1])
    df['holiday'] = np.where(valid_condition, df['holiday'], np.nan)
    df = df.dropna(subset=['holiday'])

    df.to_csv(cleaned_data_path)
    print(f"Cleaned data has been developed successfully -- {data_path}")

if __name__ == "__main__":
    clean_data()