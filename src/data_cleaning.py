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
        df[c] = df[c].fillna(median).astype('Int64') 

    df['demand'] = pd.to_numeric(df['demand'], errors='coerce')
    valid_condition = df['demand'].isin([0, 1])
    df['demand'] = np.where(valid_condition, df['demand'], np.nan)
    df = df.dropna(subset=['demand'])

    df.to_csv(cleaned_data_path, index=False)

if __name__ == "__main__":
    clean_data()