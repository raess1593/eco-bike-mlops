import numpy as np
import pandas as pd
from pathlib import Path

def generate_raw_data():
    root_path = Path(__file__).parent.parent
    data_path = root_path / 'data' / 'raw_data.csv'

    n = 500
    data = {
        'temp': np.random.randint(-10, 40, n).astype(object),
        'humidity': np.random.randint(0, 100, n),
        'holiday': np.random.choice([0, 1], n)
    }

    df = pd.DataFrame(data)

    df['demand'] = 50 + 2 * np.abs(df['temp'] - 20) 
    df['demand'] += (100 - df['humidity']) * 0.5 
    df['demand'] += df['holiday'] * 30 
    df['demand'] += np.random.normal(0, 5, n)
    df['demand'] = df['demand'].clip(lower=0).astype(int)

    df['temp'] = df['temp'].astype('object')

    r = np.random.choice([0, 1])
    print(r)
    if r == 0:
        for _ in range(20):
            x = np.random.randint(n-1)
            df.loc[x, 'temp'] = '?'
            df.loc[x+1, 'humidity'] = -8

    df.to_csv(data_path, index=False)

if __name__ == "__main__":
    generate_raw_data()