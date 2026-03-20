import pandas as pd
import numpy as np

def preprocessing(filepath):
    data = pd.read_csv(filepath, sep=';')

    # Fix decimal format
    cols = ['CO(GT)', 'C6H6(GT)', 'T', 'RH', 'AH']
    for col in cols:
        data[col] = data[col].astype(str).str.replace(',', '.')
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Time features
    data['Time'] = data['Time'].str.replace('.', ':')
    data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S', errors='coerce')
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

    data['Hour'] = data['Time'].dt.hour
    data['Month'] = data['Date'].dt.month
    data['DayOfWeek'] = data['Date'].dt.dayofweek

    # Remove NMHC
    if 'NMHC(GT)' in data.columns:
        data = data.drop(columns=['NMHC(GT)'])

    # Replace -200 with hourly mean
    for col in data.columns:
        if col not in ['Date', 'Time']:
            data[col] = data.groupby('Hour')[col].transform(
                lambda x: x.replace(-200, np.nan).fillna(x.mean())
            )

    data = data.dropna()

    return data
