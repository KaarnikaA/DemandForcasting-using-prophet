# forecast_utils.py

import pandas as pd
import matplotlib.pyplot as plt

def prepare_data(df):
    def parse_date(date_str):
        try:
            return pd.to_datetime(date_str, format='%d-%b-%y')
        except ValueError:
            try:
                return pd.to_datetime(date_str, format='%d-%m-%Y')
            except ValueError:
                return pd.to_datetime(date_str, errors='coerce')

    print(f"Original data shape: {df.shape}")
    print(f"Sample of original data:\n{df.head()}")

    df['InvoiceDate'] = df['InvoiceDate'].apply(parse_date)
    df = df.rename(columns={'InvoiceDate': 'ds', 'Quantity': 'y'})
    df = df.dropna(subset=['ds'])  # Remove rows with NaT dates

    print(f"Data shape after date parsing: {df.shape}")
    print(f"Sample of processed data:\n{df.head()}")

    return df.sort_values('ds').reset_index(drop=True)

def plot_forecast(model, forecast, stock_code, forecast_weeks):
    fig, ax = plt.subplots(figsize=(12, 6))
    model.plot(forecast, ax=ax)
    ax.set_title(f'Forecast for Stock Code {stock_code} - Next {forecast_weeks} Weeks')
    plt.show()