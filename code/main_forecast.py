# main_forecast.py

import pandas as pd
import os
from prophet import Prophet
from forecast_utils import prepare_data, plot_forecast

def forecast_stock(df, stock_code, forecast_weeks=15):
    # Filter data for the given stock code
    filtered_df = df[df['StockCode'] == stock_code][['InvoiceDate', 'Quantity']]
    
    print(f"Data for stock code {stock_code}:")
    print(filtered_df.head())
    print(f"Shape of filtered data: {filtered_df.shape}")

    if filtered_df.empty:
        raise ValueError(f"No data found for stock code {stock_code}")

    # Prepare data for Prophet
    prophet_df = prepare_data(filtered_df)

    if len(prophet_df) < 2:
        raise ValueError(f"Insufficient data for stock code {stock_code} after date parsing")

    # Create and fit the Prophet model
    model = Prophet()
    model.fit(prophet_df)

    # Create future dates for forecasting
    future_dates = model.make_future_dataframe(periods=forecast_weeks * 7)  # 7 days per week
    
    # Generate forecast
    forecast = model.predict(future_dates)

    # Plot the forecast
    plot_forecast(model, forecast, stock_code, forecast_weeks)

    # Return the forecast dataframe
    return forecast

if __name__ == "__main__":

        df = pd.read_csv(r"E:\Projects\salesForecast\Data\cleaned_data.csv")
        
        print(f"Loaded data shape: {df.shape}")
        print("Column names:")
        print(df.columns)
        print("\nSample data:")
        print(df.head())

        # Get user input for stock code
        stock_code = input("Enter the stock code to forecast: ")
        
        # Generate and display forecast
        forecast = forecast_stock(df, stock_code)
        print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
