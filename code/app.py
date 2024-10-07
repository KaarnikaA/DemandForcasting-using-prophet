# app.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
import plotly
import json

app = Flask(__name__)

def prepare_data(df):
    def parse_date(date_str):
        try:
            return pd.to_datetime(date_str, format='%d-%b-%y')
        except ValueError:
            try:
                return pd.to_datetime(date_str, format='%d-%m-%Y')
            except ValueError:
                return pd.to_datetime(date_str, errors='coerce')

    df['InvoiceDate'] = df['InvoiceDate'].apply(parse_date)
    df = df.rename(columns={'InvoiceDate': 'ds', 'Quantity': 'y'})
    df = df.dropna(subset=['ds'])  # Remove rows with NaT dates
    return df.sort_values('ds').reset_index(drop=True)

def forecast_stock(df, stock_code, forecast_weeks=15):
    filtered_df = df[df['StockCode'] == stock_code][['InvoiceDate', 'Quantity']]
    
    if filtered_df.empty:
        raise ValueError(f"No data found for stock code {stock_code}")

    prophet_df = prepare_data(filtered_df)

    if len(prophet_df) < 2:
        raise ValueError(f"Insufficient data for stock code {stock_code} after date parsing")

    model = Prophet()
    model.fit(prophet_df)

    future_dates = model.make_future_dataframe(periods=forecast_weeks * 7)
    forecast = model.predict(future_dates)

    return forecast, prophet_df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def get_forecast():
    stock_code = request.form['stock_code']
    df = pd.read_csv(r"E:\Projects\salesForecast\Data\cleaned_data.csv")  # Replace with your actual file name
    
    try:
        forecast, historical_data = forecast_stock(df, stock_code)
        
        # Create historical data trace
        historical_trace = go.Scatter(
            x=historical_data['ds'],
            y=historical_data['y'],
            mode='markers',
            name='Historical Data'
        )

        # Create forecast trace
        forecast_trace = go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name='Forecast'
        )

        # Create upper and lower bound traces
        upper_bound = go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            mode='lines',
            line=dict(width=0),
            showlegend=False
        )
        lower_bound = go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            mode='lines',
            line=dict(width=0),
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False
        )

        data = [historical_trace, forecast_trace, upper_bound, lower_bound]
        layout = go.Layout(title=f'Forecast for Stock Code {stock_code}',
                           xaxis=dict(title='Date'),
                           yaxis=dict(title='Quantity'))
        fig = go.Figure(data=data, layout=layout)

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return jsonify({'graph': graphJSON})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)