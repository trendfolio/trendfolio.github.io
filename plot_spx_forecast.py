import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def plot_spx_forecast():
    try:
        # Read the JSON file
        with open('forecast_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract SPX data
        spx_data = data['instruments']['SPX']
        dates = pd.to_datetime(spx_data['dates'])
        
        # Create figure and axis
        plt.figure(figsize=(12, 6))
        
        # Check what data is available and plot accordingly
        if 'values' in spx_data:
            values = spx_data['values']
            plt.plot(dates, values, label='Historical', color='blue')
        
        if 'forecast' in spx_data:
            forecast = spx_data['forecast']
            forecast_dates = dates[-len(forecast):]  # Assuming forecast aligns with last dates
            plt.plot(forecast_dates, forecast, label='Forecast', color='red', linestyle='--')
        
        # Customize the plot
        plt.title('SPX Historical Data and Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('spx_forecast.png')
        print("Plot saved as spx_forecast.png")
        
        # Display some basic statistics
        print("\nData Summary:")
        print(f"Date range: {dates.min()} to {dates.max()}")
        if 'values' in spx_data:
            print(f"Number of historical points: {len(spx_data['values'])}")
        if 'forecast' in spx_data:
            print(f"Number of forecast points: {len(spx_data['forecast'])}")
        
    except FileNotFoundError:
        print("Error: forecast_data.json not found")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    plot_spx_forecast() 