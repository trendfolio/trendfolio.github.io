import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_and_plot_spx():
    try:
        # Create a ticker object for S&P 500
        spx = yf.Ticker("^GSPC")
        
        # Get historical data with auto-adjust enabled
        end_date = datetime.now()
        start_date = datetime(1929, 1, 1)
        
        df = spx.history(
            start=start_date,
            end=end_date,
            auto_adjust=True,
            interval="1d"
        )
        
        # Filter out non-trading days (where closing price is the same as previous day)
        df['Price_Change'] = df['Close'].diff()
        df_trading = df[df['Price_Change'] != 0].copy()
        
        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.plot(df_trading.index, df_trading['Close'], label='SPX', color='blue')
        
        # Customize the plot
        plt.title('S&P 500 (^GSPC) Historical Data (Trading Days Only)')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Display some statistics
        print(f"\nData Summary:")
        print(f"Date range: {df_trading.index[0].strftime('%Y-%m-%d')} to {df_trading.index[-1].strftime('%Y-%m-%d')}")
        print(f"Number of trading days: {len(df_trading)}")
        print(f"Number of non-trading days filtered: {len(df) - len(df_trading)}")
        print(f"Days removed due to no price change: {len(df[df['Price_Change'] == 0])}")
        print(f"Current SPX value: {df_trading['Close'].iloc[-1]:.2f}")
        
        # Show the plot
        plt.show()
        
        return df_trading
        
    except Exception as e:
        print(f"Error fetching/plotting data: {e}")
        return None

if __name__ == "__main__":
    df = fetch_and_plot_spx() 