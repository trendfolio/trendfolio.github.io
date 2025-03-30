import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta

def fetch_and_plot_market_data():
    try:
        # Define tickers to fetch
        tickers = {
            "^GSPC": "SPX",
            "^NDX": "QQQ",
            "^DJI": "DJI",
            "SMH": "SMH",
            "XLE": "XLE",
            "CVX": "Chevron",
            "XOM": "Exxon Mobil",
            "NVDA": "NVIDIA",
            "META": "Meta",
            "AAPL": "Apple",
            "MSFT": "Microsoft",
            "TSLA": "Tesla",
            "AMD": "AMD",
            "AVGO": "Broadcom",
            "AMZN": "Amazon",
            "GOOGL": "Google",
            "XLF": "XLF",
            "XLY": "XLY",
            "XLC": "XLC",
            "XLK": "XLK",
            "XLV": "XLV",
            "XLI": "XLI",
            "XLP": "XLP",
            "XLU": "XLU"
        }
        
        # Get historical data for all tickers
        end_date = datetime.now()
        start_date = datetime(1929, 1, 1)
        
        dfs = {}
        for symbol in tickers.keys():
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                auto_adjust=True,
                interval="1d"
            )
            dfs[symbol] = df.copy()
        
        # Create subplots
        fig = make_subplots(
            rows=6, cols=4,
            subplot_titles=[f"{name} ({symbol})" for symbol, name in tickers.items()]
        )
        
        # Plot each ticker in its own subplot
        row = 1
        col = 1
        for symbol, name in tickers.items():
            fig.add_trace(
                go.Scatter(
                    x=dfs[symbol].index,
                    y=dfs[symbol]['Close'],
                    name=name,
                    mode='lines'
                ),
                row=row,
                col=col
            )
            
            # Update subplot layout
            fig.update_xaxes(title_text="Date", row=row, col=col)
            fig.update_yaxes(title_text="Price", row=row, col=col)
            
            # Move to next subplot position
            col += 1
            if col > 4:  # Changed from 3 to 4 columns
                col = 1
                row += 1
        
        # Update layout
        fig.update_layout(
            height=2400,
            width=2000,
            showlegend=False,
            title_text="Market Data - Dividend Adjusted Prices",
        )
        
        # Display statistics for each ticker
        print("\nData Summary:")
        for symbol, name in tickers.items():
            df = dfs[symbol]
            print(f"\n{name} ({symbol}):")
            print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            print(f"Number of trading days: {len(df)}")
            print(f"Current value: {df['Close'].iloc[-1]:.2f}")
        
        # Create combined dataframe with just the closing prices
        combined_df = pd.DataFrame()
        for symbol, name in tickers.items():
            combined_df[name] = dfs[symbol]['Close']
        
        # Show the plot
        fig.show()
        
        return dfs, combined_df
        
    except Exception as e:
        print(f"Error fetching/plotting data: {e}")
        return None, None

if __name__ == "__main__":
    dfs, combined_df = fetch_and_plot_market_data() 