import pytest
from yahooquery import Ticker
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

def test_ticker_data_retrieval():
    # Print current working directory and Python path for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Data directory exists: {os.path.exists('Data')}")
    print(f"Python path: {sys.path}")
    
    # Test with XLE ticker
    ticker = Ticker('XLE')
    
    # Test historical data retrieval
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    history = ticker.history(start=start_date, adj_ohlc=True)
    
    # Reset index to handle multi-level index
    if isinstance(history.index, pd.MultiIndex):
        history = history.reset_index()
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join('Data', f'XLE_data_{timestamp}.csv')
    
    print(f"\nDataFrame info:")
    print(history.info())
    print(f"\nFirst few rows of data:")
    print(history.head())
    
    # Save to CSV
    try:
        history.to_csv(output_file, index=False)
        print(f"\nSuccessfully saved to {output_file}")
        print(f"File exists: {os.path.exists(output_file)}")
        print(f"File size: {os.path.getsize(output_file)} bytes")
    except Exception as e:
        print(f"\nError saving file: {str(e)}")
        raise
    
    # Verify file was created and has content
    assert os.path.exists(output_file), f"CSV file was not created at {output_file}"
    assert os.path.getsize(output_file) > 0, f"CSV file is empty at {output_file}"
    
    # Try to read the file back to verify it's valid
    try:
        df_check = pd.read_csv(output_file)
        print(f"\nSuccessfully verified CSV file. Shape: {df_check.shape}")
    except Exception as e:
        print(f"\nError reading back CSV file: {str(e)}")
        raise

def test_invalid_ticker():
    # Test with an invalid ticker
    invalid_ticker = Ticker('INVALID_TICKER_123456')
    history = invalid_ticker.history(start='2023-01-01')
    assert history.empty 