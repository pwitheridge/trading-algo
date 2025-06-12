import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_bitcoin_data():
    print("Testing yfinance Bitcoin data retrieval...")
    
    # Test 1: Basic ticker creation
    print("\nTest 1: Creating BTC-USD ticker")
    try:
        btc = yf.Ticker("BTC-USD")
        print("✓ Successfully created BTC-USD ticker")
    except Exception as e:
        print(f"✗ Failed to create ticker: {str(e)}")
        return

    # Test 2: Get basic info
    print("\nTest 2: Getting basic info")
    try:
        info = btc.info
        print("✓ Successfully retrieved basic info")
        print(f"Current price: ${info.get('currentPrice', 'N/A')}")
    except Exception as e:
        print(f"✗ Failed to get basic info: {str(e)}")

    # Test 3: Get historical data
    print("\nTest 3: Getting historical data")
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        hist = btc.history(start=start_date, end=end_date)
        print(f"✓ Successfully retrieved {len(hist)} days of historical data")
        print("\nLast 5 days of data:")
        print(hist.tail())
    except Exception as e:
        print(f"✗ Failed to get historical data: {str(e)}")

if __name__ == "__main__":
    test_bitcoin_data() 