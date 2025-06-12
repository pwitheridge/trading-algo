import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import time

print("Testing yfinance...")

def test_direct_api():
    print("\nTesting direct API access...")
    url = "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD"
    try:
        response = requests.get(url)
        print(f"API Response Status: {response.status_code}")
        print(f"API Response Content: {response.text[:200]}...")  # Print first 200 chars
    except Exception as e:
        print(f"API request failed: {str(e)}")

def test_yfinance():
    print("\nTesting yfinance...")
    try:
        print("Creating ticker object...")
        btc = yf.Ticker("BTC-USD")
        print("Ticker object created successfully")
        
        # Get last 7 days of data
        end = datetime.now()
        start = end - timedelta(days=7)
        
        print(f"Current time: {end}")
        print(f"Start date: {start}")
        print(f"End date: {end}")
        
        print("\nUsing period='7d' method...")
        data = btc.history(period="7d")
        
        if data.empty:
            print("No data returned with period method")
            print("\nTrying start/end dates method...")
            data = btc.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
            if data.empty:
                print("Still no data with start/end dates method")
            else:
                print(f"\nSuccessfully retrieved {len(data)} rows of data")
                print("\nLast 5 rows:")
                print(data.tail())
        else:
            print(f"\nSuccessfully retrieved {len(data)} rows of data")
            print("\nLast 5 rows:")
            print(data.tail())
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Error type: {type(e)}")

def test_googl():
    print("\nTesting GOOGL...")
    try:
        googl = yf.Ticker("GOOGL")
        end = datetime.now()
        start = end - timedelta(days=7)
        print(f"Fetching GOOGL data from {start.date()} to {end.date()}")
        data = googl.history(start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
        if data.empty:
            print("No data returned for GOOGL")
        else:
            print(f"\nSuccessfully retrieved {len(data)} rows of GOOGL data")
            print("\nLast 5 rows:")
            print(data.tail())
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    test_direct_api()
    time.sleep(2)  # Add a delay to avoid rate limiting
    test_yfinance()
    time.sleep(2)  # Add a delay to avoid rate limiting
    test_googl() 