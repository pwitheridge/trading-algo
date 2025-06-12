import yfinance as yf

# Try different possible ticker symbols for Hornbeam Unit Trust
possible_tickers = [
    "HORNBEAM.L",  # Common format for UK unit trusts
    "HORNBEAM.UK",
    "HORNBEAM",
    "HORN.L",
    "HORN.UK",
    "HBM.L",  # Possible abbreviation
    "HBM.UK"
]

print("Checking for Hornbeam Unit Trust on Yahoo Finance...")
print("\nTrying different possible ticker symbols:")

for ticker in possible_tickers:
    try:
        fund = yf.Ticker(ticker)
        info = fund.info
        if info and 'regularMarketPrice' in info:
            print(f"\nFound data for {ticker}:")
            print(f"Name: {info.get('longName', 'N/A')}")
            print(f"Current Price: {info.get('regularMarketPrice', 'N/A')}")
            print(f"Currency: {info.get('currency', 'N/A')}")
        else:
            print(f"\nNo data found for {ticker}")
    except Exception as e:
        print(f"\nError checking {ticker}: {str(e)}")

print("\nNote: If no data was found, Hornbeam Unit Trust might not be listed on Yahoo Finance.")
print("You might want to check other financial data providers or the fund's official website.") 