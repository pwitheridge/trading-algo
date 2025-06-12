import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Try different variations of the ticker symbol
possible_tickers = [
    "SYHORNB.L",
    "SYHORNB.LN",
    "SYHORNB",
    "HORNB.L",
    "HORNB.LN"
]

print("Trying to fetch Hornbeam Unit Trust data...")

for ticker in possible_tickers:
    try:
        print(f"\nTrying ticker: {ticker}")
        hornbeam = yf.Ticker(ticker)
        hornbeam_hist = hornbeam.history(period="max")
        
        if not hornbeam_hist.empty:
            print(f"Successfully found data for {ticker}")
            
            # Create the plot
            plt.figure(figsize=(15, 8))
            plt.plot(hornbeam_hist.index, hornbeam_hist['Close'], label='Hornbeam Unit Trust Price')
            plt.title('Hornbeam Unit Trust Price History')
            plt.xlabel('Date')
            plt.ylabel('Price (GBP)')
            plt.grid(True)
            plt.legend()
            plt.show()

            # Print statistics
            print("\nHornbeam Unit Trust Statistics:")
            print(f"Starting price: £{hornbeam_hist['Close'].iloc[0]:.2f}")
            print(f"Current price: £{hornbeam_hist['Close'].iloc[-1]:.2f}")
            print(f"Highest price: £{hornbeam_hist['Close'].max():.2f}")
            print(f"Lowest price: £{hornbeam_hist['Close'].min():.2f}")
            print(f"Average price: £{hornbeam_hist['Close'].mean():.2f}")

            # Print the most recent prices
            print("\nMost Recent Prices:")
            print(hornbeam_hist['Close'].tail())
            break
        else:
            print(f"No data found for {ticker}")
    except Exception as e:
        print(f"Error with {ticker}: {str(e)}")

print("\nNote: If no data was found, you might want to:")
print("1. Verify the ticker symbol")
print("2. Check if the fund is listed on a different exchange")
print("3. Contact the fund manager for historical data") 