import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Fetch FTSE 100 data
ftse = yf.Ticker("^FTSE")  # Yahoo Finance symbol for FTSE 100
start_date = '2008-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

# Get historical data
ftse_hist = ftse.history(start=start_date, end=end_date)

# Create the plot
plt.figure(figsize=(15, 8))
plt.plot(ftse_hist.index, ftse_hist['Close'], label='FTSE 100 Price')
plt.title('FTSE 100 Price History (2008-Present)')
plt.xlabel('Date')
plt.ylabel('Price (GBP)')
plt.grid(True)
plt.legend()

# Add some key events annotations
plt.annotate('Financial Crisis', xy=('2008-09-15', ftse_hist['Close']['2008-09-15']),
            xytext=(10, 10), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='red'))
plt.annotate('COVID-19', xy=('2020-03-23', ftse_hist['Close']['2020-03-23']),
            xytext=(10, 10), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='red'))

plt.show()

# Print some statistics
print("\nFTSE 100 Statistics:")
print(f"Starting price (2008): £{ftse_hist['Close'].iloc[0]:.2f}")
print(f"Current price: £{ftse_hist['Close'].iloc[-1]:.2f}")
print(f"Highest price: £{ftse_hist['Close'].max():.2f}")
print(f"Lowest price: £{ftse_hist['Close'].min():.2f}")
print(f"Average price: £{ftse_hist['Close'].mean():.2f}") 