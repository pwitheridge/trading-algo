import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set the start date
start_date = '2023-12-01'  # Starting from December 2023
end_date = datetime.now().strftime('%Y-%m-%d')

# Fetch Bitcoin data
btc = yf.Ticker("BTC-USD")
# Get historical data for the specified period
btc_hist = btc.history(start=start_date, end=end_date)

# Create features
btc_hist['Returns'] = btc_hist['Close'].pct_change()
btc_hist['MA5'] = btc_hist['Close'].rolling(window=5).mean()
btc_hist['MA20'] = btc_hist['Close'].rolling(window=20).mean()
btc_hist['Volatility'] = btc_hist['Returns'].rolling(window=20).std()

# Drop NaN values
btc_hist = btc_hist.dropna()

# Prepare features for the model
features = ['Returns', 'MA5', 'MA20', 'Volatility']
X = btc_hist[features].values
y = btc_hist['Close'].values

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on both training and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Generate future predictions
last_data = btc_hist[features].iloc[-1:].values
future_prices = []
current_data = last_data.copy()

for _ in range(30):  # Predict next 30 days
    next_price = model.predict(current_data)[0]
    future_prices.append(next_price)
    
    # Update features for next prediction
    current_data[0][0] = (next_price - btc_hist['Close'].iloc[-1]) / btc_hist['Close'].iloc[-1]  # Returns
    current_data[0][1] = (btc_hist['MA5'].iloc[-1] * 4 + next_price) / 5  # MA5
    current_data[0][2] = (btc_hist['MA20'].iloc[-1] * 19 + next_price) / 20  # MA20
    current_data[0][3] = btc_hist['Volatility'].iloc[-1]  # Keep same volatility

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(btc_hist.index[:len(X_train)], y_train, 'b-', label='Training Data')
plt.plot(btc_hist.index[len(X_train):], y_test, 'g-', label='Testing Data')
plt.plot(btc_hist.index[:len(X_train)], y_train_pred, 'b--', label='Training Predictions')
plt.plot(btc_hist.index[len(X_train):], y_test_pred, 'g--', label='Testing Predictions')
plt.plot(pd.date_range(start=btc_hist.index[-1], periods=31)[1:], future_prices, 
         'r--', label='Future Predictions')
plt.title(f'Bitcoin Price: Training, Testing, and Future Predictions\nFrom {start_date} to {end_date}')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.legend()
plt.show()

# Print model performance metrics
print("\nModel Performance Metrics:")
print(f"Training R-squared score: {r2_score(y_train, y_train_pred):.4f}")
print(f"Testing R-squared score: {r2_score(y_test, y_test_pred):.4f}")
print(f"Training RMSE: ${np.sqrt(mean_squared_error(y_train, y_train_pred)):.2f}")
print(f"Testing RMSE: ${np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
print(f"\nPredicted price for tomorrow: ${future_prices[0]:.2f}")

# Print feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values('Importance', ascending=False))


