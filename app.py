import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time

# Set page config
st.set_page_config(
    page_title="Bitcoin Price Prediction",
    page_icon="₿",
    layout="wide"
)

# Title and description
st.title("Bitcoin Price Prediction Dashboard")
st.markdown("""
This dashboard provides Bitcoin price analysis and predictions using machine learning models.
Select your preferred date range and model to get started.
""")

# Sidebar for controls
st.sidebar.header("Controls")

# Date range selector
default_start = datetime.now() - timedelta(days=365)
default_end = datetime.now()

start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start,
    max_value=default_end
)

end_date = st.sidebar.date_input(
    "End Date",
    value=default_end,
    max_value=default_end
)

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "Linear Regression", "XGBoost"]
)

# Prediction horizon
prediction_days = st.sidebar.slider(
    "Prediction Horizon (Days)",
    min_value=1,
    max_value=30,
    value=7
)

# Fetch Bitcoin data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_data(start_date, end_date):
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            st.write(f"Attempt {attempt + 1} to fetch Bitcoin data...")
            btc = yf.Ticker("BTC-USD")
            st.write("Ticker object created, fetching history...")
            
            # Convert dates to string format that yfinance expects
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            st.write(f"Fetching data from {start_str} to {end_str}")
            
            data = btc.history(start=start_str, end=end_str)
            
            if data.empty:
                st.warning("No data available for the selected date range. Please try a different range.")
                return None
                
            # Ensure required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in data.columns for col in required_columns):
                st.error(f"Missing required data columns. Available columns: {data.columns.tolist()}")
                return None
                
            st.write(f"Successfully fetched {len(data)} rows of data")
            return data
            
        except Exception as e:
            st.write(f"Error details: {str(e)}")
            if attempt < max_retries - 1:
                st.warning(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            else:
                st.error(f"Failed to fetch data after {max_retries} attempts. Error: {str(e)}")
                st.info("Please check your internet connection and try again.")
                return None

# Create features
def create_features(data):
    if data is None or data.empty:
        return None
    try:
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Calculate returns
        df['Returns'] = df['Close'].pct_change()
        
        # Calculate moving averages
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # Calculate volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # Drop NaN values
        df = df.dropna()
        
        # Verify we have enough data after processing
        if len(df) < 20:
            st.warning("Not enough data points after processing. Please select a longer date range.")
            return None
            
        return df
        
    except Exception as e:
        st.error(f"Error creating features: {str(e)}")
        return None

# Main content
try:
    # Fetch and process data
    with st.spinner("Fetching Bitcoin data..."):
        data = fetch_data(start_date, end_date)
        if data is None:
            st.stop()
    
    with st.spinner("Processing data..."):
        data = create_features(data)
        if data is None:
            st.stop()

    # Display current price
    current_price = data['Close'].iloc[-1]
    st.metric("Current Bitcoin Price", f"${current_price:,.2f}")

    # Create price chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA5'],
        name='5-day MA',
        line=dict(color='orange')
    ))
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA20'],
        name='20-day MA',
        line=dict(color='blue')
    ))

    fig.update_layout(
        title='Bitcoin Price History',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # Technical Analysis Section
    st.header("Technical Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price Statistics")
        st.write(f"Highest Price: ${data['High'].max():,.2f}")
        st.write(f"Lowest Price: ${data['Low'].min():,.2f}")
        st.write(f"Average Price: ${data['Close'].mean():,.2f}")
        st.write(f"Price Volatility: {data['Volatility'].iloc[-1]:.4f}")

    with col2:
        st.subheader("Trading Volume")
        volume_fig = go.Figure()
        volume_fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            name='Volume'
        ))
        volume_fig.update_layout(
            title='Trading Volume',
            yaxis_title='Volume',
            xaxis_title='Date',
            height=300
        )
        st.plotly_chart(volume_fig, use_container_width=True)

    # Model Training and Prediction
    if st.sidebar.button("Generate Predictions"):
        with st.spinner("Training model and generating predictions..."):
            try:
                # Prepare features for prediction
                features = ['Returns', 'MA5', 'MA20', 'Volatility']
                X = data[features].values
                y = data['Close'].values

                # Ensure we have enough data for training
                if len(X) < 40:
                    st.error("Not enough data points for training. Please select a longer date range.")
                    st.stop()

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                # Display metrics
                st.header("Model Performance")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RMSE", f"${rmse:,.2f}")
                with col2:
                    st.metric("R² Score", f"{r2:.4f}")

                # Feature importance
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                st.bar_chart(importance_df.set_index('Feature'))

            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.info("Please try adjusting the date range or model parameters.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please try adjusting the date range or refreshing the page.") 