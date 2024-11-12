import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

# Function to fetch stock market data with 1m interval for a single day
def fetch_stock_data(ticker, start_date, end_date, interval='1m'):
    # Fetch data from Yahoo Finance with the specified date range and interval
    data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return data

# Function to calculate technical indicators (e.g., moving averages, RSI)
def add_technical_indicators(df):
    # Debugging: Check the shape and type of 'Close'
    print(f"Shape of df['Close']: {df['Close'].shape}")
    print(f"First few rows of df['Close']: {df['Close'].head()}")

    # Convert 'Close' to a 1D numpy array (ensure it's 1D)
    close_prices = df['Close'].iloc[:, 0].values  # This ensures it's a 1D numpy array
    
    # Check if the close_prices is a 1D array (it should be)
    if close_prices.ndim != 1:
        raise ValueError("close_prices should be a 1D numpy array!")

    # Apply the technical indicators
    df['MA5'] = pd.Series(ta.SMA(close_prices, timeperiod=5), index=df.index)
    df['MA20'] = pd.Series(ta.SMA(close_prices, timeperiod=20), index=df.index)

    # RSI (Relative Strength Index)
    df['RSI'] = pd.Series(ta.RSI(close_prices, timeperiod=14), index=df.index)

    # Bollinger Bands (upper, middle, and lower)
    upper, middle, lower = ta.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_upper'] = pd.Series(upper, index=df.index)
    df['BB_lower'] = pd.Series(lower, index=df.index)

    # MACD (Moving Average Convergence Divergence)
    macd, macdsignal, macdhist = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = pd.Series(macd, index=df.index)
    df['MACD_signal'] = pd.Series(macdsignal, index=df.index)
    
    return df


# Function to generate the target variable (price up or down after 10 candles)
def generate_target(df, n_periods=10):
    df['Target'] = np.where(df['Close'].shift(-n_periods) > df['Close'], 1, 0)  # 1 = price went up, 0 = price went down
    return df.dropna()  # Drop missing values

# Function to prepare features and target for training
def prepare_data(df):
    features = ['MA5', 'MA20', 'RSI', 'BB_upper', 'BB_lower', 'MACD', 'MACD_signal']
    X = df[features].values
    y = df['Target'].values
    return X, y

# Function to compute Gini impurity
def gini_impurity(probabilities):
    return 1 - np.sum(probabilities ** 2)

# Cascading function to filter predictions based on Gini impurity
def cascading_predict(models, X, y, max_impurity=0.1):
    unpruned = []  # Store predictions for the unpruned data
    level_accuracies = []  # Store level-wise accuracy
    all_predictions = []  # Track predictions at each level for plotting

    for i, model in enumerate(models):
        print(f"Using model {i+1}...")
        probs = model.predict_proba(X)
        
        correct_predictions = 0
        total_predictions = 0
        next_X = []
        next_y = []
        
        model_predictions = []  # Store predictions for this level
        
        # For each data point, calculate Gini impurity and decide whether to prune
        for idx, prob in enumerate(probs):
            gini = gini_impurity(prob)
            
            if gini <= max_impurity:
                # If confident, make prediction and add to unpruned data
                predicted_label = np.argmax(prob)
                correct = predicted_label == y[idx]
                if correct:
                    correct_predictions += 1
                total_predictions += 1
                unpruned.append((prob, X[idx], y[idx]))
                
                # Track prediction at this level for plotting
                model_predictions.append((X[idx], y[idx], predicted_label, correct))
            else:
                # If uncertain, prune and pass to the next model
                next_X.append(X[idx])
                next_y.append(y[idx])
        
        # Calculate and store the accuracy for this model (level)
        if total_predictions > 0:
            level_accuracy = correct_predictions / total_predictions
        else:
            level_accuracy = 0
        level_accuracies.append(level_accuracy)
        
        # Update for the next model
        X = np.array(next_X)
        y = np.array(next_y)
        
        # Store model predictions for this level
        all_predictions.append(model_predictions)
    
    return unpruned, level_accuracies, all_predictions

# Train a few models (e.g., 3 models in the cascade)
def train_cascade(X_train, y_train, num_models=3):
    models = []
    for i in range(num_models):
        print(f"Training model {i+1}...")
        model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000)
        model.fit(X_train, y_train)
        models.append(model)
    return models

# Example: Fetch stock data for the ticker "AAPL" (Apple Inc.) using 1-minute data for one day
ticker = 'AAPL'

# Set the date for fetching one day's worth of data (e.g., today)
# Note: Yahoo Finance only allows 1-minute data for the past 30 days
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')  # Fetch data from the previous day

# Fetch data and add technical indicators
stock_data = fetch_stock_data(ticker, start_date, end_date, interval='1m')
if stock_data.empty:
    print(f"Error: No data found for {ticker} on {start_date}. Please check the date and try again.")
else:
    stock_data = add_technical_indicators(stock_data)
    stock_data = generate_target(stock_data, n_periods=10)

    # Prepare the features and target for training
    X, y = prepare_data(stock_data)

    # Train a cascade of models
    models = train_cascade(X, y)

    # Make predictions with the cascading process
    unpruned, level_accuracies, all_predictions = cascading_predict(models, X, y)

    # Calculate accuracy on unpruned data points
    if unpruned:
        predictions, features, labels = zip(*unpruned)
        predicted_labels = [np.argmax(p) for p in predictions]
        accuracy = accuracy_score(labels, predicted_labels)
        print(f"Overall Accuracy on unpruned data: {accuracy:.4f}")

    # Display Level-wise Accuracy
    print("\nLevel-wise Accuracy:")
    for i, level_accuracy in enumerate(level_accuracies):
        print(f"Level {i+1} Accuracy: {level_accuracy:.4f}")

    # Plotting the scatter plot for each level
    for level, predictions in enumerate(all_predictions):
        plt.figure(figsize=(8, 6))
        
        # Correct predictions (green) and incorrect predictions (red)
        correct_points = [point for point in predictions if point[3] == True]
        incorrect_points = [point for point in predictions if point[3] == False]
        
        # Extract coordinates for correct and incorrect points
        correct_x = [point[0][0] for point in correct_points]
        correct_y = [point[0][1] for point in correct_points]
        incorrect_x = [point[0][0] for point in incorrect_points]
        incorrect_y = [point[0][1] for point in incorrect_points]
        
        # Scatter plot of correct and incorrect predictions
        plt.scatter(correct_x, correct_y, color='green', label='Correct', alpha=0.7)
        plt.scatter(incorrect_x, incorrect_y, color='red', label='Incorrect', alpha=0.7)
        plt.title(f"Level {level + 1} Predictions")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()
