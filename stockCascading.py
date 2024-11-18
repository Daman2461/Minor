import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

# Function to fetch stock market data with 1m interval for a single day
def fetch_stock_data(ticker, start_date, end_date, interval='5m'):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        if data.empty:
            print(f"No data found for {ticker} during the specified date range.")
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame if there's an error

 
def add_technical_indicators(df):
    # Check if 'Close' column exists
    if 'Close' not in df.columns:
        raise ValueError("Error: 'Close' column not found in DataFrame!")

    # Extract 'Close' prices as a 1D array
    close_prices = df['Close'].values.flatten()  # Ensures it's 1D
    
    # Debugging: Check if close_prices is a 1D array
    if close_prices.ndim != 1:
        raise ValueError("Error: 'close_prices' should be a 1D numpy array after flattening!")

    # Calculate technical indicators
    df['MA5'] = ta.SMA(close_prices, timeperiod=5)
    df['MA20'] = ta.SMA(close_prices, timeperiod=20)
    df['RSI'] = ta.RSI(close_prices, timeperiod=14)
    
    # Bollinger Bands (upper, middle, and lower)
    upper, middle, lower = ta.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_upper'] = upper
    df['BB_lower'] = lower

    # MACD (Moving Average Convergence Divergence)
    macd, macdsignal, macdhist = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macdsignal
    
    return df.dropna()  # Drop NaNs introduced by indicators

# Function to generate the target variable (price up or down after 10 candles)
def generate_target(df, n_periods=10):
    df['Target'] = np.where(df['Close'].shift(-n_periods) > df['Close'], 1, 0)
    return df.dropna()

# Function to prepare features and target for training
from sklearn.preprocessing import StandardScaler

def prepare_data(df):
    features = ['MA5', 'MA20', 'RSI', 'BB_upper', 'BB_lower', 'MACD', 'MACD_signal']
    X = df[features].values
    y = df['Target'].values

    # # Apply normalization to features
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    return X, y

# Function to compute Gini impurity
def gini_impurity(probabilities):
    return 1 - np.sum(probabilities ** 2)

# Cascading function to filter predictions based on Gini impurity
# Cascading function to filter predictions based on Gini impurity
def cascading_predict(models, X, y, max_impurity=0.1):
    unpruned = []  # Store confident predictions
    level_accuracies = []  
    all_predictions = []  

    for i, model in enumerate(models):
        if X.size == 0:  # Check if X is empty before processing
            print(f"No more data to process at level {i + 1}.")
            break

        print(f"Using model {i + 1}...")
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        probs = model.predict_proba(X)
        next_X, next_y = [], []
        model_predictions = []
        
        correct_predictions = 0
        total_predictions = 0
        
        for idx, prob in enumerate(probs):
            gini = gini_impurity(prob)
            if gini <= max_impurity:
                predicted_label = np.argmax(prob)
                correct = predicted_label == y[idx]
                
                if correct:
                    correct_predictions += 1
                total_predictions += 1
                
                unpruned.append((prob, X[idx], y[idx]))
                model_predictions.append((X[idx], y[idx], predicted_label, correct))
            else:
                next_X.append(X[idx])
                next_y.append(y[idx])
        
        level_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        level_accuracies.append(level_accuracy)
        
        X, y = np.array(next_X), np.array(next_y)
        all_predictions.append(model_predictions)
    
    return unpruned, level_accuracies, all_predictions


# Train a few models (e.g., 3 models in the cascade)
def train_cascade(X_train, y_train, num_models=3):
    models = []
    for i in range(num_models):
        print(f"Training model {i+1}...")
        model = MLPClassifier(hidden_layer_sizes=(10,), solver='adam', alpha = 0.01,max_iter=1000)
        model.fit(X_train, y_train)
        models.append(model)
    return models


# Example: Fetch stock data for the ticker "AAPL" using 5-minute data for one day
ticker = "RELIANCE.NS"
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')


# Fetch data and add technical indicators
stock_data = fetch_stock_data(ticker, start_date, end_date, interval='5m')
if stock_data.empty:
    print(f"No data found for {ticker} on {start_date}.")

else:
    stock_data = add_technical_indicators(stock_data)
    stock_data = generate_target(stock_data, n_periods=10)

    # Prepare the data (with scaling)
    X, y = prepare_data(stock_data)

    # Train the cascade of models with adjusted hyperparameters
    models = train_cascade(X, y, num_models=3)

    # Make predictions with cascading and adjusted impurity threshold
    unpruned, level_accuracies, all_predictions = cascading_predict(models, X, y, max_impurity=0.1)

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
        
        correct_points = [point for point in predictions if point[3] == True]
        incorrect_points = [point for point in predictions if point[3] == False]
        
        correct_x = [point[0][0] for point in correct_points]
        correct_y = [point[0][1] for point in correct_points]
        incorrect_x = [point[0][0] for point in incorrect_points]
        incorrect_y = [point[0][1] for point in incorrect_points]
        
        plt.scatter(correct_x, correct_y, color='green', label='Correct', alpha=0.7)
        plt.scatter(incorrect_x, incorrect_y, color='red', label='Incorrect', alpha=0.7)
        plt.title(f"Level {level + 1} Predictions")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.show()
