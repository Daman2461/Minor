import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Generate synthetic data
X, y = make_classification(
    n_samples=400,        # Number of samples
    n_features=5,         # Number of features
    n_classes=2,          # Number of classes
    class_sep=0.5,        # Reduce class separation (make classes closer)
    flip_y=0.1,           # Introduce some noise by flipping 10% of the labels
    n_informative=3,      # Number of informative features
    n_redundant=0,        # No redundant features
    n_clusters_per_class=1 # One cluster per class
)

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
        model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000 )
        model.fit(X_train, y_train)
        models.append(model)
    return models

# Example: Training a cascade of models
models = train_cascade(X, y)

# Make predictions with cascading process
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
    plt.scatter(correct_x, correct_y, color='green', label='Correct', alpha=0.6)
    plt.scatter(incorrect_x, incorrect_y, color='red', label='Incorrect', alpha=0.6)
    
    # Title, labels, and legend
    plt.title(f"Level {level + 1} Predictions")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()
