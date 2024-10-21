# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load Breast Cancer Dataset
def load_data():
    """Load the breast cancer dataset and return features and target."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y

# Preprocess Data
def preprocess_data(X):
    """Handle missing values and scale features."""
    # Impute missing values (if any)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Scale features for methods that assume normally distributed data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled

#  Feature Selection
def feature_selection(X, y):
    """Select important features using different methods."""
    # Filter Method using SelectKBest with Chi-Square
    filter_selector = SelectKBest(chi2, k=10)
    X_positive = np.maximum(X, 0)  # Ensure no negative values
    X_filtered = filter_selector.fit_transform(X_positive, y)
    selected_features_filter = filter_selector.get_support(indices=True)

    # Wrapper Method using Recursive Feature Elimination (RFE)
    log_reg = LogisticRegression(max_iter=10000)
    rfe_selector = RFE(estimator=log_reg, n_features_to_select=10, step=1)
    X_wrapper = rfe_selector.fit_transform(X, y)
    selected_features_wrapper = rfe_selector.get_support(indices=True)

    # Embedded Method using Random Forest Feature Importance
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    importances = rf_model.feature_importances_
    selected_features_embedded = np.argsort(importances)[-10:]

    return (selected_features_filter, selected_features_wrapper, selected_features_embedded)

# Model Training and Evaluation
def evaluate_model(X_train, X_test, y_train, y_test, feature_indices):
    """Train models using selected features and evaluate accuracy."""
    log_reg = LogisticRegression(max_iter=10000)

    # Filter Method Evaluation
    X_train_filter = X_train[:, feature_indices[0]]
    X_test_filter = X_test[:, feature_indices[0]]
    log_reg.fit(X_train_filter, y_train)
    y_pred_filter = log_reg.predict(X_test_filter)
    accuracy_filter = accuracy_score(y_test, y_pred_filter)

    # Wrapper Method Evaluation
    X_train_wrapper = X_train[:, feature_indices[1]]
    X_test_wrapper = X_test[:, feature_indices[1]]
    log_reg.fit(X_train_wrapper, y_train)
    y_pred_wrapper = log_reg.predict(X_test_wrapper)
    accuracy_wrapper = accuracy_score(y_test, y_pred_wrapper)

    # Embedded Method Evaluation
    X_train_embedded = X_train[:, feature_indices[2]]
    X_test_embedded = X_test[:, feature_indices[2]]
    log_reg.fit(X_train_embedded, y_train)
    y_pred_embedded = log_reg.predict(X_test_embedded)
    accuracy_embedded = accuracy_score(y_test, y_pred_embedded)

    return accuracy_filter, accuracy_wrapper, accuracy_embedded

# Hyperparameter Tuning
def tune_hyperparameters(X_train, y_train, feature_indices):
    """Perform hyperparameter tuning using RandomizedSearchCV."""
    rf_model = RandomForestClassifier(random_state=42)
    param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    # Create RandomizedSearchCV object
    random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist,
                                       n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, verbose=2, random_state=42)

    # Fit RandomizedSearchCV
    random_search.fit(X_train[:, feature_indices], y_train)
    return random_search.best_estimator_

# Execute all steps without main function
# Load data
X, y = load_data()

# Preprocess data
X_scaled = preprocess_data(X)

# Feature selection
feature_indices = feature_selection(X_scaled, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model evaluation
accuracies = evaluate_model(X_train, X_test, y_train, y_test, feature_indices)
print("\nModel Accuracies:")
print(f"Filter Method: {accuracies[0] * 100:.2f}%")
print(f"Wrapper Method: {accuracies[1] * 100:.2f}%")
print(f"Embedded Method: {accuracies[2] * 100:.2f}%")

# Hyperparameter tuning
best_rf_model = tune_hyperparameters(X_train, y_train, feature_indices[2])  # Use embedded method's selected features
test_accuracy = accuracy_score(y_test, best_rf_model.predict(X_test[:, feature_indices[2]]))
print(f"\nTest Accuracy with Best Random Forest Model: {test_accuracy * 100:.2f}%")

# Cross-Validation Scores
cv_scores = cross_val_score(best_rf_model, X_scaled, y, cv=3)
print("Mean Cross-Validated Accuracy: {:.2f}%".format(np.mean(cv_scores) * 100))
