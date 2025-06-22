import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score, mean_absolute_error,
    root_mean_squared_error, confusion_matrix,
    f1_score
)
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to save confusion matrix
def save_confusion_matrix_png(cm, labels, output_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Binned Regression Output)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Load data
test_df = pd.read_csv("co2emissions_preprocessing/test_processed.csv")
train_df = pd.read_csv("co2emissions_preprocessing/train_processed.csv")

X_train = train_df.drop(columns=['target'])
y_train = train_df['target']
X_test = test_df.drop(columns=['target'])
y_test = test_df['target']

input_example = X_train.iloc[0:5]

# Enable autolog before training (so it's active!)
mlflow.autolog(log_models=False)  # Disable model autolog to avoid double logging

# Discretize regression output into bins for classification metrics
# You can skip this part if your task is truly regression only
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
y_train_binned = discretizer.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_binned = discretizer.transform(y_test.values.reshape(-1, 1)).ravel()

# Grid Search
param_grid = {'n_estimators': [50, 100], 'max_depth': [8, 12], 'n_jobs': [1]}
grid_search = GridSearchCV(RandomForestRegressor(random_state=10), param_grid, cv=3)

with mlflow.start_run():
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)

    # Classification-friendly binning for confusion matrix & f1
    y_pred_binned = discretizer.transform(y_pred.reshape(-1, 1)).ravel()

    # Custom metrics
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    f1 = f1_score(y_test_binned, y_pred_binned, average='macro')
    cm = confusion_matrix(y_test_binned, y_pred_binned)
    cm_labels = [f"Bin {i}" for i in range(cm.shape[0])]
    cm_path = "confusion_matrix.png"
    save_confusion_matrix_png(cm, cm_labels, cm_path)

    # Log additional metrics manually
    mlflow.log_metric("custom_r2_score", r2)
    mlflow.log_metric("custom_rmse", rmse)
    mlflow.log_metric("custom_mae", mae)
    mlflow.log_metric("custom_f1_score", f1)
    mlflow.log_artifact("confusion_matrix.png")

    # Flatten confusion matrix for logging
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            mlflow.log_metric(f"confusion_matrix_{i}_{j}", cm[i][j])

    # Log model manually (to add input example)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        input_example=input_example,
        registered_model_name=None
    )
    