# src/models/test_models.py
import numpy as np
from sklearn.metrics import mean_squared_error
from .train_models import train_elasticnet, train_decision_tree, train_xgboost, train_lightgbm

# ----------------------------
# Generate synthetic test data
# ----------------------------
np.random.seed(42)
X_train = np.random.randn(100, 5)
y_train = np.random.randn(100)
X_val = np.random.randn(20, 5)
y_val = np.random.randn(20)

# ----------------------------
# Helper function to evaluate models
# ----------------------------
def evaluate_model(train_func, X_train, y_train, X_val, y_val, name="Model"):
    model, rmse = train_func(X_train, y_train, X_val, y_val)
    print(f"{name} RMSE on validation set: {rmse:.5f}")
    return model, rmse

# ----------------------------
# Test all models
# ----------------------------
if __name__ == "__main__":
    print("Testing all models on synthetic data...\n")

    evaluate_model(train_elasticnet, X_train, y_train, X_val, y_val, "ElasticNet")
    evaluate_model(train_decision_tree, X_train, y_train, X_val, y_val, "Decision Tree")
    evaluate_model(train_xgboost, X_train, y_train, X_val, y_val, "XGBoost")
    evaluate_model(train_lightgbm, X_train, y_train, X_val, y_val, "LightGBM")
