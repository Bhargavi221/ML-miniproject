import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

def train_model(model_name, X_train, y_train):
    if model_name == "ElasticNet":
        model = ElasticNetCV(cv=5, l1_ratio=[0.1,0.5,0.9], n_alphas=50, max_iter=5000)
    elif model_name == "DecisionTree":
        model = DecisionTreeRegressor(max_depth=5)
    elif model_name == "XGBoost":
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)
    elif model_name == "LightGBM":
        model = lgb.LGBMRegressor(n_estimators=100, max_depth=5)
    else:
        raise ValueError(f"Unknown model {model_name}")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return y_pred, rmse
