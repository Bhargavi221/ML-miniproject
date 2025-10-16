import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# -----------------------------
# MODEL TRAINING
# -----------------------------
def train_model(model_name, X_train, y_train):
    if model_name == "ElasticNet":
        model = ElasticNetCV(cv=5, l1_ratio=[0.1, 0.5, 0.9], n_alphas=50, max_iter=5000)
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


# -----------------------------
# ROLLING WINDOW TRAINING
# -----------------------------
def rolling_window_training(df, model_name, train_size=1250, val_size=250, test_size=160):
    """
    Rolling-window training and testing.
    Each window: 5y train, 1y val, 8m test (approx for daily data).
    """
    X = df.drop(columns=['target']).values
    y = df['target'].values

    n_samples = len(df)
    results = []

    start = 0
    while start + train_size + val_size + test_size < n_samples:
        train_end = start + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size

        X_train, y_train = X[start:train_end], y[start:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:test_end], y[val_end:test_end]

        model = train_model(model_name, X_train, y_train)
        y_pred_val, rmse_val = evaluate_model(model, X_val, y_val)
        y_pred_test = model.predict(X_test)

        results.append({
            "window_start": start,
            "rmse_val": rmse_val,
            "y_true": y_test,
            "y_pred": y_pred_test
        })

        start += test_size  # slide forward

    return results


# -----------------------------
# TRADING SIMULATION
# -----------------------------
def simulate_trading(y_true, y_pred, threshold=0.005, capital=1000):
    """
    Simple long/short strategy based on predicted returns.
    Buy if prediction > threshold, short if < -threshold, else hold.
    """
    position = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
    trade_returns = position * y_true

    pnl = capital * (1 + trade_returns).cumprod()[-1] - capital
    num_trades = np.count_nonzero(position)

    return pnl, num_trades


# -----------------------------
# EVALUATE STRATEGY OVER WINDOWS
# -----------------------------
def evaluate_strategy(results):
    thresholds = [0, 0.0025, 0.005, 0.0075, 0.01]
    summary = []

    for thr in thresholds:
        total_pnl, total_trades = 0, 0
        rmses = []

        for res in results:
            pnl, trades = simulate_trading(res['y_true'], res['y_pred'], threshold=thr)
            total_pnl += pnl
            total_trades += trades
            rmses.append(res['rmse_val'])

        avg_rmse = np.mean(rmses)
        summary.append({
            "Threshold": thr,
            "Avg RMSE": round(avg_rmse, 6),
            "Total Profit (EUR)": round(total_pnl, 2),
            "Total Trades": int(total_trades)
        })

    return pd.DataFrame(summary)


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run_all_models(df):
    models = ["ElasticNet", "DecisionTree", "XGBoost", "LightGBM"]
    all_results = {}

    for m in models:
        print(f"\n🚀 Running {m} model...")
        results = rolling_window_training(df, m)
        summary = evaluate_strategy(results)
        all_results[m] = summary
        print(summary)

    return all_results


# -----------------------------
# OPTIONAL: SAVE RESULTS
# -----------------------------
def save_results(all_results, path="src/outputs/model_comparison.xlsx"):
    with pd.ExcelWriter(path) as writer:
        for model_name, df_summary in all_results.items():
            df_summary.to_excel(writer, sheet_name=model_name, index=False)
    print(f"\n💾 Results saved to {path}")
