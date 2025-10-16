"""
Main script for Quantitative Trading with Machine Learning
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# ✅ Import project modules
from .data.load_data import load_returns
from .features.multi_features import create_lagged_features
from .models.train_models import train_model, evaluate_model


# =========================
# 1️⃣ Load Data
# =========================
tickers = ["ALV.DE", "BMW.DE", "CON.DE", "SAP.DE", "VOW3.DE"]


print("📥 Loading returns data...")
returns = load_returns(tickers)
print(f"✅ Loaded returns: {returns.shape}")

# =========================
# 2️⃣ Create Lagged Features
# =========================
print("🧩 Creating lagged features...")
X = create_lagged_features(returns, lags=5)
y = returns[tickers[0]].iloc[len(returns) - len(X):]  # predicting first ticker
print(f"✅ Features created: {X.shape}, Target: {y.shape}")

# =========================
# 3️⃣ Train/Test Split
# =========================
split_ratio = 0.8
split_idx = int(len(X) * split_ratio)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

if X_train.empty or X_test.empty:
    raise ValueError(f"❌ Empty train/test split: X_train={X_train.shape}, X_test={X_test.shape}")

print(f"✅ Split data: train={X_train.shape}, test={X_test.shape}")

# =========================
# 4️⃣ Train & Evaluate Multiple Models
# =========================
models_to_run = ["ElasticNet", "DecisionTree", "XGBoost", "LightGBM"]
results = {}

for model_name in models_to_run:
    print(f"\n🤖 Training {model_name} model...")
    model = train_model(model_name, X_train, y_train)
    y_pred, rmse = evaluate_model(model, X_test, y_test)
    r2 = r2_score(y_test, y_pred)
    print(f"✅ {model_name} Evaluation - RMSE: {rmse:.6f}, R²: {r2:.4f}")
    results[model_name] = {"model": model, "y_pred": y_pred, "rmse": rmse, "r2": r2}

# =========================
# 5️⃣ Save Outputs & Plots
# =========================
output_dir = os.path.join("src", "outputs")
os.makedirs(output_dir, exist_ok=True)

for model_name, res in results.items():
    preds_path = os.path.join(output_dir, f"{model_name}_predictions.csv")
    pd.DataFrame({"Actual": y_test, "Predicted": res["y_pred"]}).to_csv(preds_path, index=False)
    print(f"💾 Predictions saved for {model_name}: {preds_path}")

    # Line Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='Actual Returns', color='blue', alpha=0.6)
    plt.plot(res["y_pred"], label='Predicted Returns', color='orange', alpha=0.8)
    plt.title(f"{model_name}: Predicted vs Actual Returns")
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    line_plot_path = os.path.join(output_dir, f"{model_name}_line_plot.png")
    plt.savefig(line_plot_path, dpi=300)
    plt.close()
    print(f"📈 Saved line plot: {line_plot_path}")

    # Scatter Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, res["y_pred"], alpha=0.5)
    plt.title(f"{model_name}: Predicted vs Actual Scatter")
    plt.xlabel("Actual Returns")
    plt.ylabel("Predicted Returns")
    plt.grid(True)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f"{model_name}_scatter_plot.png")
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"📊 Saved scatter plot: {scatter_path}")

print("\n✅ All models trained and evaluated successfully!")
