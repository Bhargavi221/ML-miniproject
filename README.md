# ML-miniproject


# Quantitative Trading Using Machine Learning

## Project Overview
This project demonstrates a machine learning-based quantitative trading model that predicts future stock price movements using historical price data and basic technical indicators.

Two models — Decision Tree Regressor and XGBoost Regressor — are trained and compared to analyze their performance in forecasting next-day stock closing prices.

---

## Problem Statement
Stock price movements are influenced by numerous unpredictable factors, making prediction highly challenging. Traditional models often fail to capture nonlinear relationships in market data.

This project aims to:
> Develop a simple ML-based trading model that can predict stock price trends based on historical data.

---

## Approach

### 1. Data Collection
- Historical stock data obtained from Yahoo Finance using the `yfinance` library.
- Dataset includes approximately 100–200 days of price information.
- Columns typically include: Open, High, Low, Close, Volume.

### 2. Feature Engineering
- Computed technical indicators:
  - Moving Averages (MA5, MA10, MA20)
  - Daily Returns
  - Price Momentum
- Target variable: Next-day closing price.

### 3. Modeling
- **Decision Tree Regressor** (baseline)
- **XGBoost Regressor** (advanced ensemble model)

Both models were trained on the same dataset and evaluated on test data.

### 4. Evaluation
- Predictions stored in:
  - `DecisionTree_predictions.csv`
  - `XGBoost_predictions.csv`
- Models compared using metrics such as Mean Squared Error (MSE) and trend matching visualization.

---

## Implementation

**Languages and Libraries Used**
- Python  
- Pandas, NumPy  
- scikit-learn  
- XGBoost  
- Matplotlib  
- yfinance  

**Steps to Run**
```bash
# 1. Clone this repository
git clone https://github.com/Bhargavi221/ML-miniproject.git
cd ML-miniproject

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the model
python train_models.py

# 4. View predictions
open DecisionTree_predictions.csv
open XGBoost_predictions.csv


Results

XGBoost outperformed Decision Tree by providing smoother and more realistic price predictions.

Decision Tree overfit on training data, while XGBoost generalized better.

The project demonstrates the use of ML in data-driven trading with minimal setup.

Challenges

Limited dataset (only about 100 days) restricted performance.

High market volatility and noise.

Avoiding data leakage in time-series splits.

Conclusions

Machine learning can effectively capture stock market trends, even with simple models.

Ensemble methods like XGBoost show better adaptability to dynamic markets.

Future improvements could include:

Longer datasets

Feature scaling

Deep learning models (e.g., LSTM)
