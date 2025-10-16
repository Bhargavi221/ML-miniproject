import numpy as np

def backtest(y_pred, y_true, threshold=0.0, initial_capital=1.0):
    """
    Simple backtest: go long if prediction > threshold, short if < -threshold.
    """
    positions = np.where(y_pred > threshold, 1, np.where(y_pred < -threshold, -1, 0))
    returns = positions * y_true
    capital = initial_capital * (1 + returns).cumprod()[-1]
    trades = np.sum(np.abs(np.diff(positions)) > 0)
    return capital, int(trades)
