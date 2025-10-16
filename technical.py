import pandas as pd

def add_technical_indicators(df):
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['STD5'] = df['Close'].rolling(5).std()
    df['RSI'] = compute_rsi(df['Close'])
    return df.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    RS = gain / loss
    return 100 - (100 / (1 + RS))
