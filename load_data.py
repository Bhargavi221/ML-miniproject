import os
import pandas as pd

def load_returns(tickers, data_dir="src/data/raw"):
    """
    Load stock returns from CSV files for given tickers.

    Returns a DataFrame with daily percentage returns for all tickers.
    """
    all_returns = []

    for ticker in tickers:
        file_path_csv = os.path.join(data_dir, f"{ticker}.csv")
        if not os.path.exists(file_path_csv):
            raise FileNotFoundError(f"❌ File not found: {file_path_csv}")

        # Skip first 3 rows and set proper column names
        df = pd.read_csv(
            file_path_csv,
            skiprows=3,
            names=["Date", "Close", "High", "Low", "Open", "Volume"],
            parse_dates=["Date"],
        )

        # Convert Close to float
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df.dropna(subset=['Close'], inplace=True)

        # Set Date as index
        df.set_index('Date', inplace=True)

        # Keep only the Close series, rename to ticker
        series = df['Close'].rename(ticker)
        all_returns.append(series)

    # Combine all tickers into one DataFrame
    returns_df = pd.concat(all_returns, axis=1)

    # Compute daily percentage returns
    returns_df = returns_df.pct_change().dropna()

    print(f"✅ Loaded returns for {len(tickers)} tickers: {returns_df.shape}")
    return returns_df
