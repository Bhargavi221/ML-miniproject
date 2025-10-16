# src/data/download_data.py
import yfinance as yf
from pathlib import Path

DATA_DIR = Path(__file__).parent / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def download_tickers(tickers, start="2005-01-01", end="2021-05-01"):
    missing = []
    for ticker in tickers:
        file_path = DATA_DIR / f"{ticker}.csv"
        if file_path.exists():
            print(f"{ticker} already exists. Skipping download.")
            continue
        try:
            print(f"Downloading {ticker}...")
            df = yf.download(ticker, start=start, end=end, auto_adjust=True)
            if df.empty:
                print(f"Warning: {ticker} returned no data!")
                missing.append(ticker)
                continue
            df.to_csv(file_path)
            print(f"{ticker} saved to {file_path}")
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            missing.append(ticker)
    if missing:
        print("Failed downloads:", missing)
    return missing

if __name__ == "__main__":
    # example tickers
    tickers = ["BMW.DE", "SAP.DE", "VOW3.DE", "ALV.DE", "CON.DE"]
    download_tickers(tickers)
