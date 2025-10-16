# src/data/download_suppliers.py
from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_and_save(symbol: str, start: str = "2005-01-01", end: str = "2021-05-01"):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        print(f"[WARN] No data for {symbol}")
        return None
    # keep adjusted close and close
    df = df[['Open','High','Low','Close','Adj Close','Volume']].rename(columns={'Adj Close':'AdjClose','Close':'Close'})
    df.index = pd.to_datetime(df.index)
    out = DATA_DIR / f"{symbol}.parquet"
    df.to_parquet(out)
    print(f"[OK] Saved {symbol} -> {out}")
    return out

def download_all(list_file: str = "suppliers.txt"):
    with open(list_file, "r") as f:
        symbols = [line.strip() for line in f if line.strip()]
    for s in symbols:
        try:
            fetch_and_save(s)
        except Exception as e:
            print(f"[ERR] {s}: {e}")

if __name__ == "__main__":
    download_all()
