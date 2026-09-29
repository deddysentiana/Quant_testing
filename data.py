from pathlib import Path
 
import numpy as np
import pandas as pd
 
 
def load_data(ticker, start, end, synthetic=False):
    if synthetic:  # data acak untuk tes tanpa internet
        rng = np.random.default_rng(42)
        idx = pd.bdate_range("2018-01-01", periods=1800)
        close = 5000 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(idx))))
        return pd.DataFrame({"Close": close}, index=idx)
 
    cache = Path("data") / f"{ticker}_{start}_{end}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
 
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(f"Data kosong untuk {ticker}. Cek kode ticker (contoh: BBCA.JK).")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Close"]].dropna()
    cache.parent.mkdir(exist_ok=True)
    df.to_csv(cache)
    return df