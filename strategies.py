import numpy as np
 
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))
 
 
def sma_rsi(df, fast=10, slow=50, rsi_max=70, rsi_n=14):
    """Beli saat SMA cepat > SMA lambat dan RSI belum overbought."""
    close = df["Close"]
    trend = close.rolling(fast).mean() > close.rolling(slow).mean()
    return (trend & (rsi(close, rsi_n) < rsi_max)).astype(int)
 
 
STRATEGIES = {
    "sma_rsi": {
        "signal": sma_rsi,
        "grid": [{"fast": f, "slow": s}
                 for f in (5, 10, 20) for s in (30, 50, 100, 200) if f < s],
    },
}