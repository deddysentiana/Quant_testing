import numpy as np
import pandas as pd

B_FEE = 0.0015
S_FEE = 0.0025

def run(df, signal):
    pos = signal.shift(1).fillna(0)
    ret = df["Close"].pct_change().fillna(0)
    chg = pos.diff().fillna(pos)
    cost = np.where(chg > 0, B_FEE , 0) + np.where(chg < 0, S_FEE, 0)
    return pd.DataFrame({"pos": pos, "ret": ret, "net": pos * ret - cost})