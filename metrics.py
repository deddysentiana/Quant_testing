import numpy as np
 
 
def metrics(net, pos=None):
    eq = (1 + net).cumprod()
    yrs = len(net) / 252
    out = {
        "Total return": eq.iloc[-1] - 1,
        "CAGR": eq.iloc[-1] ** (1 / yrs) - 1,
        "Sharpe": net.mean() / net.std() * np.sqrt(252) if net.std() > 0 else 0.0,
        "Max drawdown": (eq / eq.cummax() - 1).min(),
    }
    if pos is not None:
        tid = (pos.diff().fillna(pos) > 0).cumsum().where(pos > 0)
        trades = net.groupby(tid).apply(lambda x: (1 + x).prod() - 1)
        out["Jumlah trade"] = len(trades)
        out["Win rate"] = (trades > 0).mean() if len(trades) else 0.0
    return out
 
 
def show(title, m):
    print(f"\n{title}")
    for k, v in m.items():
        if k == "Sharpe":
            print(f"  {k:<14}: {v:>9.2f}")
        elif k == "Jumlah trade":
            print(f"  {k:<14}: {v:>9d}")
        else:
            print(f"  {k:<14}: {v:>9.2%}")