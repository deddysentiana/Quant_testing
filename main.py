import argparse
from datetime import date
from pathlib import Path
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from backtest import run
from data import load_data
from metrics import metrics, show
from strategies import STRATEGIES
 
 
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="BBCA.JK")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=str(date.today()))
    p.add_argument("--strategy", default="sma_rsi", choices=STRATEGIES)
    p.add_argument("--split", type=float, default=0.7, help="porsi in-sample")
    p.add_argument("--synthetic", action="store_true")
    a = p.parse_args()
 
    df = load_data(a.ticker, a.start, a.end, a.synthetic)
    cut = int(len(df) * a.split)
    strat = STRATEGIES[a.strategy]
 
    # Optimasi parameter HANYA di in-sample
    best, best_sharpe = None, -np.inf
    for params in strat["grid"]:
        res = run(df, strat["signal"](df, **params)).iloc[:cut]
        sharpe = metrics(res["net"])["Sharpe"]
        if sharpe > best_sharpe:
            best, best_sharpe = params, sharpe
    print(f"{a.ticker} | {a.strategy} | parameter terbaik (in-sample): {best}")
 
    res = run(df, strat["signal"](df, **best))
    ins, oos = res.iloc[:cut], res.iloc[cut:]
    show("IN-SAMPLE (strategi)", metrics(ins["net"], ins["pos"]))
    show("OUT-OF-SAMPLE (strategi)  <- angka yang jujur", metrics(oos["net"], oos["pos"]))
    show("OUT-OF-SAMPLE (buy & hold)", metrics(oos["ret"]))
 
    out = Path("out")
    out.mkdir(exist_ok=True)
    plt.figure(figsize=(10, 5))
    (1 + res["net"]).cumprod().plot(label=f"Strategi {a.strategy}")
    (1 + res["ret"]).cumprod().plot(label="Buy & hold")
    plt.axvline(df.index[cut], color="gray", ls="--", label="batas in/out-of-sample")
    plt.title(f"Equity curve {a.ticker}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "equity.png", dpi=120)
    print(f"\nGrafik disimpan: {out / 'equity.png'}")
 
 
if __name__ == "__main__":
    main()