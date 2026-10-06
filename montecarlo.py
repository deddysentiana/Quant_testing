import numpy as np
import pandas as pd

from backtest import run


def _stats(r):
    """Metrik vektor untuk matriks return (simulasi x hari)."""
    eq = np.cumprod(1 + r, axis=1)
    sd = r.std(axis=1)
    return {
        "Total return": eq[:, -1] - 1,
        "Sharpe": np.divide(r.mean(axis=1), sd, out=np.zeros(len(r)), where=sd > 0) * np.sqrt(252),
        "Max drawdown": (eq / np.maximum.accumulate(eq, axis=1) - 1).min(axis=1),
    }


def bootstrap(net, n=1000, block=20, seed=0):
    """Block bootstrap return harian: blok berurutan dipertahankan agar autokorelasi tidak hilang."""
    rng = np.random.default_rng(seed)
    r = np.asarray(net)
    k = -(-len(r) // block)
    starts = rng.integers(0, len(r), (n, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :len(r)] % len(r)
    return _stats(r[idx])


def permutation_test(df, cut, signal_fn, params, n=200, seed=0):
    """Acak urutan return harga OOS lalu jalankan ulang strategi.
    p-value kecil = strategi menangkap pola yang tidak ada di harga acak."""
    rng = np.random.default_rng(seed)
    close = df["Close"].to_numpy()
    logr = np.diff(np.log(close[cut - 1:]))
    real = run(df, signal_fn(df, **params)).iloc[cut:]["net"]
    real_sharpe = _stats(real.to_numpy()[None])["Sharpe"][0]

    sims = np.empty(n)
    for i in range(n):
        fake = close[cut - 1] * np.exp(np.cumsum(rng.permutation(logr)))
        d = pd.DataFrame({"Close": np.concatenate([close[:cut], fake])}, index=df.index)
        net = run(d, signal_fn(d, **params)).iloc[cut:]["net"].to_numpy()
        sims[i] = _stats(net[None])["Sharpe"][0]
    p = (np.sum(sims >= real_sharpe) + 1) / (n + 1)
    return real_sharpe, sims, p


def show_mc(title, dist):
    print(f"\n{title}")
    print(f"  {'':<14}  {'P5':>9} {'Median':>9} {'P95':>9}")
    for k, v in dist.items():
        q = np.percentile(v, [5, 50, 95])
        fmt = "{:>9.2f}" if k == "Sharpe" else "{:>9.2%}"
        print(f"  {k:<14}: " + " ".join(fmt.format(x) for x in q))
    print(f"  {'P(rugi)':<14}: {np.mean(dist['Total return'] < 0):>9.2%}")
