"""Same rules on MN / W1 / D1 / H4 for BTC, ETH, SOL, with and without martingale, plus the 3-coin combo.

Execution is always simulated on Binance 1h bars. The regime and the candles come from the tested timeframe.
Daily Coinbase history (BTC from 2015, ETH from 2016) is used only to warm up the monthly/weekly MACD.
Trading starts once 35 bars of the timeframe exist, and not before COMMON_START.
"""
import os, sys
import numpy as np, pandas as pd
from backtest import Cfg, run

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
COINS = {"BTC": ("data/oos/btcusdt_1h_binance.csv.gz", "data/btcusd_1d_coinbase_warmup.csv"),
         "ETH": ("data/ethusdt_1h_binance.csv.gz", "data/ethusd_1d_coinbase_warmup.csv"),
         "SOL": ("data/oos/solusdt_1h_binance.csv.gz", None)}
TFS = ["MN", "W1", "D1", "H4"]
COMMON_START = pd.Timestamp("2018-05-07", tz="UTC")
SIZINGS = {"fixed 2%": dict(),
           "recover, cap 4x": dict(martingale=True, mg_mode="recover", mg_cap=4.0),
           "recover, cap 16x": dict(martingale=True, mg_mode="recover", mg_cap=16.0, max_lev=10.0),
           "double, max 4x": dict(martingale=True, mg_mode="double", mg_max_steps=2)}


def period_start(t, tf):
    if tf == "MN": return t.dt.to_period("M").dt.start_time.dt.tz_localize("UTC")
    if tf == "W1": return (t - pd.to_timedelta(t.dt.weekday, unit="D")).dt.normalize()
    if tf == "D1": return t.dt.normalize()
    if tf == "H4": return t.dt.floor("4h")


def build(coin, tf):
    hf, df = COINS[coin]
    h = pd.read_csv(hf, parse_dates=["time"])
    h["week"] = period_start(h.time, tf)                       # engine groups 1h bars by this column
    src = h
    if df and tf in ("MN", "W1"):                              # older daily bars: MACD warm-up only
        d = pd.read_csv(df, parse_dates=["time"]); d = d[d.time < h.time.min().normalize()].copy()
        d["week"] = period_start(d.time, tf); src = pd.concat([d, h])
    w = src.groupby("week").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
    w = w.iloc[:-1]                                            # drop the still-forming bar
    h = h[h.week.isin(w.index)]
    first = max(w.index[35], w.index[w.index >= COMMON_START][0], h.week.min())
    return h, w, first


def daily_stats(tr, hc):
    e = hc.mark.resample("1D").last().ffill(); r = e.pct_change().dropna()
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    worst = (hc.worst / hc.mark.cummax() - 1).min()
    out = dict(start=e.index[0].date(), final=e.iloc[-1], cagr=e.iloc[-1] ** (1 / yrs) - 1 if e.iloc[-1] > 0 else -1.0, maxdd=worst,
               sharpe=r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0.0, trades=len(tr))
    if len(tr):
        out.update(win=(tr.pnl > 0).mean(), avgR=tr.R.mean(), max_mult=tr.mult.max(),
                   tstat=tr.R.mean() / tr.R.std() * np.sqrt(len(tr)) if len(tr) > 2 else np.nan,
                   R_2025_26=tr.R[tr.entry_time.dt.year >= 2025].mean())
    return out, e


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    rows, curves = [], {}
    for coin in COINS:
        for tf in TFS:
            h, w, first = build(coin, tf)
            for nm, kw in SIZINGS.items():
                tr, cv, hc = run(h, w, first, Cfg(**kw), hourly=True)
                st, e = daily_stats(tr, hc)
                rows.append(dict(coin=coin, tf=tf, sizing=nm, **st)); curves[(coin, tf, nm)] = e
                print(rows[-1], file=sys.stderr)
    res = pd.DataFrame(rows); res.to_csv("results/mtf.csv", index=False)
    print("=== PER COIN x TIMEFRAME x SIZING (5 bps/side, drawdown on 1h worst prices) ===")
    print(res.round(3).to_string(index=False))

    print("\n=== COMBO BTC+ETH+SOL: each coin trades its own signals and risks 2% of the shared account ===")
    out = []
    for tf in TFS:
        for nm in SIZINGS:
            for label, coins in [("BTC+ETH+SOL", ["BTC", "ETH", "SOL"]), ("BTC+ETH", ["BTC", "ETH"])]:
                R = pd.concat([curves[(c, tf, nm)].pct_change() for c in coins], axis=1, keys=coins).dropna()
                p = R.sum(axis=1); e = (1 + p).cumprod(); yrs = len(p) / 365.25
                solo = {c: round(float(R[c].mean() / R[c].std() * np.sqrt(365)), 2) for c in coins}
                out.append(dict(tf=tf, sizing=nm, combo=label, start=R.index[0].date(), final=e.iloc[-1], cagr=e.iloc[-1] ** (1 / yrs) - 1 if e.iloc[-1] > 0 else -1,
                                maxdd_daily=(e / e.cummax() - 1).min(), sharpe=p.mean() / p.std() * np.sqrt(365), solo_sharpes=solo,
                                avg_corr=R.corr().values[np.triu_indices(len(coins), 1)].mean()))
    cb = pd.DataFrame(out); cb.to_csv("results/combo.csv", index=False)
    print(cb.round(3).to_string(index=False))
    pd.concat({f"{k[0]}_{k[1]}": v for k, v in curves.items() if k[2] == "fixed 2%"}, axis=1).to_csv("results/mtf_equity_fixed.csv")
