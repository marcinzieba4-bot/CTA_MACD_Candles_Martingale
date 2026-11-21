"""Diagnostics for mtf.py: (a) gross vs net edge per timeframe, (b) martingale vs fixed risk of the same average size."""
import numpy as np, pandas as pd
from backtest import Cfg, run
from mtf import build, daily_stats, COINS, TFS
pd.set_option("display.width", 250)

print("=== (a) average trade in R: no costs vs 5 bps/side; average stop distance and leverage (fixed 2% risk) ===")
rows = []
for coin in COINS:
    for tf in TFS:
        h, w, first = build(coin, tf)
        t0, _ = run(h, w, first, Cfg(cost=0.0)); t1, _ = run(h, w, first, Cfg())
        dist = (abs(t1.entry - t1.sl0) / t1.entry)
        rows.append(dict(coin=coin, tf=tf, trades=len(t1), avgR_gross=t0.R.mean(), avgR_net=t1.R.mean(), cost_in_R=t0.R.mean() - t1.R.mean(),
                         stop_dist_median=dist.median(), lev_mean=(0.02 / dist).clip(upper=3).mean(), sumR_net_per_year=t1.R.sum() / ((t1.exit_time.max() - t1.entry_time.min()).days / 365.25)))
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=== (b) W1: 'recover' martingale (cap 4x) vs FIXED risk equal to the martingale's average risk ===")
rows = []
for coin in COINS:
    h, w, first = build(coin, "W1")
    tm, _, hm = run(h, w, first, Cfg(martingale=True, mg_mode="recover", mg_cap=4.0), hourly=True); sm, _ = daily_stats(tm, hm)
    avg = tm.mult.mean()
    tf_, _, hf = run(h, w, first, Cfg(risk_frac=0.02 * avg), hourly=True); sf, _ = daily_stats(tf_, hf)
    streak = (tm.pnl <= 0).astype(int); longest = streak.groupby((streak == 0).cumsum()).sum().max()
    rows.append(dict(coin=coin, avg_mult=avg, share_at_cap=(tm.mult >= 3.999).mean(), share_at_1x=(tm.mult <= 1.001).mean(), longest_losing_streak=longest,
                     mart_final=sm["final"], mart_maxdd=sm["maxdd"], mart_sharpe=sm["sharpe"],
                     fixed_risk=0.02 * avg, fixed_final=sf["final"], fixed_maxdd=sf["maxdd"], fixed_sharpe=sf["sharpe"]))
print(pd.DataFrame(rows).round(3).to_string(index=False))
