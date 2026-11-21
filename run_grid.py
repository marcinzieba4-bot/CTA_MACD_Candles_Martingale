"""Run all strategy variants and robustness cuts; writes results/*.csv."""
import itertools, os
import numpy as np, pandas as pd
from backtest import Cfg, load_data, run, stats

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
os.makedirs("results", exist_ok=True)
h, w, fw = load_data()

rows = []
for line, sl, hold, re, mg in itertools.product(["signal", "macd"], ["low", "halfwick"], ["weekly", "trail"], [0, 1, 2], [False, True]):
    cfg = Cfg(regime_line=line, sl_mode=sl, hold_mode=hold, max_reentries=re, martingale=mg)
    tr, cv = run(h, w, fw, cfg)
    rows.append(dict(line=line, sl=sl, hold=hold, reentries=re, martingale=mg, **stats(tr, cv)))
g = pd.DataFrame(rows)
g.to_csv("results/grid.csv", index=False)
cols = ["line", "sl", "hold", "reentries", "martingale", "final", "cagr", "maxdd", "mar", "sharpe", "trades", "win", "avgR", "pf", "long_pnl", "short_pnl"]
print("=== GRID (risk 2%/trade, lev cap 3x, cost 0.1%/side) ==="); print(g[cols].round(3).to_string())

bh = w.close[w.index >= fw]
yrs = (bh.index[-1] - bh.index[0]).days / 365.25
print(f"\nBuy&hold: x{bh.iloc[-1]/bh.iloc[0]:.2f}, CAGR {(bh.iloc[-1]/bh.iloc[0])**(1/yrs)-1:.1%}, maxDD {(bh/bh.cummax()-1).min():.1%}, {bh.index[0].date()} -> {bh.index[-1].date()}")

base = Cfg()
tr, cv = run(h, w, fw, base)
tr.to_csv("results/trades_base.csv", index=False); cv.to_csv("results/equity_base.csv")
print("\n=== BASE: by setup / kind / direction ===")
print(tr.groupby(["dir", "setup", "kind"]).agg(n=("R", "size"), win=("pnl", lambda x: (x > 0).mean()), avgR=("R", "mean"), sumR=("R", "sum")).round(2))
print("\nexit reasons:"); print(tr.groupby("reason").agg(n=("R", "size"), avgR=("R", "mean"), sumR=("R", "sum")).round(2))
print("\n=== BASE: sum of R per year (by entry) ===")
y = tr.assign(year=pd.to_datetime(tr.entry_time, utc=True).dt.year).groupby("year").agg(n=("R", "size"), sumR=("R", "sum"), win=("pnl", lambda x: (x > 0).mean()))
y["equity_ret"] = cv.equity.groupby(cv.index.year).last().pct_change().fillna(cv.equity.groupby(cv.index.year).last().iloc[0] - 1)
y["eth_ret"] = cv.close.groupby(cv.index.year).last().pct_change()
print(y.round(2))
print("\ntop 5 trades share of total R:", round(tr.R.nlargest(5).sum() / tr.R.sum(), 2), "| max losing streak:",
      max((sum(1 for _ in grp) for k, grp in itertools.groupby(tr.pnl <= 0) if k), default=0))

print("\n=== ROBUSTNESS around base ===")
alts = {"base": {}, "long side only": dict(sides="long"), "short side only": dict(sides="short"), "ABLATION regime inverted": dict(invert_regime=True), "A only": dict(use_B=False), "B only": dict(use_A=False),
        "cost 0.3%/side": dict(cost=0.003), "notional 1x": dict(sizing="notional", max_lev=1.0), "notional 1x halfwick": dict(sizing="notional", max_lev=1.0, sl_mode="halfwick"),
        "risk 5%": dict(risk_frac=0.05), "level re-entry x1": dict(max_reentries=1), "martingale x2 (max 4x)": dict(martingale=True), "martingale x2 (max 16x)": dict(martingale=True, mg_max_steps=4)}
r = []
for k, v in alts.items():
    t2, c2 = run(h, w, fw, Cfg(**v)); r.append(dict(variant=k, **stats(t2, c2)))
r = pd.DataFrame(r); r.to_csv("results/robustness.csv", index=False)
print(r[["variant", "final", "cagr", "maxdd", "mar", "sharpe", "trades", "win", "avgR", "pf"]].round(3).to_string())

print("\n=== RE-ENTRY designs (signal line, hold=weekly) ===")
rr = []
for sl in ["low", "halfwick"]:
    for mode, n in [("none", 0), ("level", 1), ("level", 2), ("spring", 1), ("spring", 2), ("spring", 3)]:
        t2, c2 = run(h, w, fw, Cfg(sl_mode=sl, max_reentries=n, reentry_mode=mode if n else "level")); s = stats(t2, c2)
        re_ = t2[t2.kind == "reentry"]
        rr.append(dict(sl=sl, reentry=mode, max_per_week=n, final=s["final"], cagr=s["cagr"], maxdd=s["maxdd"], mar=s["mar"], trades=s["trades"],
                       re_n=len(re_), re_win=(re_.pnl > 0).mean() if len(re_) else np.nan, re_avgR=re_.R.mean() if len(re_) else np.nan,
                       re_sumR=re_.R.sum(), re_pnl=re_.pnl.sum(), re_long_R=re_.R[re_.dir == 1].sum(), re_short_R=re_.R[re_.dir == -1].sum()))
rr = pd.DataFrame(rr); rr.to_csv("results/reentries.csv", index=False); print(rr.round(3).to_string())

print("\n=== MACD parameter sensitivity (base rules) ===")
import backtest
orig = backtest.add_macd
for p in [(8, 17, 9), (12, 26, 9), (19, 39, 9), (24, 52, 18)]:
    backtest.add_macd = lambda w_, p=p: orig(w_, *p)
    for line in ["signal", "macd"]:
        t2, c2 = run(h, w, fw, Cfg(regime_line=line)); s = stats(t2, c2)
        print(p, line, {k: round(s[k], 2) for k in ["final", "cagr", "maxdd", "avgR", "trades"]})
backtest.add_macd = orig

print("\n=== halves (base) ===")
mid = cv.index[len(cv) // 2]
for nm, c in [("first half", cv[cv.index < mid]), ("second half", cv[cv.index >= mid])]:
    e = c.equity / c.equity.iloc[0]; yy = (c.index[-1] - c.index[0]).days / 365.25
    print(nm, c.index[0].date(), "->", c.index[-1].date(), f"CAGR {e.iloc[-1]**(1/yy)-1:.1%} maxDD {(e/e.cummax()-1).min():.1%}")
