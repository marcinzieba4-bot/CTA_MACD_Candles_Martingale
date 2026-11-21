"""ETH W1: hold / breakeven / take-profit / pyramiding study."""
import sys, itertools
import numpy as np, pandas as pd
from backtest import Cfg, run
from mtf import build, daily_stats
from hold import HCfg, run_hold, hstats
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 300)
COIN, TF = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("ETH", "W1")
h, w, first = build(COIN, TF)
cols = ["final", "cagr", "maxdd", "sharpe", "trades", "win", "avgR", "sumR", "tstat", "R_2025_26", "hold_days", "avg_lev", "in_mkt"]
R = lambda kw: {k: v for k, v in hstats(*run_hold(h, w, first, HCfg(**kw))).items() if k in cols}

tr, cv, hc = run(h, w, first, Cfg(), hourly=True); st, _ = daily_stats(tr, hc)
print(f"{COIN} {TF} from {first.date()}  | REFERENCE (previous base: stop re-set under each new candle, exit when a week has no setup A): "
      f"final {st['final']:.2f}x  maxDD {st['maxdd']:.1%}  Sharpe {st['sharpe']:.2f}  trades {st['trades']}  avgR {st['avgR']:.2f}")

print("\n=== 1. HOLD while regime intact (single position, initial stop never moved) + breakeven + take-profit ===")
BE = {"no BE": {}, "BE after +1R": dict(be_R=1), "BE after +2R": dict(be_R=2), "BE after 2 candles in profit": dict(be_bars=2), "BE after 4 candles in profit": dict(be_bars=4)}
TP = {"no TP": None, "TP 10%": .10, "TP 20%": .20, "TP 30%": .30, "TP 50%": .50, "TP 100%": 1.0}
rows = []
for (bn, bk), (tn, tp) in itertools.product(BE.items(), TP.items()):
    rows.append(dict(breakeven=bn, take_profit=tn, **R(dict(bk, tp_pct=tp))))
g = pd.DataFrame(rows); g.to_csv(f"results/hold_{COIN}_{TF}_grid.csv", index=False)
print(g.round(3).to_string(index=False))
print("\nSharpe:"); print(g.pivot(index="breakeven", columns="take_profit", values="sharpe")[list(TP)].loc[list(BE)].round(2))
print("\nfinal equity (x):"); print(g.pivot(index="breakeven", columns="take_profit", values="final")[list(TP)].loc[list(BE)].round(2))
print("\nmax drawdown:"); print(g.pivot(index="breakeven", columns="take_profit", values="maxdd")[list(TP)].loc[list(BE)].round(3))

print("\n=== 2. exit reasons for three representative cells ===")
for nm, kw in [("hold, no BE, no TP", {}), ("hold, BE +1R, no TP", dict(be_R=1)), ("hold, BE +1R, TP 20%", dict(be_R=1, tp_pct=.2))]:
    t_, _ = run_hold(h, w, first, HCfg(**kw))
    print(nm); print(t_.groupby("reason").agg(n=("R", "size"), avgR=("R", "mean"), sumR=("R", "sum")).round(2).to_string())

print("\n=== 3. PYRAMIDING (2% risk per leg, total leverage cap 3x) ===")
rows = []
for (bn, bk), tp, legs, rule in itertools.product([("no BE", {}), ("BE after +1R", dict(be_R=1))], [None, .20], [1, 2, 3, 5], ["any", "protected", "profit"]):
    if legs == 1 and rule != "any": continue
    rows.append(dict(breakeven=bn, tp="TP 20%" if tp else "no TP", max_legs=legs, add_rule=rule if legs > 1 else "-", **R(dict(bk, tp_pct=tp, max_legs=legs, add_rule=rule))))
p = pd.DataFrame(rows); p.to_csv(f"results/hold_{COIN}_{TF}_pyramid.csv", index=False); print(p.round(3).to_string(index=False))

print("\n=== 4. is pyramiding more than leverage? single position at higher risk vs pyramid of similar average leverage ===")
for rf in [0.02, 0.04, 0.06]:
    s = R(dict(risk_frac=rf)); print(f"single leg, no BE/TP, risk {rf:.0%}: final {s['final']:.1f}x maxDD {s['maxdd']:.1%} Sharpe {s['sharpe']:.2f} avg lev {s['avg_lev']:.2f}")
