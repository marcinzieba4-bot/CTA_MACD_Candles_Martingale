"""Is the ETH W1 hold / breakeven / take-profit picture repeated on other coins? Rules untouched."""
import numpy as np, pandas as pd
from backtest import Cfg, run
import mtf
from mtf import build, daily_stats
from hold import HCfg, run_hold, hstats
pd.set_option("display.width", 250)
mtf.COINS.update({c: (f"data/oos/{c.lower()}usdt_1h_binance.csv.gz", None) for c in ["BNB", "XRP", "LTC", "ADA", "DOGE"]})
V = {"hold": {}, "hold+BE1R": dict(be_R=1), "BE1R+TP10%": dict(be_R=1, tp_pct=.1), "BE1R+TP20%": dict(be_R=1, tp_pct=.2), "BE1R+TP30%": dict(be_R=1, tp_pct=.3),
     "BE1R+TP50%": dict(be_R=1, tp_pct=.5), "hold, 3 legs": dict(max_legs=3), "hold, 6% risk": dict(risk_frac=.06)}
sh, dd = [], []
for c in mtf.COINS:
    h, w, first = build(c, "W1")
    tr, cv, hc = run(h, w, first, Cfg(), hourly=True); st, _ = daily_stats(tr, hc)
    a, b = dict(coin=c, reference=st["sharpe"]), dict(coin=c, reference=st["maxdd"])
    for k, kw in V.items():
        s = hstats(*run_hold(h, w, first, HCfg(**kw))); a[k] = s["sharpe"]; b[k] = s["maxdd"]
    sh.append(a); dd.append(b)
sh, dd = pd.DataFrame(sh).set_index("coin"), pd.DataFrame(dd).set_index("coin")
sh.loc["MEDIAN"] = sh.median(); dd.loc["MEDIAN"] = dd.median()
print("Sharpe (W1, 2% risk per leg, 5 bps/side); reference = previous base"); print(sh.round(2))
print("\nmax drawdown"); print(dd.round(2))
