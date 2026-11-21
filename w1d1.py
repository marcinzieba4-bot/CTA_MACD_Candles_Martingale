"""W1 regime + D1 entries: direction from the weekly MACD signal line (last CLOSED week), candle rules on daily bars."""
import sys
import numpy as np, pandas as pd
from backtest import Cfg, run, add_macd
from mtf import build, daily_stats, COINS
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
ALL = dict(COINS, **{c: (f"data/oos/{c.lower()}usdt_1h_binance.csv.gz", None) for c in ["BNB", "XRP", "LTC", "ADA", "DOGE"]})
COINS.update(ALL)


def weekly_regime_on_days(coin, d_index):
    """For every daily bar: sign(signal[last closed week] - signal[week before]). Nothing from the running week."""
    _, ww, wfirst = build(coin, "W1")
    s = add_macd(ww).signal
    last, prev = s.copy(), s.copy()
    last.index = last.index + pd.Timedelta(days=7); prev.index = prev.index + pd.Timedelta(days=14)   # label = week in which it is known
    reg = np.sign(last - prev.reindex(last.index)).replace(0, np.nan).ffill()
    wk = d_index - pd.to_timedelta(d_index.weekday, unit="D")
    return reg.reindex(wk).fillna(0).values, wfirst


def setup(coin):
    h, w, first = build(coin, "D1")
    reg, wfirst = weekly_regime_on_days(coin, w.index)
    return h, w, max(first, wfirst), reg


if __name__ == "__main__":
    S = lambda st: {k: st.get(k) for k in ["start", "final", "cagr", "maxdd", "sharpe", "trades", "win", "avgR", "tstat", "R_2025_26"]}
    rows, curves = [], {}
    for coin in ALL:
        h, w, first, reg = setup(coin)
        for nm, kw, ro in [("W1 regime + D1 entries", {}, reg), ("D1 regime + D1 entries", {}, None)]:
            tr, cv, hc = run(h, w, first, Cfg(**kw), regime_override=ro, hourly=True); st, e = daily_stats(tr, hc)
            rows.append(dict(coin=coin, system=nm, **S(st))); curves[(coin, nm)] = e
        hw, ww, wfirst = build(coin, "W1")
        tr, cv, hc = run(hw, ww, max(wfirst, first - pd.Timedelta(days=first.weekday())), Cfg(), hourly=True); st, e = daily_stats(tr, hc)
        rows.append(dict(coin=coin, system="W1 regime + W1 entries", **S(st))); curves[(coin, "W1 regime + W1 entries")] = e
        print(coin, file=sys.stderr)
    r = pd.DataFrame(rows); r.to_csv("results/w1d1.csv", index=False)
    print("=== per coin (2% risk/trade, 5 bps/side, drawdown on 1h worst prices) ==="); print(r.round(3).to_string(index=False))
    print("\n=== median over the 8 coins ==="); print(r.groupby("system")[["sharpe", "avgR", "tstat", "maxdd", "R_2025_26", "trades"]].median().round(3))

    print("\n=== variants of W1 regime + D1 entries (BTC / ETH / SOL Sharpe, avg R, maxDD) ===")
    V = {"base: stop below low, exit when no setup A": {}, "half-wick stop": dict(sl_mode="halfwick"), "hold & trail": dict(hold_mode="trail"),
         "setup A only": dict(use_B=False), "setup B only": dict(use_A=False), "long side only": dict(sides="long"), "short side only": dict(sides="short"),
         "1 re-entry at level": dict(max_reentries=1), "no costs": dict(cost=0.0), "10 bps/side": dict(cost=0.001), "stop slips 0.25%": dict(stop_slip=0.0025),
         "entry 1h late": dict(entry_delay=1), "martingale recover cap 4x": dict(martingale=True, mg_mode="recover")}
    data = {c: setup(c) for c in ["BTC", "ETH", "SOL"]}
    out = []
    for nm, kw in V.items():
        row = dict(variant=nm)
        for c, (h, w, first, reg) in data.items():
            tr, cv, hc = run(h, w, first, Cfg(**kw), regime_override=reg, hourly=True); st, _ = daily_stats(tr, hc)
            row.update({f"{c}_sharpe": st["sharpe"], f"{c}_avgR": st["avgR"], f"{c}_maxdd": st["maxdd"], f"{c}_final": st["final"]})
        out.append(row)
    print(pd.DataFrame(out).round(2).to_string(index=False))

    print("\n=== placebo on ETH/BTC/SOL: weekly regime shifted in time (whole weeks), D1 entries unchanged ===")
    for c, (h, w, first, reg) in data.items():
        s0 = w.index.get_loc(first); live = reg[s0:]; real = daily_stats(*run(h, w, first, Cfg(), regime_override=reg, hourly=True)[::2])[0]["sharpe"]
        res = []
        for k in range(70, len(live) - 70, 49):
            ro = reg.copy(); ro[s0:] = np.roll(live, k)
            tr, cv, hc = run(h, w, first, Cfg(), regime_override=ro, hourly=True); res.append(daily_stats(tr, hc)[0]["sharpe"])
        res = np.array(res); print(f"{c}: real Sharpe {real:.2f} | placebo median {np.median(res):.2f}, 95th pct {np.percentile(res, 95):.2f}, share beating real {100 * (res > real).mean():.0f}% (n={len(res)})")

    print("\n=== combo BTC+ETH+SOL, each coin risks 2% of a shared account ===")
    for nm in ["W1 regime + D1 entries", "W1 regime + W1 entries", "D1 regime + D1 entries"]:
        R = pd.concat([curves[(c, nm)].pct_change() for c in ["BTC", "ETH", "SOL"]], axis=1, keys=["BTC", "ETH", "SOL"]).dropna()
        R = R[R.index >= "2021-04-20"]; p = R.sum(axis=1); e = (1 + p).cumprod()
        print(f"{nm:24s} from {R.index[0].date()}: final {e.iloc[-1]:.2f}x  CAGR {100 * (e.iloc[-1] ** (365.25 / len(p)) - 1):.1f}%  maxDD(daily) {100 * (e / e.cummax() - 1).min():.1f}%  Sharpe {p.mean() / p.std() * np.sqrt(365):.2f}  solo {dict((R.mean() / R.std() * np.sqrt(365)).round(2))}  corr {R.corr().values[np.triu_indices(3, 1)].mean():.2f}")
        yr = (1 + p).groupby(p.index.year).prod() - 1; print("   by year:", dict((100 * yr).round(0)))
