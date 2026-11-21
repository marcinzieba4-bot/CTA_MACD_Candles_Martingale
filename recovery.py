"""W1 only, base rules (stop re-set under each new candle): how fast does equity get back to a new high with and without martingale?"""
import numpy as np, pandas as pd
from backtest import Cfg, run
import mtf
from mtf import build
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
mtf.COINS.update({c: (f"data/oos/{c.lower()}usdt_1h_binance.csv.gz", None) for c in ["BNB", "XRP", "LTC", "ADA", "DOGE"]})


def episodes(e):
    """Underwater episodes of a daily equity series: start (last high), depth, days to trough, days to new high."""
    peak = e.cummax(); uw = e < peak; out = []; grp = (~uw).cumsum()
    for _, s in e[uw].groupby(grp[uw]):
        start = s.index[0] - pd.Timedelta(days=1); pk = peak[s.index[0]]; recovered = s.index[-1] < e.index[-1]
        out.append(dict(start=start.date(), depth=s.min() / pk - 1, to_trough=(s.idxmin() - start).days, climb=(s.index[-1] - s.idxmin()).days + 1,
                        days=(s.index[-1] - start).days + 1, recovered=recovered))
    return pd.DataFrame(out)


def measure(tr, hc):
    e = hc.mark.resample("1D").last().ffill(); ep = episodes(e); r = e.pct_change().dropna(); yrs = len(e) / 365.25
    big = ep[ep.days >= 30]
    # trade-level: after a losing trade that makes a new equity low vs last high, how many trades until a new high of closed equity
    ce = (1 + 0 * tr.pnl).cumsum() * 0 + tr.eq0.iloc[0] + tr.pnl.cumsum(); pk = ce.cummax(); n_uw = []; c = 0
    for u in (ce < pk):
        if u: c += 1
        elif c: n_uw.append(c); c = 0
    if c: n_uw.append(c)
    return dict(final=e.iloc[-1], cagr=e.iloc[-1] ** (1 / yrs) - 1, maxdd=(hc.worst / hc.mark.cummax() - 1).min(), sharpe=r.mean() / r.std() * np.sqrt(365),
                pct_time_underwater=(e < e.cummax()).mean(), longest_days=ep.days.max(), median_days_30plus=big.days.median(), n_30plus=len(big),
                avg_days_per_year_in_30plus=big.days.sum() / yrs, mean_trades_underwater=np.mean(n_uw), max_trades_underwater=np.max(n_uw), avg_mult=tr.mult.mean()), ep


if __name__ == "__main__":
    rows, eps = [], {}
    for coin in mtf.COINS:
        h, w, first = build(coin, "W1")
        tm, _, hm = run(h, w, first, Cfg(martingale=True, mg_mode="recover", mg_cap=4.0), hourly=True)
        V = {"fixed 2%": Cfg(), "martingale recover (cap 4x)": None, "martingale double (max 4x)": Cfg(martingale=True, mg_mode="double", mg_max_steps=2),
             "martingale recover (cap 8x)": Cfg(martingale=True, mg_mode="recover", mg_cap=8.0, max_lev=6.0),
             "fixed, same avg risk as recover 4x": Cfg(risk_frac=0.02 * tm.mult.mean())}
        for nm, cfg in V.items():
            tr, hc = (tm, hm) if cfg is None else run(h, w, first, cfg, hourly=True)[::2]
            m, ep = measure(tr, hc); rows.append(dict(coin=coin, sizing=nm, **m)); eps[(coin, nm)] = ep
    r = pd.DataFrame(rows); r.to_csv("results/recovery.csv", index=False)
    show = ["coin", "sizing", "final", "maxdd", "sharpe", "avg_mult", "pct_time_underwater", "longest_days", "median_days_30plus", "n_30plus", "mean_trades_underwater", "max_trades_underwater"]
    print("=== per coin (W1, base rules, 5 bps/side; underwater = below the previous equity high, daily marks) ===")
    print(r[r.coin.isin(["ETH", "BTC", "SOL"])][show].round(3).to_string(index=False))
    print("\n=== median over 8 coins ===")
    order = ["fixed 2%", "fixed, same avg risk as recover 4x", "martingale recover (cap 4x)", "martingale double (max 4x)", "martingale recover (cap 8x)"]
    print(r.groupby("sizing")[show[2:]].median().loc[order].round(3).to_string())
    print("\n=== on how many of the 8 coins is the martingale FASTER than fixed 2% / than fixed with the same average risk ===")
    P = r.pivot(index="coin", columns="sizing")
    for met, lab in [("longest_days", "longest underwater period"), ("pct_time_underwater", "share of time underwater"), ("median_days_30plus", "median length of 30+ day periods"), ("mean_trades_underwater", "avg trades to a new high")]:
        for mg in order[2:4]:
            print(f"{lab:36s} {mg:28s} faster than fixed 2%: {(P[met][mg] < P[met]['fixed 2%']).sum()}/8   faster than fixed same-risk: {(P[met][mg] < P[met][order[1]]).sum()}/8")
    print("\n=== ETH: the 6 longest underwater periods, same calendar window, per sizing ===")
    for nm in order[:4]:
        print(nm); print(eps[("ETH", nm)].nlargest(6, "days").round(3).to_string(index=False))
    print("\n=== ETH: same drawdowns side by side (periods of fixed 2% lasting 60+ days; when did each sizing get back to its pre-drawdown high?) ===")
    h, w, first = build("ETH", "W1"); curves = {}
    for nm, cfg in [("fixed 2%", Cfg()), ("fixed same risk", Cfg(risk_frac=0.02 * 2.309)), ("recover 4x", Cfg(martingale=True, mg_mode="recover")), ("double 4x", Cfg(martingale=True, mg_mode="double"))]:
        curves[nm] = run(h, w, first, cfg, hourly=True)[2].mark.resample("1D").last().ffill()
    out = []
    for _, ep in eps[("ETH", "fixed 2%")].query("days >= 60").iterrows():
        s = pd.Timestamp(ep.start, tz="UTC"); row = dict(start=ep.start)
        for nm, e in curves.items():
            base = e[:s].max(); after = e[s + pd.Timedelta(days=1):]; rec = after[after >= base]
            row[f"{nm} depth"] = after[: rec.index[0] if len(rec) else None].min() / base - 1
            row[f"{nm} days"] = (rec.index[0] - s).days if len(rec) else np.nan
        out.append(row)
    print(pd.DataFrame(out).round(3).to_string(index=False))
