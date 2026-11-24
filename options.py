"""ETH W1 signals traded with options instead of the perp: buy a 5% OTM call on a long signal, a 5% OTM put on a short signal.

No historical option quotes are used -- prices are MODELLED:
  * Black-Scholes, r = 0, USD-settled (Deribit options are coin-settled; ignored).
  * Implied vol = Deribit DVOL (30-day index) of the PREVIOUS day from Mar 2021; before that DVOL is estimated from
    trailing 30-day realised vol with the DVOL/realised relation fitted on 2021+ (flat across strikes and expiries: no skew, no term structure).
  * Costs: half-spread as % of the premium on every buy and sell + Deribit fee min(0.03% of underlying, 12.5% of premium).
  * An option with exactly the wanted expiry and strike is assumed to exist.
Sizing: a fixed premium budget per trade (1,000 = the whole risk). If the option expires while the signal is still on,
it is settled at intrinsic value and a new 5% OTM option is bought with the same budget.
Exit modes: "mirror"  -> close the option whenever the perp strategy would exit (price stop included)
            "no stop" -> the premium is the stop: close only on the weekly rules (no setup A at the weekly open, regime flip)
"""
import sys, itertools
import numpy as np, pandas as pd
from math import erf, sqrt
from backtest import Cfg, load_data, run
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
UNIT = 1000.0
N = np.vectorize(lambda x: 0.5 * (1 + erf(x / sqrt(2))))


def bs(S, K, tau, iv, call):
    S, tau, iv = np.asarray(S, float), np.maximum(np.asarray(tau, float), 1e-9), np.asarray(iv, float)
    d1 = (np.log(S / K) + 0.5 * iv ** 2 * tau) / (iv * np.sqrt(tau)); d2 = d1 - iv * np.sqrt(tau)
    c = S * N(d1) - K * N(d2)
    return c if call else c - S + K


def iv_series(h):
    """Hourly implied vol known at that hour (previous day's DVOL close; modelled from realised vol before DVOL exists)."""
    px = h.set_index("time").close
    rv = np.log(px).diff().rolling(24 * 30).std() * np.sqrt(8760)
    rv_d = rv.resample("1D").last().shift(1)                                  # yesterday's 30d realised vol
    dv = pd.read_csv("data/eth_dvol_1d_deribit.csv", parse_dates=["time"]).set_index("time").close / 100
    dv_d = dv.shift(1)                                                        # yesterday's DVOL close
    j = pd.concat([rv_d.rename("rv"), dv_d.rename("dv")], axis=1).dropna()
    b, a = np.polyfit(j.rv, j.dv, 1)
    iv_d = dv_d.reindex(rv_d.index).fillna(a + b * rv_d).clip(lower=0.25)
    info = dict(a=a, b=b, corr=j.corr().iloc[0, 1], mean_dvol=j.dv.mean(), mean_rv=j.rv.mean(), modelled_until=dv.index[0].date())
    return iv_d.reindex(px.index, method="ffill").bfill(), info


def simulate(trades, h, iv, T_days=30, otm=0.05, spread=0.03, iv_mult=1.0, martingale=False, cap=4.0):
    """Returns per-trade P&L and an hourly cumulative P&L / worst-mark series in money."""
    px = h.set_index("time").close; t_idx = px.index
    cum = np.zeros(len(t_idx)); out = []; pool = 0.0; realised = 0.0
    fee = lambda S, prem: min(0.0003 * S, 0.125 * prem)
    for tr in trades.itertuples():
        call = tr.dir == 1; budget = min(UNIT + pool, cap * UNIT) if martingale else UNIT
        i0, i1 = t_idx.searchsorted(tr.entry_time), t_idx.searchsorted(tr.exit_time)
        S_path = px.values[i0:i1 + 1].copy(); S_path[0] = tr.entry; S_path[-1] = tr.exit        # entry / exit happen at the signal's own prices
        ivp = iv.values[i0:i1 + 1] * iv_mult; n = len(S_path); pnl = 0.0; k = 0; rolls = 0; mark = np.zeros(n)
        while True:
            S0 = S_path[k]; K = S0 * (1 + otm) if call else S0 * (1 - otm)
            prem = float(bs(S0, K, T_days / 365, ivp[k], call)); ask = prem * (1 + spread) + fee(S0, prem)
            qty = budget / ask; pnl -= budget
            kexp = k + int(T_days * 24)
            seg = np.arange(k, min(kexp, n - 1) + 1)
            tau = (T_days * 24 - (seg - k)) / 8760
            val = bs(S_path[seg], K, tau, ivp[seg], call)
            mark[seg] = pnl + qty * val * (1 - spread)                                       # marked at the bid
            if kexp <= n - 1:                                                                # option expires before (or exactly at) the exit
                intrinsic = max(S_path[kexp] - K, 0) if call else max(K - S_path[kexp], 0)
                pnl += qty * intrinsic; mark[kexp] = pnl
                if kexp == n - 1: break
                k = kexp; rolls += 1; continue
            v = float(val[-1]); pnl += qty * (v * (1 - spread) - fee(S_path[-1], v)); mark[-1] = pnl
            break
        cum[i0:i1 + 1] = realised + mark; realised += pnl; cum[i1 + 1:] = realised
        if martingale: pool = max(0.0, pool - pnl)
        out.append(dict(entry_time=tr.entry_time, exit_time=tr.exit_time, dir=tr.dir, budget=budget, pnl=pnl, rolls=rolls, days=(tr.exit_time - tr.entry_time).total_seconds() / 86400))
    return pd.DataFrame(out), pd.Series(cum, index=t_idx)


def summary(tr, cum):
    e = cum.resample("1D").last().ffill(); yrs = (e.index[-1] - e.index[0]).days / 365.25; dd = cum - cum.cummax()
    uw = e < e.cummax(); y = e.groupby(e.index.year).last(); y = y.diff().fillna(y.iloc[0])
    return dict(per_year=cum.iloc[-1] / yrs, total=cum.iloc[-1], maxdd=dd.min(), ratio=cum.iloc[-1] / yrs / -dd.min(), losing_years=int((y < 0).sum()), worst_year=y.min(), best_year=y.max(),
                longest_underwater_days=int(uw.groupby((~uw).cumsum()).sum().max()), trades=len(tr), win=(tr.pnl > 0).mean(), avg_trade=tr.pnl.mean(), median_trade=tr.pnl.median(),
                biggest_win=tr.pnl.max(), premium_spent_per_year=(tr.budget * (1 + tr.rolls)).sum() / yrs, rolls=int(tr.rolls.sum())), y


if __name__ == "__main__":
    h, w, fw = load_data()
    iv, info = iv_series(h)
    print("IV model:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()})
    T = {"mirror": run(h, w, fw, Cfg(fixed_stake=True, max_lev=20))[0], "no stop": run(h, w, fw, Cfg(fixed_stake=True, max_lev=20, no_stop=True))[0]}
    print(f"signal trades: mirror {len(T['mirror'])}, no stop {len(T['no stop'])} | average holding {T['mirror'].eval('(exit_time - entry_time).dt.total_seconds()').mean() / 86400:.1f} / {T['no stop'].eval('(exit_time - entry_time).dt.total_seconds()').mean() / 86400:.1f} days")
    print("PERP reference, base risk 1,000 per trade: fixed 11,057 per year, max drawdown -11,724 | martingale 4x 23,673 per year, max drawdown -19,435")

    print("\n=== 1. 5% OTM, premium 1,000 per trade (money, no compounding; half-spread 3% of premium per side) ===")
    rows = []; keep = {}
    for mode, Td, mg in itertools.product(T, [7, 14, 30, 60, 90], [False, True]):
        tr, cum = simulate(T[mode], h, iv, T_days=Td, martingale=mg); s, y = summary(tr, cum)
        rows.append(dict(exit=mode, expiry_days=Td, sizing="martingale 4x" if mg else "fixed", **s)); keep[(mode, Td, mg)] = (tr, cum, y)
    g = pd.DataFrame(rows); g.to_csv("results/options_grid.csv", index=False); print(g.round(2).to_string(index=False))

    print("\n=== 2. by calendar year (fixed 1,000 premium per trade) ===")
    print(pd.DataFrame({f"{m}, {d}d": keep[(m, d, False)][2] for m in T for d in [7, 30, 90]}).round(0).to_string())
    print("\nby calendar year, martingale 4x:")
    print(pd.DataFrame({f"{m}, {d}d": keep[(m, d, True)][2] for m in T for d in [7, 30, 90]}).round(0).to_string())

    print("\n=== 3. sensitivity (no stop, 30-day expiry, fixed premium): profit per year / max drawdown ===")
    for nm, kw in [("base", {}), ("no spread", dict(spread=0.0)), ("spread 8%", dict(spread=0.08)), ("IV +15%", dict(iv_mult=1.15)), ("IV +30%", dict(iv_mult=1.30)), ("IV -15%", dict(iv_mult=0.85)),
                   ("at the money", dict(otm=0.0)), ("10% OTM", dict(otm=0.10)), ("20% OTM", dict(otm=0.20)), ("5% in the money", dict(otm=-0.05))]:
        for mode in T:
            s, _ = summary(*simulate(T[mode], h, iv, T_days=30, **kw))
            print(f"{nm:18s} {mode:8s} per year {s['per_year']:>9,.0f}   max drawdown {s['maxdd']:>9,.0f}   ratio {s['ratio']:.2f}   win {s['win']:.0%}")

    print("\n=== 4. long calls only / short puts only (no stop, 30-day, fixed premium) ===")
    for nm, d in [("calls (long signals)", 1), ("puts (short signals)", -1)]:
        for mode in T:
            sub = T[mode][T[mode].dir == d]; s, _ = summary(*simulate(sub, h, iv, T_days=30))
            print(f"{nm:22s} {mode:8s} trades {s['trades']:3d}  per year {s['per_year']:>8,.0f}  max drawdown {s['maxdd']:>9,.0f}  win {s['win']:.0%}  biggest win {s['biggest_win']:>8,.0f}")

    print("\n=== 5. modelled vs real implied vol: results since DVOL exists (Apr 2021 ->), no stop, fixed premium ===")
    for Td in [7, 30, 90]:
        tr, cum = keep[("no stop", Td, False)][:2]; c = cum[cum.index >= "2021-04-01"]; c = c - c.iloc[0]; yrs = (c.index[-1] - c.index[0]).days / 365.25
        print(f"{Td:3d}d: per year {c.iloc[-1] / yrs:>8,.0f}   max drawdown {(c - c.cummax()).min():>9,.0f}   (whole history: {keep[('no stop', Td, False)][1].iloc[-1] / 9.2:>8,.0f} per year)")

    print("\n=== 6. sensitivity of the 7-day version (mirror, fixed premium) ===")
    for nm, kw in [("base", {}), ("spread 8%", dict(spread=0.08)), ("spread 15%", dict(spread=0.15)), ("IV +15%", dict(iv_mult=1.15)), ("IV +30%", dict(iv_mult=1.30)), ("IV +15% and spread 8%", dict(iv_mult=1.15, spread=0.08))]:
        tr, cum = simulate(T["mirror"], h, iv, T_days=7, **kw); s, _ = summary(tr, cum)
        top = tr.pnl.nlargest(5).sum()
        print(f"{nm:24s} per year {s['per_year']:>9,.0f}   max drawdown {s['maxdd']:>9,.0f}   ratio {s['ratio']:.2f}   win {s['win']:.0%}   5 best trades = {top / s['total']:.0%} of profit")
