"""Critical audit of the base strategy: lookahead, execution realism, statistics, placebo, second venue."""
import numpy as np, pandas as pd
from backtest import Cfg, load_data, run, stats, add_macd

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
h, w, fw = load_data()
base = Cfg()
tr, cv, hc = run(h, w, fw, base, hourly=True)
S = stats(tr, cv)
show = lambda s: {k: round(float(s[k]), 3) for k in ["final", "cagr", "maxdd", "sharpe", "trades", "win", "avgR", "pf"]}
print("BASE @5bps:", show(S))

# ---------------------------------------------------------------- 1. causality
print("\n=== 1. CAUSALITY: re-run on data truncated at T; every trade closed before T must be identical ===")
key = ["dir", "setup", "entry", "entry_time", "sl0", "exit", "exit_time", "pnl"]
for T in ["2019-03-11", "2020-10-05", "2022-06-13", "2024-02-19", "2025-09-01"]:
    T = pd.Timestamp(T, tz="UTC")
    h2, w2, fw2 = load_data(cutoff=T)
    t2, _ = run(h2, w2, fw2, base)
    a = tr[tr.exit_time < T][key].reset_index(drop=True); b = t2[(t2.exit_time < T) & (t2.reason != "end")][key].reset_index(drop=True)  # "end" = forced close of the still-open trade at the cut
    # the decision taken AT T (regime/setup/level/stop of week T) from truncated data vs full data
    wm, wm2 = add_macd(w), add_macd(w2)
    same_sig = np.allclose(wm.signal[wm.index < T].values, wm2.signal.values)
    print(T.date(), "closed trades:", len(a), "identical:", a.equals(b), "| weekly bars & signal line before T identical:", same_sig and w[w.index < T].equals(w2))
wm = add_macd(w)
print("\nwhat the regime of a week is read from (last 6 weeks):")
x = pd.DataFrame({"signal[t-2]": wm.signal.shift(2), "signal[t-1]": wm.signal.shift(1), "signal[t] (NOT used, unknown at the open)": wm.signal}).tail(6).round(2)
x["regime_used"] = np.where(x["signal[t-1]"] > x["signal[t-2]"], "UP", "DOWN"); print(x)

# ---------------------------------------------------------------- 2. exposure / intraweek drawdown
print("\n=== 2. EXPOSURE & DRAWDOWN measured on 1h bars ===")
qty_notional = (0.02 * tr["mult"] / (abs(tr.entry - tr.sl0) / tr.entry)).clip(upper=base.max_lev)
print("leverage at entry (notional/equity): mean %.2f  median %.2f  max %.2f  | share of trades at the 3x cap: %.0f%%" % (qty_notional.mean(), qty_notional.median(), qty_notional.max(), 100 * (qty_notional >= base.max_lev - 1e-9).mean()))
hrs = ((tr.exit_time - tr.entry_time).dt.total_seconds() / 3600).sum(); tot = (hc.index[-1] - hc.index[0]).total_seconds() / 3600
print("time in market: %.0f%%" % (100 * hrs / tot), "| avg holding: %.1f days" % (hrs / len(tr) / 24))
print("max drawdown  weekly closes: %.1f%%   1h closes: %.1f%%   1h worst-price marks: %.1f%%" % (100 * S["maxdd"], 100 * (hc.mark / hc.mark.cummax() - 1).min(), 100 * (hc.worst / hc.mark.cummax() - 1).min()))
under = (cv.equity < cv.equity.cummax()); run_len = (under.groupby((~under).cumsum()).cumsum()).max()
print("longest stretch below a previous equity high: %d weeks" % run_len)

# ---------------------------------------------------------------- 3. execution stress
print("\n=== 3. EXECUTION STRESS ===")
tests = {"base (5 bps/side)": {}, "10 bps/side": dict(cost=0.001), "30 bps/side": dict(cost=0.003),
         "A-entry 1h late": dict(entry_delay=1), "A-entry 4h late": dict(entry_delay=4), "A-entry 24h late": dict(entry_delay=24),
         "stop slips 0.25%": dict(stop_slip=0.0025), "stop slips 0.5%": dict(stop_slip=0.005), "stop slips 1%": dict(stop_slip=0.01),
         "long carry 10%/yr": dict(funding_apr=0.10), "ALL: 10bps + 4h late + 0.5% stop slip + 10% carry": dict(cost=0.001, entry_delay=4, stop_slip=0.005, funding_apr=0.10)}
for k, v in tests.items():
    print(f"{k:52s}", show(stats(*run(h, w, fw, Cfg(**v)))))

# ---------------------------------------------------------------- 4. statistics
print("\n=== 4. STATISTICS OF THE EDGE ===")
R = tr.R.values; rng = np.random.default_rng(0)
bs = rng.choice(R, (20000, len(R))).mean(1)
print("mean R %.2f  median R %.2f  std %.2f  t-stat %.2f  bootstrap 95%% CI of mean R [%.2f, %.2f]" % (R.mean(), np.median(R), R.std(ddof=1), R.mean() / R.std(ddof=1) * np.sqrt(len(R)), *np.percentile(bs, [2.5, 97.5])))
srt = np.sort(R)[::-1]
for k in (1, 3, 5, 10):
    print("  without the %2d best trades: mean R %.2f, total R %.1f (of %.1f)" % (k, srt[k:].mean(), srt[k:].sum(), R.sum()))
tr["year"] = tr.entry_time.dt.year
print("mean R by period:", {p: round(tr.R[(tr.year >= a) & (tr.year <= b)].mean(), 2) for p, a, b in [("2017-19", 2017, 2019), ("2020-21", 2020, 2021), ("2022-23", 2022, 2023), ("2024", 2024, 2024), ("2025-26", 2025, 2026)]})
wr = cv.equity.pct_change().dropna()
print("weekly returns: skew %.1f, best week %+.1f%%, worst %+.1f%%, share of weeks > 0: %.0f%%" % (wr.skew(), 100 * wr.max(), 100 * wr.min(), 100 * (wr > 0).mean()))

# ---------------------------------------------------------------- 5. placebo regimes
print("\n=== 5. PLACEBO: same candle rules, regime series shifted in time (destroys the MACD timing, keeps its persistence) ===")
reg = np.sign(wm.signal.shift(1) - wm.signal.shift(2)).fillna(0).values
s0 = w.index.get_loc(fw) + 1; live = reg[s0:]
res = []
for k in range(8, len(live) - 8, 3):
    ro = reg.copy(); ro[s0:] = np.roll(live, k)
    st = stats(*run(h, w, fw, base, regime_override=ro)); res.append((st["final"], st["avgR"], st["sharpe"]))
res = np.array(res)
print("placebo runs: %d | final equity median %.2fx, 5-95%% [%.2f, %.2f] | avg R median %.2f | Sharpe median %.2f" % (len(res), np.median(res[:, 0]), *np.percentile(res[:, 0], [5, 95]), np.median(res[:, 1]), np.median(res[:, 2])))
print("real strategy %.2fx / %.2fR / Sharpe %.2f  ->  share of placebos that beat it: equity %.1f%%, avgR %.1f%%, Sharpe %.1f%%" % (S["final"], S["avgR"], S["sharpe"], *(100 * (res > np.array([S["final"], S["avgR"], S["sharpe"]])).mean(0))))
for nm, ro in [("always long", np.ones(len(reg))), ("always short", -np.ones(len(reg)))]:
    print(f"candle rules with regime = {nm}:", show(stats(*run(h, w, fw, base, regime_override=ro))))

# ---------------------------------------------------------------- 6. fair benchmarks
print("\n=== 6. BENCHMARKS (weekly closes, same period) ===")
px = cv.close; r = px.pct_change().dropna()
print("ETH buy&hold: Sharpe %.2f, CAGR %.1f%%, maxDD %.1f%%" % (r.mean() / r.std() * np.sqrt(52), 100 * ((px.iloc[-1] / px.iloc[0]) ** (52 / len(r)) - 1), 100 * (px / px.cummax() - 1).min()))
k = wr.std() / r.std(); e = (1 + k * r).cumprod()
print("ETH buy&hold de-levered to the strategy's volatility (%.0f%% of equity in ETH): final %.2fx, CAGR %.1f%%, maxDD %.1f%%" % (100 * k, e.iloc[-1], 100 * (e.iloc[-1] ** (52 / len(r)) - 1), 100 * (e / e.cummax() - 1).min()))
pr = (cv.regime.values[1:] * r.values); e2 = pd.Series((1 + k * pr)).cumprod()
print("MACD regime only, no candle rules (long/short at weekly open, same vol scaling): Sharpe %.2f, final %.2fx, maxDD %.1f%%" % (pr.mean() / pr.std() * np.sqrt(52), e2.iloc[-1], 100 * (e2 / e2.cummax() - 1).min()))
print("strategy: Sharpe %.2f, final %.2fx, maxDD %.1f%%" % (S["sharpe"], S["final"], 100 * S["maxdd"]))

# ---------------------------------------------------------------- 7. second venue
print("\n=== 7. SECOND VENUE: Binance ETHUSDT 1h (independent wicks / stop hits) ===")
hb, wb, fwb = load_data("data/ethusdt_1h_binance.csv.gz")
tb, cb = run(hb, wb, fwb, base); print("Binance :", show(stats(tb, cb)), cb.index[0].date(), "->", cb.index[-1].date())
print("Bitstamp:", show(S), cv.index[0].date(), "->", cv.index[-1].date())
j = cv.equity.pct_change().to_frame("bs").join(cb.equity.pct_change().rename("bn")).dropna()
print("correlation of weekly strategy returns across venues: %.2f" % j.corr().iloc[0, 1])

# ---------------------------------------------------------------- 8. other coins, rules untouched
print("\n=== 8. OUT-OF-SAMPLE: identical rules on other coins (Binance 1h, MACD warmed up on the first 35 weeks, then traded) ===")
import os
from backtest import weekly_from_hourly
rows = []
for sym in ["ETHUSDT", "BTCUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT"]:
    f = "data/ethusdt_1h_binance.csv.gz" if sym == "ETHUSDT" else f"data/oos/{sym.lower()}_1h_binance.csv.gz"
    if not os.path.exists(f):
        continue
    ho = pd.read_csv(f, parse_dates=["time"]); ho, wo = weekly_from_hourly(ho)
    t_, c_ = run(ho, wo, wo.index[35], base); s_ = stats(t_, c_)
    p = c_.close; rr = p.pct_change().dropna()
    rows.append(dict(coin=sym, start=c_.index[0].date(), **show(s_), tstat=round(t_.R.mean() / t_.R.std() * np.sqrt(len(t_)), 2),
                     R_2025_26=round(t_.R[t_.entry_time.dt.year >= 2025].mean(), 2), hold_x=round(p.iloc[-1] / p.iloc[0], 2), hold_sharpe=round(rr.mean() / rr.std() * np.sqrt(52), 2)))
print(pd.DataFrame(rows).to_string(index=False))
