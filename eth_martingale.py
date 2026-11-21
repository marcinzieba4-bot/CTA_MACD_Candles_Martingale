"""ETH W1, base rules, 'recover' martingale capped at 4x. Fixed stake: base risk = 1,000 per trade (up to 4,000), no compounding."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from backtest import Cfg, load_data, run
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
UNIT = 1000.0                                   # money risked on a base (1x) trade
RF = 0.02; K = UNIT / RF                        # engine capital 1.0  ->  K in money
h, w, fw = load_data()                          # Bitstamp ETHUSD, Aug 2017 ->
MG = dict(martingale=True, mg_mode="recover", mg_cap=4.0)


def report(name, cfg):
    tr, cv, hc = run(h, w, fw, cfg, hourly=True)
    pnl = (hc.mark - 1) * K; worst = (hc.worst - 1) * K; peak = pnl.cummax()
    dd = worst - peak; e = pnl.resample("1D").last().ffill(); yrs = (e.index[-1] - e.index[0]).days / 365.25
    tr = tr.assign(money=tr.pnl * K, risk=tr.mult * UNIT, notional=tr.mult * UNIT / (abs(tr.entry - tr.sl0) / tr.entry), year=tr.exit_time.dt.year)
    uw = e < e.cummax(); longest = uw.groupby((~uw).cumsum()).sum().max()
    y = pd.DataFrame({"profit": e.groupby(e.index.year).last().diff().fillna(e.groupby(e.index.year).last().iloc[0]),
                      "worst_drawdown_in_year": dd.groupby(dd.index.year).min(), "trades": tr.groupby("year").size(),
                      "win_rate": tr.groupby("year").money.apply(lambda x: (x > 0).mean()), "avg_risk_mult": tr.groupby("year").mult.mean()})
    streak = (tr.money <= 0).astype(int); ls = streak.groupby((streak == 0).cumsum()).sum().max()
    summ = {"period": f"{e.index[0].date()} -> {e.index[-1].date()} ({yrs:.1f} years)", "total profit": pnl.iloc[-1], "average profit per year": pnl.iloc[-1] / yrs,
            "median calendar year": y.profit.iloc[1:-1].median(), "best year": y.profit.max(), "worst year": y.profit.min(), "losing years": int((y.profit < 0).sum()),
            "MAX DRAWDOWN (peak to worst 1h price)": dd.min(), "date of the low": dd.idxmin().date(), "profit per year / max drawdown": pnl.iloc[-1] / yrs / -dd.min(),
            "longest time below a previous high (days)": int(longest), "trades": len(tr), "win rate": (tr.money > 0).mean(), "average trade": tr.money.mean(), "median trade": tr.money.median(),
            "biggest win": tr.money.max(), "biggest loss": tr.money.min(), "longest losing streak (trades)": int(ls), "sum of losses in the worst streak": tr.money.groupby((streak == 0).cumsum()).apply(lambda x: x[x <= 0].sum()).min(),
            "average risk per trade": tr.risk.mean(), "share of trades at 4x": (tr.mult >= 3.999).mean(), "share of trades at 1x": (tr.mult <= 1.001).mean(),
            "average position size (notional)": tr.notional.mean(), "largest position size (notional)": tr.notional.max()}
    print(f"\n=================== {name} ===================")
    for k, v in summ.items(): print(f"{k:45s} {v:,.0f}" if isinstance(v, (float, np.floating)) and abs(v) >= 10 else f"{k:45s} {v:.2f}" if isinstance(v, (float, np.floating)) else f"{k:45s} {v}")
    print("\nby calendar year (money; 2017 and 2026 are partial years):"); print(y.round(2).to_string())
    return pnl, dd, y, summ


print(f"ETH W1 | all figures in money, base risk per trade = {UNIT:,.0f} (martingale raises it up to {4 * UNIT:,.0f}); no compounding; 5 bps per side")
pm, dm, ym, sm = report("MARTINGALE recover, cap 4x", Cfg(fixed_stake=True, max_lev=20, **MG))
pf, df, yf, sf = report("for comparison: FIXED 1x", Cfg(fixed_stake=True, max_lev=20))
ym.to_csv("results/eth_martingale_by_year.csv")

print("\n=================== same thing compounded, as % of an account (base risk 2% of equity, up to 8%) ===================")
for nm, cfg in [("martingale 4x", Cfg(**MG)), ("fixed 2%", Cfg())]:
    tr, cv, hc = run(h, w, fw, cfg, hourly=True); e = hc.mark.resample("1D").last().ffill(); yrs = (e.index[-1] - e.index[0]).days / 365.25
    yr = e.groupby(e.index.year).last().pct_change(); yr.iloc[0] = e.groupby(e.index.year).last().iloc[0] - 1
    print(f"{nm:14s} growth {e.iloc[-1]:.1f}x  per year {e.iloc[-1] ** (1 / yrs) - 1:.1%}  max drawdown {(hc.worst / hc.mark.cummax() - 1).min():.1%}  by year: " + "  ".join(f"{k}: {v:+.0%}" for k, v in yr.items()))

BLUE, ORANGE, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#d9d8d2", "xtick.color": MUTED, "ytick.color": MUTED})
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[2, 1], facecolor=SURF)
for ax in (a, b):
    ax.set_facecolor(SURF); ax.grid(True, color="#ecebe6", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
a.plot(pm.index, pm, color=BLUE, lw=2, label="Martingale recover, cap 4x"); a.plot(pf.index, pf, color=ORANGE, lw=2, label="Fixed 1x")
a.set_title(f"ETH W1: cumulative profit, base risk {UNIT:,.0f} per trade, no compounding", loc="left", color=INK, fontsize=11.5); a.legend(frameon=False, loc="upper left")
a.annotate(f"{pm.iloc[-1] / 1000:,.0f}k", (pm.index[-1], pm.iloc[-1]), xytext=(6, 0), textcoords="offset points", color=INK)
a.annotate(f"{pf.iloc[-1] / 1000:,.0f}k", (pf.index[-1], pf.iloc[-1]), xytext=(6, 0), textcoords="offset points", color=INK)
b.plot(dm.index, dm, color=BLUE, lw=1.5); b.plot(df.index, df, color=ORANGE, lw=1.5)
b.set_title("Drawdown from the previous high (worst 1h price)", loc="left", color=INK, fontsize=10.5)
b.set_xlim(pm.index[0], pm.index[-1] + pd.Timedelta(days=200)); fig.tight_layout(); fig.savefig("results/eth_martingale.png", dpi=200)
