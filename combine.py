"""Combine two ETH strategies on one 10,000 account (flat sizing, no compounding):

A  MACD regime + candle entries on W1, traded with the PERP, 'recover' martingale capped at 4x.
   Base risk 200 per trade (2% of the starting account), up to 800.
B  Weekly momentum with 5% OTM options (from the repo GridOfSellCallsETH, study 5): every Friday 08:00 UTC,
   last week up -> buy a weekly call 5% OTM, down -> buy a put 5% OTM, hold to expiry. Premium 200 per week.
   Priced from real Deribit trades. Its weekly returns on premium are in data/weekly_momentum_otm_options_R.csv.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from backtest import Cfg, load_data, run
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
ACC, UNIT = 10_000.0, 200.0

# ---- B: option leg (P&L lands at expiry, one week after purchase)
rb = pd.read_csv("data/weekly_momentum_otm_options_R.csv", parse_dates=["t"], index_col="t")["R_0.050"]
b_pnl = (UNIT * rb); b_pnl.index = b_pnl.index + pd.Timedelta(days=7)
fridays = pd.date_range(rb.index[0], b_pnl.index[-1], freq="7D")                 # Fri 08:00 UTC grid
B = b_pnl.reindex(fridays).fillna(0).cumsum()

# ---- A: perp leg, hourly marks
h, w, fw = load_data()
def perp(martingale):
    kw = dict(martingale=True, mg_mode="recover", mg_cap=4.0) if martingale else {}
    tr, cv, hc = run(h, w, fw, Cfg(fixed_stake=True, max_lev=20, **kw), hourly=True)
    K = UNIT / 0.02
    m = ((hc.mark - 1) * K); wst = ((hc.worst - 1) * K)
    m, wst = m[m.index >= fridays[0]], wst[wst.index >= fridays[0]]
    return m - m.iloc[0], wst - m.iloc[0], tr
Am, Aw, trA = perp(True)
Af, _, _ = perp(False)
A = Am.reindex(fridays, method="ffill"); AF = Af.reindex(fridays, method="ffill")

# hourly worst case: option marked at zero for the whole week it is open
Bh = B.reindex(Am.index, method="ffill"); open_prem = pd.Series(UNIT, index=rb.index).reindex(Am.index, method="ffill").fillna(0)
open_prem[Am.index >= b_pnl.index[-1]] = 0


def stats(cum, worst=None, name=""):
    eq = ACC + cum; pnl = cum.diff().dropna(); yrs = (cum.index[-1] - cum.index[0]).days / 365.25
    dd = eq - eq.cummax(); ddp = eq / eq.cummax() - 1; uw = dd < 0
    y = cum.groupby(cum.index.year).last().diff(); y.iloc[0] = cum.groupby(cum.index.year).last().iloc[0]
    r12 = cum.diff(52).dropna(); cagr = (eq.iloc[-1] / ACC) ** (1 / yrs) - 1
    s = {"profit per year": cum.iloc[-1] / yrs, "total profit": cum.iloc[-1], "max drawdown (weekly marks)": dd.min(), "max drawdown % of peak equity": 100 * ddp.min(),
         "date of the low": dd.idxmin().date(), "profit per year / max drawdown": cum.iloc[-1] / yrs / -dd.min(), "Calmar (CAGR / max DD %)": cagr / -ddp.min(),
         "Sharpe (weekly, annualised)": pnl.mean() / pnl.std() * np.sqrt(52),
         "weekly volatility": pnl.std(), "best week": pnl.max(), "worst week": pnl.min(), "5% worst week (VaR 95)": pnl.quantile(0.05), "avg of 5% worst weeks": pnl[pnl <= pnl.quantile(0.05)].mean(),
         "winning weeks %": 100 * (pnl > 0).mean(), "worst 12 months": r12.min(), "12-month windows with a loss %": 100 * (r12 < 0).mean(),
         "losing calendar years": int((y < 0).sum()), "worst calendar year": y.min(), "longest time below a high (weeks)": int(uw.groupby((~uw).cumsum()).sum().max()),
         "share of profit from the 5 best weeks %": 100 * pnl.nlargest(5).sum() / cum.iloc[-1]}
    if worst is not None:
        s["max drawdown (worst hourly price, options marked at zero)"] = (worst - cum.reindex(worst.index, method="ffill").cummax()).min()
    return s, y

legs = {"A: MACD perp, martingale 4x": (A, Aw), "B: weekly 5% OTM options": (B, Bh - open_prem), "A + B combined": (A + B, Aw + Bh - open_prem),
        "half A + half B": ((A + B) / 2, (Aw + Bh - open_prem) / 2), "A fixed size (no martingale) + B": (AF + B, None)}
S, Y = {}, {}
for k, (c, wc) in legs.items(): S[k], Y[k] = stats(c, wc, k)
tab = pd.DataFrame(S)
fmt = lambda v: v if not isinstance(v, (float, np.floating)) else (f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}")
print(f"Account {ACC:,.0f}, flat sizing, no compounding | {fridays[0].date()} -> {fridays[-1].date()} ({(fridays[-1] - fridays[0]).days / 365.25:.1f} years), weekly marks Friday 08:00 UTC")
print(tab.map(fmt).to_string())
print("\nby calendar year:"); print(pd.DataFrame(Y).round(0).to_string())
pa, pb = A.diff().dropna(), B.diff().dropna()
print(f"\ncorrelation of weekly P&L, A vs B: {pa.corr(pb):.2f} | 13-week P&L: {A.diff(13).corr(B.diff(13)):.2f} | weeks both lose: {100 * ((pa < 0) & (pb < 0)).mean():.0f}% | both win: {100 * ((pa > 0) & (pb > 0)).mean():.0f}%")
ddA, ddB = (A - A.cummax()), (B - B.cummax())
print(f"A drawdown when B was at its worst ({ddB.idxmin().date()}): {ddA[ddB.idxmin()]:,.0f} | B drawdown when A was at its worst ({ddA.idxmin().date()}): {ddB[ddA.idxmin()]:,.0f}")
print(f"sum of the two separate max drawdowns: {ddA.min() + ddB.min():,.0f} vs combined max drawdown: {((A + B) - (A + B).cummax()).min():,.0f}")
tab.to_csv("results/combined_stats.csv"); pd.DataFrame({"A": A, "B": B, "combined": A + B}).to_csv("results/combined_weekly.csv")

BLUE, ORANGE, AQUA, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#d9d8d2", "xtick.color": MUTED, "ytick.color": MUTED})
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True, height_ratios=[2, 1], facecolor=SURF)
for ax in (a, b):
    ax.set_facecolor(SURF); ax.grid(True, color="#ecebe6", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:,.0f}k"))
C = A + B
for s, col, lab in [(C, AQUA, "A + B combined"), (A, BLUE, "A: MACD perp, martingale 4x"), (B, ORANGE, "B: weekly 5% OTM options")]:
    a.plot(s.index, s, color=col, lw=2.4 if col == AQUA else 1.8, label=lab)
    a.annotate(f"{s.iloc[-1] / 1000:,.0f}k", (s.index[-1], s.iloc[-1]), xytext=(6, -3), textcoords="offset points", color=INK)
    b.plot(s.index, s - s.cummax(), color=col, lw=2 if col == AQUA else 1.4)
a.set_title("ETH: cumulative profit on a 10,000 account, flat sizing (200 base risk per trade / 200 premium per week)", loc="left", color=INK, fontsize=11); a.legend(frameon=False, loc="upper left")
b.set_title("Drawdown from the previous high (weekly marks)", loc="left", color=INK, fontsize=10.5)
b.set_xlim(C.index[0], C.index[-1] + pd.Timedelta(days=170)); fig.tight_layout(); fig.savefig("results/combined.png", dpi=200)
