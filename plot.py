"""Equity curve of the base strategy vs ETH buy & hold (both indexed to 1, log scale) + drawdowns."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd

cv = pd.read_csv("results/equity_base.csv", parse_dates=["week"]).set_index("week")
bh = cv.close / cv.close.iloc[0]
BLUE, ORANGE, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#d9d8d2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, height_ratios=[2.2, 1], facecolor=SURF)
for ax in (a, b):
    ax.set_facecolor(SURF); ax.grid(True, color="#ecebe6", lw=0.8); ax.spines[["top", "right"]].set_visible(False)
a.plot(cv.index, cv.equity, color=BLUE, lw=2, label="Strategy (2% risk / trade)")
a.plot(bh.index, bh, color=ORANGE, lw=2, label="ETH buy & hold")
a.set_yscale("log"); a.set_yticks([0.25, 0.5, 1, 2, 5, 10, 15]); a.set_yticklabels(["0.25x", "0.5x", "1x", "2x", "5x", "10x", "15x"])
a.set_title("W1 ETHUSD: MACD-signal regime + previous-candle entries vs buy & hold  (growth of 1, log scale)", loc="left", color=INK, fontsize=11.5)
a.annotate(f"Strategy {cv.equity.iloc[-1]:.1f}x", (cv.index[-1], cv.equity.iloc[-1]), xytext=(6, 4), textcoords="offset points", color=INK)
a.annotate(f"Buy & hold {bh.iloc[-1]:.1f}x", (bh.index[-1], bh.iloc[-1]), xytext=(6, -10), textcoords="offset points", color=INK)
a.legend(frameon=False, loc="upper left")
for s, c in ((cv.equity, BLUE), (bh, ORANGE)):
    b.plot(s.index, (s / s.cummax() - 1) * 100, color=c, lw=2)
b.set_title("Drawdown from peak, %", loc="left", color=INK, fontsize=10.5)
b.set_xlim(cv.index[0], cv.index[-1] + pd.Timedelta(days=330))
fig.tight_layout(); fig.savefig("results/equity_base.png", dpi=200)
