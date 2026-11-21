"""Hold-the-position variant of the candle strategy, with breakeven stop, take-profit and pyramiding.

Regime and setups are identical to backtest.py (regime from the last CLOSED candle of the timeframe).
Differences:
  * a position ("leg") is HELD while the regime is intact; its stop stays where it was set at entry
    (below the low of the candle before entry) -- it is not trailed and not closed when a week has no setup;
  * breakeven: stop moves to the entry price once the leg is `be_R` x initial risk in profit (touch on a 1h bar,
    active from the next bar) or after `be_bars` closed candles that finished in profit;
  * take-profit: limit at entry * (1 +/- tp_pct);
  * pyramiding: up to `max_legs` legs; a new candle that meets setup A / B opens one more leg with its own stop.
    add_rule = "any"        -> add whenever a new setup appears
               "protected"  -> add only when every open leg already has its stop at breakeven or better
               "profit"     -> add only when every open leg is in profit at that moment
Worst case inside a 1h bar: stop is checked before take-profit; an entry bar that also touches the stop is a loss.
"""
from dataclasses import dataclass
import numpy as np, pandas as pd
from backtest import add_macd


@dataclass
class HCfg:
    regime_line: str = "signal"
    sl_mode: str = "low"
    be_R: float = None          # move stop to entry after this many R of open profit
    be_bars: int = None         # ... or after this many closed candles in profit
    tp_pct: float = None        # take profit, fraction of entry price
    max_legs: int = 1
    add_rule: str = "any"
    risk_frac: float = 0.02     # risk per leg
    max_lev: float = 3.0        # total notional / equity
    cost: float = 0.0005
    sl_buf: float = 0.001
    stop_slip: float = 0.0


def run_hold(h, w, first, cfg: HCfg, regime_override=None):
    w = add_macd(w); line = w[cfg.regime_line].values; widx = w.index
    O, H, L, C = (w[c].values for c in ("open", "high", "low", "close"))
    bars = {k: g[["open", "high", "low", "close", "time"]].values for k, g in h.groupby("week")}
    eq = 1.0; legs = []; trades = []; hc = []; regime = 0

    def close(leg, px, when, reason):
        nonlocal eq
        pnl = leg["dir"] * leg["qty"] * (px - leg["entry"]) - cfg.cost * leg["qty"] * px
        eq += pnl; leg["pnl"] += pnl
        leg.update(exit=px, exit_time=when, reason=reason, R=leg["pnl"] / leg["risk"]); trades.append(leg)

    def open_leg(d, px, sl, when, setup, n):
        nonlocal eq
        dist = abs(px - sl)
        room = cfg.max_lev * eq - sum(l["qty"] * px for l in legs)
        q = min(eq * cfg.risk_frac / dist, max(room, 0) / px)
        if q <= 0: return
        fee = cfg.cost * q * px; eq -= fee
        legs.append(dict(dir=d, setup=setup, leg_no=n, entry=px, entry_time=when, sl=sl, sl0=sl, qty=q, risk=q * dist, pnl=-fee, be=False, bars_held=0,
                         tp=px * (1 + d * cfg.tp_pct) if cfg.tp_pct else None, be_px=px + d * cfg.be_R * dist if cfg.be_R else None))

    for t in range(widx.get_loc(first) + 1, len(w)):
        wk = widx[t]
        if wk not in bars: continue
        b = bars[wk]
        if regime_override is not None: regime = int(regime_override[t])
        elif line[t - 1] > line[t - 2]: regime = 1
        elif line[t - 1] < line[t - 2]: regime = -1
        d = regime; po, ph, pl, pc = O[t - 1], H[t - 1], L[t - 1], C[t - 1]; wopen, t0 = b[0][0], b[0][4]
        # regime flip -> everything out at the open
        for leg in [l for l in legs if l["dir"] != d]:
            close(leg, wopen, t0, "regime_flip"); legs.remove(leg)
        # time-based breakeven (uses the candle that has just closed)
        for leg in legs:
            leg["bars_held"] += 1
            if cfg.be_bars and not leg["be"] and leg["bars_held"] >= cfg.be_bars and (pc - leg["entry"]) * d > 0 and (wopen - leg["entry"]) * d > 0:
                leg["sl"], leg["be"] = leg["entry"], True
        if d == 0: continue
        if d == 1:
            setup = "A" if pc > po else "B"; level = wopen if setup == "A" else po + (ph - po) / 2
            sl = (pl if cfg.sl_mode == "low" else pl + (min(po, pc) - pl) / 2) * (1 - cfg.sl_buf)
        else:
            setup = "A" if pc < po else "B"; level = wopen if setup == "A" else po - (po - pl) / 2
            sl = (ph if cfg.sl_mode == "low" else ph - (ph - max(po, pc)) / 2) * (1 + cfg.sl_buf)

        def may_add(px):
            if len(legs) >= cfg.max_legs: return False
            if not legs or cfg.add_rule == "any": return True
            if cfg.add_rule == "protected": return all(l["be"] for l in legs)
            return all((px - l["entry"]) * d > 0 for l in legs)

        done = False                                      # at most one new leg per candle
        if setup == "A":
            if may_add(wopen) and (wopen - sl) * d > 0: open_leg(d, wopen, sl, t0, "A", len(legs) + 1)
            done = True
        for i in range(len(b)):
            bo, bh, bl, bc, bt = b[i]
            for leg in list(legs):
                if leg["entry_time"] == bt and leg["setup"] == "B": continue          # handled when it was opened
                if (d == 1 and bl <= leg["sl"]) or (d == -1 and bh >= leg["sl"]):
                    px = (min(leg["sl"], bo) if d == 1 else max(leg["sl"], bo)) * (1 - cfg.stop_slip * d)
                    close(leg, px, bt, "breakeven" if leg["be"] else "stop"); legs.remove(leg); continue
                if leg["tp"] and ((d == 1 and bh >= leg["tp"]) or (d == -1 and bl <= leg["tp"])):
                    close(leg, max(leg["tp"], bo) if d == 1 else min(leg["tp"], bo), bt, "take_profit"); legs.remove(leg); continue
                if leg["be_px"] and not leg["be"] and ((d == 1 and bh >= leg["be_px"]) or (d == -1 and bl <= leg["be_px"])):
                    leg["sl"], leg["be"] = leg["entry"], True                        # effective from the next bar
            if not done and ((bh >= level) if d == 1 else (bl <= level)):
                fill = bo if (bo - level) * d >= 0 else level
                done = True
                if may_add(fill) and (fill - sl) * d > 0:
                    open_leg(d, fill, sl, bt, "B", len(legs) + 1)
                    if legs and legs[-1]["entry_time"] == bt and ((d == 1 and bl <= sl) or (d == -1 and bh >= sl)):
                        leg = legs.pop(); close(leg, sl * (1 - cfg.stop_slip * d), bt, "stop")
            m = eq + sum(l["dir"] * l["qty"] * (bc - l["entry"]) for l in legs)
            wst = eq + sum(l["dir"] * l["qty"] * ((bl if l["dir"] == 1 else bh) - l["entry"]) for l in legs)
            hc.append((bt, m, wst, sum(l["qty"] for l in legs) * bc / m if m > 0 else 0.0))
    for leg in legs: close(leg, b[-1][3], b[-1][4], "end")
    return pd.DataFrame(trades), pd.DataFrame(hc, columns=["time", "mark", "worst", "lev"]).set_index("time")


def hstats(tr, hc):
    e = hc.mark.resample("1D").last().ffill(); r = e.pct_change().dropna(); yrs = (e.index[-1] - e.index[0]).days / 365.25
    out = dict(final=e.iloc[-1], cagr=e.iloc[-1] ** (1 / yrs) - 1 if e.iloc[-1] > 0 else -1.0, maxdd=(hc.worst / hc.mark.cummax() - 1).min(),
               sharpe=r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0.0, trades=len(tr), avg_lev=hc.lev.mean(), in_mkt=(hc.lev > 0).mean())
    if len(tr):
        out.update(win=(tr.pnl > 0).mean(), avgR=tr.R.mean(), sumR=tr.R.sum(), tstat=tr.R.mean() / tr.R.std() * np.sqrt(len(tr)) if len(tr) > 2 else np.nan,
                   R_2025_26=tr.R[tr.entry_time.dt.year >= 2025].mean(), hold_days=((tr.exit_time - tr.entry_time).dt.total_seconds() / 86400).mean())
    return out
