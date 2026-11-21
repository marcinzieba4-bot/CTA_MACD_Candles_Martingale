"""W1 ETHUSD -- MACD regime + previous-candle entries + re-entries (optional martingale).

Rules (long side; shorts are the exact mirror)
----------------------------------------------
Regime   : slope of the slow MACD line on W1, read on the last CLOSED week
           (line[t-1] > line[t-2] -> UP -> longs only; falling -> shorts only).
           regime_line = "signal" (EMA9 of MACD, the slow line) or "macd".
Setup A  : previous weekly candle closed up  -> buy the open of the new week.
Setup B  : previous weekly candle closed down -> buy-stop at
           PrevOpen + (PrevHigh - PrevOpen) / 2  (middle of the upper wick).
Stop     : sl_mode = "low"      -> just below the previous candle's low
           sl_mode = "halfwick" -> middle of the previous candle's lower wick
Re-entry : after a stop-out, a new buy-stop sits at the week's entry level
           (week open for A, trigger for B) -- price has to RECLAIM the level.
           Same stop, at most `max_reentries` per week.
Exit     : stop-loss, regime flip (exit at next weekly open), or
           hold_mode = "weekly": position is closed at the weekly open when the new
                                 week has no A setup (it may come back through B);
           hold_mode = "trail" : position is kept, stop moves under the new prev candle.
Sizing   : "risk"     -> risk `risk_frac` of equity between entry and stop (leverage capped)
           "notional" -> 1x equity notional on every entry
Martingale (optional): risk multiplier x`mg_mult` after each losing trade, capped at
           `mg_max_steps` doublings, reset after a winning trade.

Execution is simulated on 1h bars so the order of entry / stop inside the week is known.
If a 1h bar touches both the entry trigger and the stop, the stop is assumed hit (worst case).
"""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- data
def load_data():
    h = pd.read_csv("data/ethusd_1h_bitstamp.csv.gz", parse_dates=["time"])
    d = pd.read_csv("data/ethusd_1d_coinbase_warmup.csv", parse_dates=["time"])

    def weekly(df):
        wk = (df.time - pd.to_timedelta(df.time.dt.weekday, unit="D")).dt.normalize()  # Monday 00:00 UTC
        return df.groupby(wk).agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                  close=("close", "last"), n=("close", "size"))

    wh, wd = weekly(h), weekly(d)
    first_full = wh.index[wh.n >= 168][0]
    w = pd.concat([wd[wd.index < first_full], wh[wh.index >= first_full]])
    if w.n.iloc[-1] < 168:           # drop the still-running week
        w = w.iloc[:-1]
    h["week"] = (h.time - pd.to_timedelta(h.time.dt.weekday, unit="D")).dt.normalize()
    return h, w.drop(columns="n"), first_full


def add_macd(w, fast=12, slow=26, sig=9):
    w = w.copy()
    ema = lambda s, n: s.ewm(span=n, adjust=False).mean()
    w["macd"] = ema(w.close, fast) - ema(w.close, slow)
    w["signal"] = ema(w.macd, sig)
    return w


# ----------------------------------------------------------------------------- config
@dataclass
class Cfg:
    regime_line: str = "signal"   # "signal" | "macd"
    sl_mode: str = "low"          # "low" | "halfwick"
    hold_mode: str = "weekly"     # "weekly" | "trail"
    max_reentries: int = 0
    reentry_mode: str = "level"   # "level": reclaim of the week's entry level, original stop
                                  # "spring": reclaim of the broken stop level, stop under the low made after the stop-out
    invert_regime: bool = False   # ablation only: trade against the MACD slope
    sizing: str = "risk"          # "risk" | "notional"
    risk_frac: float = 0.02
    max_lev: float = 3.0
    martingale: bool = False
    mg_mult: float = 2.0
    mg_max_steps: int = 2
    cost: float = 0.001           # fee + slippage per side, fraction of notional
    sl_buf: float = 0.001         # stop placed 0.1% beyond the level
    sides: str = "both"           # "both" | "long" | "short"
    use_A: bool = True
    use_B: bool = True


# ----------------------------------------------------------------------------- engine
def run(h, w, first_week, cfg: Cfg):
    w = add_macd(w)
    line = w[cfg.regime_line].values
    widx = w.index
    O, H, L, C = (w[c].values for c in ("open", "high", "low", "close"))
    bars = {k: g[["open", "high", "low", "close", "time"]].values for k, g in h.groupby("week")}

    eq = 1.0
    pos = 0; qty = 0.0; entry = 0.0; risk_usd = 0.0; tr = None
    mg_step = 0
    regime = 0
    trades, curve = [], []

    def close_pos(price, when, reason):
        nonlocal eq, pos, qty, tr, mg_step
        pnl = pos * qty * (price - entry) - cfg.cost * qty * price
        eq += pnl
        tr["pnl"] += pnl                      # entry cost was already booked into tr["pnl"]
        tr.update(exit=price, exit_time=when, reason=reason, R=tr["pnl"] / risk_usd, ret=tr["pnl"] / tr["eq0"])
        trades.append(tr)
        if cfg.martingale:
            mg_step = 0 if tr["pnl"] > 0 else min(mg_step + 1, cfg.mg_max_steps)
        pos = 0; qty = 0.0; tr = None

    def open_pos(d, price, sl, when, setup, kind):
        nonlocal eq, pos, qty, entry, risk_usd, tr
        dist = abs(price - sl)
        mult = cfg.mg_mult ** mg_step if cfg.martingale else 1.0
        if cfg.sizing == "risk":
            q = min(eq * cfg.risk_frac * mult / dist, cfg.max_lev * eq / price)
        else:
            q = min(mult, cfg.max_lev) * eq / price
        fee = cfg.cost * q * price
        tr = dict(dir=d, setup=setup, kind=kind, entry=price, entry_time=when, sl0=sl, eq0=eq, mult=mult, pnl=-fee)
        eq -= fee
        pos, qty, entry, risk_usd = d, q, price, q * dist

    start = widx.get_loc(first_week) + 1
    for t in range(start, len(w)):
        wk = widx[t]
        if wk not in bars:
            continue
        b = bars[wk]
        # --- regime from closed weeks only
        if line[t - 1] > line[t - 2]: regime = 1
        elif line[t - 1] < line[t - 2]: regime = -1
        d = -regime if cfg.invert_regime else regime
        if (d == 1 and cfg.sides == "short") or (d == -1 and cfg.sides == "long"):
            d = 0
        po, ph, pl, pc = O[t - 1], H[t - 1], L[t - 1], C[t - 1]
        wopen, t0 = b[0][0], b[0][4]

        setup = level = sl = None
        if d == 1:
            setup = "A" if pc > po else "B"
            level = wopen if setup == "A" else po + (ph - po) / 2
            sl = pl if cfg.sl_mode == "low" else pl + (min(po, pc) - pl) / 2
            sl *= 1 - cfg.sl_buf
        elif d == -1:
            setup = "A" if pc < po else "B"
            level = wopen if setup == "A" else po - (po - pl) / 2
            sl = ph if cfg.sl_mode == "low" else ph - (ph - max(po, pc)) / 2
            sl *= 1 + cfg.sl_buf
        allowed = d != 0 and ((setup == "A" and cfg.use_A) or (setup == "B" and cfg.use_B))

        # --- weekly open: manage the carried position
        initial_done = False
        if pos != 0:
            if pos != d:
                close_pos(wopen, t0, "regime_flip")
            elif setup == "B" and cfg.hold_mode == "weekly":
                close_pos(wopen, t0, "no_setup")
            elif (wopen - sl) * pos <= 0:      # sl is already this week's level
                close_pos(wopen, t0, "stop_gap"); initial_done = True
            else:
                initial_done = True           # carried on, stop moved to this week's level
        if pos == 0 and allowed and setup == "A" and not initial_done:
            if (wopen - sl) * d > 0:
                open_pos(d, wopen, sl, t0, "A", "initial")
            initial_done = True
        re_left = cfg.max_reentries
        wsl = sl                              # this week's stop level (a "spring" re-entry uses its own stop)
        ref = sl / (1 - cfg.sl_buf * d) if d else None   # un-buffered stop level
        ext = None                            # extreme made since the stop-out

        # --- walk the week hour by hour
        for i in range(len(b)):
            bo, bh, bl, bc, bt = b[i]
            if pos != 0:
                if (pos == 1 and bl <= sl) or (pos == -1 and bh >= sl):
                    px = min(sl, bo) if pos == 1 else max(sl, bo)
                    close_pos(px, bt, "stop")
                    ext = bl if d == 1 else bh
                continue                      # no new order in the bar that stopped us
            if not allowed or (initial_done and re_left <= 0):
                continue
            if i == 0 and setup == "A":
                continue                      # open of the week already handled
            trig, sl = level, wsl
            if initial_done and cfg.reentry_mode == "spring":
                if ext is None:               # the position was not stopped inside this week
                    continue
                trig, sl = ref, ext * (1 - cfg.sl_buf * d)     # stop under the low made BEFORE this bar
                ext = min(ext, bl) if d == 1 else max(ext, bh)
            hit = bh >= trig if d == 1 else bl <= trig
            if not hit:
                continue
            gap = (bo - trig) * d >= 0
            fill = bo if gap else trig
            if (fill - sl) * d <= 0:
                continue
            kind = "reentry" if initial_done else "initial"
            if initial_done: re_left -= 1
            initial_done = True
            open_pos(d, fill, sl, bt, setup, kind)
            if (d == 1 and bl <= sl) or (d == -1 and bh >= sl):   # same bar also touched the stop -> worst case
                close_pos(sl, bt, "stop")
                ext = bl if d == 1 else bh

        curve.append((wk, eq + (pos * qty * (b[-1][3] - entry) if pos else 0.0), C[t], regime))

    if pos != 0:
        close_pos(b[-1][3], b[-1][4], "end")
    cv = pd.DataFrame(curve, columns=["week", "equity", "close", "regime"]).set_index("week")
    return pd.DataFrame(trades), cv


# ----------------------------------------------------------------------------- stats
def stats(tr, cv):
    yrs = (cv.index[-1] - cv.index[0]).days / 365.25
    r = cv.equity.pct_change().dropna()
    dd = cv.equity / cv.equity.cummax() - 1
    out = dict(final=cv.equity.iloc[-1], cagr=cv.equity.iloc[-1] ** (1 / yrs) - 1 if cv.equity.iloc[-1] > 0 else -1,
               maxdd=dd.min(), sharpe=r.mean() / r.std() * np.sqrt(52) if r.std() > 0 else 0, trades=len(tr))
    if len(tr):
        win = tr.pnl > 0
        out.update(win=win.mean(), avgR=tr.R.mean(), pf=tr.pnl[win].sum() / -tr.pnl[~win].sum() if (~win).any() else np.inf,
                   long_pnl=tr.pnl[tr.dir == 1].sum(), short_pnl=tr.pnl[tr.dir == -1].sum())
    out["mar"] = out["cagr"] / -out["maxdd"] if out["maxdd"] < 0 else np.nan
    return out


if __name__ == "__main__":
    h, w, fw = load_data()
    tr, cv = run(h, w, fw, Cfg())
    print(pd.Series(stats(tr, cv)).round(3))
    print(tr.tail(8).to_string())
