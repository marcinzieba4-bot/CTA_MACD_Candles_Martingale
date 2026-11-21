# W1 ETHUSD — MACD regime + previous-candle entries

Backtest of a weekly ETHUSD strategy: the slow MACD line gives the direction, the previous
weekly candle gives the entry and the stop. Simulated on 1h Bitstamp bars, Aug 2017 → Nov 2026.

![equity](results/equity_base.png)

## Rules (long side — shorts are the exact mirror)

| | |
|---|---|
| **Regime** | W1 MACD(12,26,9) **signal line** rising on the last closed week → longs only; falling → shorts only |
| **Setup A** | previous weekly candle closed up → buy the open of the new week |
| **Setup B** | previous candle closed down → buy-stop at `PrevOpen + (PrevHigh − PrevOpen) / 2` |
| **Stop** | 0.1% below the previous candle's low (alternative: middle of its lower wick) |
| **Roll** | every new week the stop moves under the new previous candle; if the new week has no setup A the position is closed at the open (it can come back through B) |
| **Regime flip** | close at the weekly open |
| **Re-entries** | none inside the week — the next weekly setup *is* the re-entry (see below) |
| **Sizing** | risk 2% of equity between entry and stop, leverage capped at 3x |
| **Costs** | 0.1% per side (fee + slippage); funding / borrow not modelled |

Weeks start Monday 00:00 UTC. If a 1h bar touches both an entry trigger and the stop, the stop is assumed hit.

## Results (base rules)

| | Strategy | ETH buy & hold |
|---|---|---|
| Growth of 1 | 6.0x | 5.3x |
| CAGR | 21.4% | 19.8% |
| Max drawdown | −11.6% | −93.8% |
| Trades / win rate / avg R | 153 / 41% / +0.65R | — |

- All 48 grid variants (regime line × stop × hold mode × re-entries × martingale) end positive — `results/grid.csv`.
- Inverting the regime (trading against the MACD slope) kills the edge: 1.2x, avg +0.09R. The filter matters.
- Longs earn about 3x what shorts earn; setups A and B contribute about equally.
- Fat-tailed: the 5 best trades are 57% of total R; longest losing streak 7.
- 2025–2026 were flat (0% and +5%).

## Re-entries

| Design | Result |
|---|---|
| Reclaim of the week's entry level, same stop | break-even (22 trades, −0.3R in total); long side +5.8R, short side −6.1R |
| "Spring": reclaim of the broken stop level, stop under the new low | loses money and triples the drawdown |

Neither adds anything, so the base uses none.

## Martingale

Doubling risk after a loss (max 4x) lifts the result to 31x but the drawdown to −28%. Plain 5% fixed risk gives
47x with −26% — more return for less drawdown, so martingale is dominated by simply risking more.

## Files

- `fetch_data.py` — downloads data (Bitstamp 1h; Coinbase 1d only to warm up the MACD)
- `backtest.py` — engine and `Cfg` with every switch
- `run_grid.py` — grid, re-entry designs, robustness → `results/`
- `plot.py` — chart

```bash
python fetch_data.py && python run_grid.py && python plot.py
```

Backtest only: one asset, ~9 years, 153 trades. Not financial advice.
