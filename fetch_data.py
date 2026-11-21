"""Download ETHUSD history.

- Bitstamp 1h candles (full history, from Aug 2017)  -> data/ethusd_1h_bitstamp.csv.gz
  used for the bar-by-bar simulation (entries / stops inside the week).
- Coinbase 1d candles (May 2016 -> start of Bitstamp)  -> data/ethusd_1d_coinbase_warmup.csv
  used only to warm up the weekly MACD before the simulation starts.
"""
import json, time, urllib.request
from datetime import datetime, timezone
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (research-backtest)"}


def get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:  # network hiccup -> back off and retry
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def bitstamp_1h(start_ts=1502928000):  # 2017-08-17
    rows, ts, now = [], start_ts, int(time.time())
    while ts < now:
        d = get(f"https://www.bitstamp.net/api/v2/ohlc/ethusd/?step=3600&limit=1000&start={ts}")["data"]["ohlc"]
        if not d:
            ts += 1000 * 3600
            continue
        rows += d
        last = int(d[-1]["timestamp"])
        if last + 3600 <= ts:
            break
        ts = last + 3600
        time.sleep(0.15)
    df = pd.DataFrame(rows).astype({"timestamp": "int64", "open": float, "high": float, "low": float, "close": float, "volume": float})
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    df["time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df[["time", "open", "high", "low", "close", "volume"]]


def coinbase_1d(start="2016-05-18", end="2017-09-01", product="ETH-USD"):
    rows = []
    for s, e in zip(pd.date_range(start, end, freq="250D"), list(pd.date_range(start, end, freq="250D")[1:]) + [pd.Timestamp(end)]):
        rows += get(f"https://api.exchange.coinbase.com/products/{product}/candles?granularity=86400&start={s:%Y-%m-%d}T00:00:00Z&end={e:%Y-%m-%d}T00:00:00Z")
        time.sleep(0.3)
    df = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"]).drop_duplicates("ts").sort_values("ts")
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df[["time", "open", "high", "low", "close", "volume"]]


if __name__ == "__main__":
    h = bitstamp_1h()
    h.to_csv("data/ethusd_1h_bitstamp.csv.gz", index=False)
    print("bitstamp 1h:", len(h), h.time.iloc[0], "->", h.time.iloc[-1])
    d = coinbase_1d()
    d.to_csv("data/ethusd_1d_coinbase_warmup.csv", index=False)
    print("coinbase 1d:", len(d), d.time.iloc[0], "->", d.time.iloc[-1])


def binance_1h(start_ms=1502928000000, symbol="ETHUSDT"):
    """Second, independent venue (ETHUSDT) used only as a cross-check of the Bitstamp results."""
    rows, ts = [], start_ms
    while True:
        d = get(f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&limit=1000&startTime={ts}")
        if not d:
            break
        rows += d
        ts = d[-1][0] + 3600_000
        if len(d) < 1000:
            break
        time.sleep(0.1)
    df = pd.DataFrame([r[:6] for r in rows], columns=["ts", "open", "high", "low", "close", "volume"]).astype(float)
    df["time"] = pd.to_datetime(df.ts.astype("int64"), unit="ms", utc=True)
    return df.drop_duplicates("time").sort_values("time")[["time", "open", "high", "low", "close", "volume"]]
