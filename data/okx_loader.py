import os, time, numpy as np, pandas as pd, requests
from typing import Optional, Literal
from datetime import datetime, timezone

OKX_BASE = "https://www.okx.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "quant_crypto/1.0"})


def _to_ms(dt):
    if isinstance(dt, str):
        dt = pd.Timestamp(dt, tz="UTC")
    elif isinstance(dt, pd.Timestamp) and dt.tz is None:
        dt = dt.tz_localize("UTC")
    elif isinstance(dt, datetime) and dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(pd.Timestamp(dt).timestamp() * 1000)


def _request(path, params, retries=3, backoff=0.5):
    for attempt in range(retries):
        try:
            r = SESSION.get(OKX_BASE + path, params=params, timeout=10)
            r.raise_for_status()
            j = r.json()
            if j.get("code") == "0":
                return j.get("data", [])
            else:
                raise RuntimeError(f"OKX error {j.get('code')}: {j.get('msg')}")
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))


def fetch_candles(inst_id, bar="1D", start="2020-01-01", end=None, verbose=True):
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end else int(time.time() * 1000)
    all_rows = []
    cursor = end_ms
    page = 0
    while True:
        page += 1
        rows = _request(
            "/api/v5/market/history-candles",
            {"instId": inst_id, "bar": bar, "after": cursor, "limit": 100},
        )
        if not rows:
            break
        all_rows.extend(rows)
        oldest_ts = int(rows[-1][0])
        if verbose and page % 5 == 0:
            print(f"  ...page {page}, oldest = {pd.Timestamp(oldest_ts, unit='ms')}")
        if oldest_ts <= start_ms:
            break
        cursor = oldest_ts
        time.sleep(0.10)
    if not all_rows:
        return pd.DataFrame()
    cols = ["ts", "open", "high", "low", "close", "volume", "volCcy", "volCcyQuote", "confirm"]
    df = pd.DataFrame(all_rows, columns=cols)
    df["ts"] = pd.to_datetime(df["ts"].astype(np.int64), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "volCcyQuote"]:
        df[c] = df[c].astype(float)
    df = df.set_index("ts").sort_index()
    df = df[(df.index >= pd.Timestamp(start, tz="UTC")) & (df.index <= pd.Timestamp(end, tz="UTC") if end else True)]
    df = df[["open", "high", "low", "close", "volume", "volCcyQuote"]]
    df = df[~df.index.duplicated(keep="last")]
    if verbose and len(df) > 0:
        print(f"  -> fetched {len(df)} bars [{df.index[0].date()} -> {df.index[-1].date()}]")
    return df


def fetch_funding_rate_history(inst_id="BTC-USDT-SWAP", start="2020-01-01", end=None, verbose=True):
    start_ms = _to_ms(start)
    end_ms = _to_ms(end) if end else int(time.time() * 1000)
    all_rows = []
    cursor = end_ms
    page = 0
    while True:
        page += 1
        rows = _request(
            "/api/v5/public/funding-rate-history",
            {"instId": inst_id, "before": 0, "after": cursor, "limit": 100},
        )
        if not rows:
            break
        all_rows.extend(rows)
        oldest = int(rows[-1]["fundingTime"])
        if verbose and page % 5 == 0:
            print(f"  ...page {page}, oldest funding = {pd.Timestamp(oldest, unit='ms')}")
        if oldest <= start_ms:
            break
        cursor = oldest
        time.sleep(0.10)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    if "fundingTime" not in df.columns:
        return pd.DataFrame()
    df["fundingTime"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    if "realizedRate" in df.columns:
        df["realizedRate"] = pd.to_numeric(df["realizedRate"], errors="coerce")
    else:
        df["realizedRate"] = df["fundingRate"]
    df = df.set_index("fundingTime").sort_index().dropna(subset=["fundingRate"])
    return df[["fundingRate", "realizedRate"]]


def load_btc_eth(start="2020-01-01", end=None, bar="1D", instrument="spot", cache_dir="./data/cache_okx", use_cache=True):
    suffix = "" if instrument == "spot" else "-SWAP"
    pairs = {"BTC": f"BTC-USDT{suffix}", "ETH": f"ETH-USDT{suffix}"}
    os.makedirs(cache_dir, exist_ok=True)
    parts = {}
    for asset, inst in pairs.items():
        cache_path = os.path.join(cache_dir, f"{inst}_{bar}.parquet")
        if use_cache and os.path.exists(cache_path):
            cdf = pd.read_parquet(cache_path)
            if cdf.index[0] <= pd.Timestamp(start, tz="UTC") and (end is None or cdf.index[-1] >= pd.Timestamp(end, tz="UTC")):
                print(f"[okx_loader] using cache for {inst}")
                parts[asset] = cdf["close"].rename(asset)
                continue
        print(f"[okx_loader] fetching {inst} from OKX...")
        cdf = fetch_candles(inst, bar=bar, start=start, end=end)
        if not cdf.empty:
            cdf.to_parquet(cache_path)
            parts[asset] = cdf["close"].rename(asset)
    if len(parts) < 2:
        raise RuntimeError("Failed to fetch both BTC and ETH from OKX.")
    out = pd.concat(parts.values(), axis=1).dropna()
    out.index = out.index.tz_convert(None)
    return out


def load_perp_with_funding(start="2020-01-01", end=None):
    prices = load_btc_eth(start=start, end=end, instrument="swap")
    fb = fetch_funding_rate_history("BTC-USDT-SWAP", start=start, end=end)
    fe = fetch_funding_rate_history("ETH-USDT-SWAP", start=start, end=end)
    fbd = fb["fundingRate"].resample("1D").sum().rename("BTC_funding")
    fed = fe["fundingRate"].resample("1D").sum().rename("ETH_funding")
    funding = pd.concat([fbd, fed], axis=1)
    funding.index = funding.index.tz_convert(None) if funding.index.tz else funding.index
    return {"prices": prices, "funding": funding}


if __name__ == "__main__":
    print("=== Testing OKX loader ===")
    df = load_btc_eth(start="2024-01-01", end="2024-12-31", bar="1D")
    print(df.tail())
    print(f"Days: {len(df)}  BTC vol: {df['BTC'].pct_change().std()*np.sqrt(365):.1%}  ETH vol: {df['ETH'].pct_change().std()*np.sqrt(365):.1%}  Corr: {df.pct_change().corr().iloc[0,1]:.3f}")
