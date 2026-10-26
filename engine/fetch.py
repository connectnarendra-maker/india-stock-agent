"""Hourly market data fetch for the Nifty 500 universe.

Runs on GitHub Actions (which can reach NSE and Yahoo Finance).
Writes into ./out/ which the workflow force-pushes to the `data` branch:
  universe.csv        Nifty 500 constituents (symbol, company, industry)
  prices_hist.csv.gz  2y daily adjusted closes, refreshed once per day
  volume_hist.csv.gz  2y daily volumes, refreshed once per day
  latest.csv          last price / change / volume for every symbol, every run
  fundamentals.json   valuation + quality + growth fields, refreshed once per day
  meta.json           timestamps, counts and failures for this run
"""
import io, json, os, sys, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import yfinance as yf

OUT = "out"
PREV = "prev"  # previous data branch checkout (may be empty on first run)
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NOW = dt.datetime.now(IST)
INDICES = {"NIFTY50": "^NSEI", "NIFTY500": "^CRSLDX"}
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}
UNIVERSE_URLS = [
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]
FUND_FIELDS = ["shortName", "sector", "industry", "marketCap", "trailingPE", "forwardPE",
               "priceToBook", "returnOnEquity", "returnOnAssets", "profitMargins",
               "operatingMargins", "debtToEquity", "revenueGrowth", "earningsGrowth",
               "dividendYield", "currentRatio", "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
               "heldPercentInsiders", "heldPercentInstitutions"]

meta = {"run_at_ist": NOW.isoformat(timespec="minutes"), "failures": {}, "notes": []}


def prev_path(name):
    p = os.path.join(PREV, name)
    return p if os.path.exists(p) else None


def age_hours(name):
    p = prev_path("meta.json")
    if not p:
        return 1e9
    m = json.load(open(p))
    ts = m.get("refreshed", {}).get(name)
    if not ts:
        return 1e9
    return (NOW - dt.datetime.fromisoformat(ts)).total_seconds() / 3600


def load_universe():
    for url in UNIVERSE_URLS:
        try:
            r = requests.get(url, headers=UA, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df = df.rename(columns={"Company Name": "company", "Industry": "industry", "Symbol": "symbol"})
            df = df[["symbol", "company", "industry"]].dropna()
            df = df[~df.symbol.str.upper().str.startswith("DUMMY")]
            if len(df) >= 450:  # sanity check on a full list
                meta["universe_source"] = url
                return df
        except Exception as e:
            meta["failures"][f"universe:{url}"] = str(e)[:200]
    p = prev_path("universe.csv")
    if p:
        meta["notes"].append("Universe download failed; reused previous list.")
        return pd.read_csv(p)
    sys.exit("No universe available")


def extra_symbols():
    """Watchlist symbols outside the Nifty 500 (ticker only, nothing else)."""
    p = "signals/watchlist.txt"
    if not os.path.exists(p):
        return []
    return [s.strip().upper() for s in open(p) if s.strip() and not s.startswith("#")]


def download(tickers, period, interval="1d"):
    closes, vols = [], []
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        for attempt in range(3):
            try:
                d = yf.download(chunk, period=period, interval=interval, auto_adjust=True,
                                group_by="column", threads=True, progress=False)
                closes.append(d["Close"])
                vols.append(d["Volume"])
                break
            except Exception as e:
                meta["failures"][f"prices:{i}:{attempt}"] = str(e)[:200]
                time.sleep(5 * (attempt + 1))
    c = pd.concat(closes, axis=1) if closes else pd.DataFrame()
    v = pd.concat(vols, axis=1) if vols else pd.DataFrame()
    return c, v


def fundamentals_for(t):
    for attempt in range(3):
        try:
            info = yf.Ticker(t).info or {}
            return t, {k: info.get(k) for k in FUND_FIELDS}
        except Exception:
            time.sleep(2 * (attempt + 1))
    return t, None


def main():
    os.makedirs(OUT, exist_ok=True)
    refreshed = {}
    uni = load_universe()
    extras = [s for s in extra_symbols() if s not in set(uni.symbol)]
    for s in extras:
        uni.loc[len(uni)] = [s, s, "Watchlist"]
    uni.to_csv(f"{OUT}/universe.csv", index=False)
    yticks = [f"{s}.NS" for s in uni.symbol] + list(INDICES.values())

    # Daily history: refresh after market close or if older than 20h
    after_close = NOW.hour >= 16
    if age_hours("prices_hist") > 20 or (after_close and age_hours("prices_hist") > 3) \
            or not prev_path("prices_hist.csv.gz"):
        c, v = download(yticks, "2y")
        c.to_csv(f"{OUT}/prices_hist.csv.gz")
        v.to_csv(f"{OUT}/volume_hist.csv.gz")
        refreshed["prices_hist"] = NOW.isoformat()
    else:
        for f in ["prices_hist.csv.gz", "volume_hist.csv.gz"]:
            os.replace(prev_path(f), f"{OUT}/{f}")
        refreshed["prices_hist"] = json.load(open(prev_path("meta.json")))["refreshed"]["prices_hist"]

    # Latest snapshot every run (last 5 sessions so today's partial bar is included)
    c5, v5 = download(yticks, "5d")
    c5, v5 = c5.ffill(), v5.fillna(0)
    latest = pd.DataFrame({
        "price": c5.iloc[-1],
        "prev_close": c5.iloc[-2],
        "volume": v5.iloc[-1],
        "bar_date": str(c5.index[-1].date()),
    })
    latest["chg_pct"] = (latest.price / latest.prev_close - 1) * 100
    latest.index.name = "ticker"
    latest.to_csv(f"{OUT}/latest.csv")
    refreshed["latest"] = NOW.isoformat()
    meta["latest_missing"] = int(latest.price.isna().sum())

    # Fundamentals once a day
    if age_hours("fundamentals") > 20 or not prev_path("fundamentals.json"):
        stock_ticks = [f"{s}.NS" for s in uni.symbol]
        with ThreadPoolExecutor(max_workers=8) as ex:
            res = dict(ex.map(fundamentals_for, stock_ticks))
        bad = [t for t, v in res.items() if v is None]
        meta["fundamentals_missing"] = len(bad)
        old = json.load(open(prev_path("fundamentals.json"))) if prev_path("fundamentals.json") else {}
        merged = {t: (v if v is not None else old.get(t)) for t, v in res.items()}
        json.dump(merged, open(f"{OUT}/fundamentals.json", "w"))
        refreshed["fundamentals"] = NOW.isoformat()
    else:
        os.replace(prev_path("fundamentals.json"), f"{OUT}/fundamentals.json")
        refreshed["fundamentals"] = json.load(open(prev_path("meta.json")))["refreshed"]["fundamentals"]

    meta["refreshed"] = refreshed
    meta["universe_count"] = int(len(uni))
    meta["indices"] = INDICES
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
