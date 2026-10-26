"""Multi-factor scoring of the Nifty 500 universe.

Reads the fetch outputs (default ./out) plus signals/news_sentiment.json,
writes scores.json and market.json next to them.

Factors (each z-scored within sector when the sector has >= 8 names,
otherwise across the whole universe; winsorised at 2.5/97.5 percentiles):
  momentum   12-1 month return (60%) + 6-1 month return (40%)
  trend      distance from 200-day average, plus 50 > 200 day crossover
  quality    ROE (when reported), operating and net margins, low debt/equity
  value      earnings yield (1/PE), book yield (1/PB)
  growth     revenue growth, earnings growth
  low_vol    negative 1-year annualised volatility
  sentiment  news sentiment from the agent, decayed with a 3-day half-life
A missing factor scores 0 (neutral) and lowers the stock's data coverage figure.
"""
import json, os, sys, datetime as dt
import numpy as np
import pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else "out"
SIGNALS = sys.argv[2] if len(sys.argv) > 2 else "signals"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NOW = dt.datetime.now(IST)

WEIGHTS = {"momentum": 0.20, "trend": 0.10, "quality": 0.20, "value": 0.15,
           "growth": 0.15, "low_vol": 0.10, "sentiment": 0.10}
LABELS = {"momentum": "Momentum", "trend": "Trend", "quality": "Quality", "value": "Value",
          "growth": "Growth", "low_vol": "Low volatility", "sentiment": "News sentiment"}
MIN_TRADED_VALUE_CR = 2.0  # 60-day median daily traded value floor, in ₹ crore


def _np(o):
    return o.item() if hasattr(o, "item") else str(o)


def winsor(s, lo=0.025, hi=0.975):
    if s.notna().sum() < 10:
        return s
    return s.clip(s.quantile(lo), s.quantile(hi))


def zscore(s, groups):
    s = winsor(s)
    out = pd.Series(np.nan, index=s.index)
    for g, idx in groups.groupby(groups).groups.items():
        sub = s.loc[idx]
        if sub.notna().sum() >= 8:
            out.loc[idx] = (sub - sub.mean()) / (sub.std() or 1)
    rest = out.isna() & s.notna()
    if rest.any():
        out[rest] = (s[rest] - s.mean()) / (s.std() or 1)
    return out.clip(-3, 3)


def combine(*parts):
    df = pd.concat(parts, axis=1)
    return df.mean(axis=1, skipna=True)


def load():
    px = pd.read_csv(f"{D}/prices_hist.csv.gz", index_col=0, parse_dates=True)
    vol = pd.read_csv(f"{D}/volume_hist.csv.gz", index_col=0, parse_dates=True)
    latest = pd.read_csv(f"{D}/latest.csv", index_col=0)
    fund = json.load(open(f"{D}/fundamentals.json"))
    uni = pd.read_csv(f"{D}/universe.csv")
    meta = json.load(open(f"{D}/meta.json"))
    # Splice today's live price onto the daily history
    bar = pd.Timestamp(str(latest.bar_date.dropna().iloc[0])) if latest.bar_date.notna().any() else None
    if bar is not None:
        row = latest.price.reindex(px.columns)
        px.loc[bar] = row.combine_first(px.loc[bar] if bar in px.index else row)
        px = px.sort_index()
    return px.ffill(limit=5), vol, latest, fund, uni, meta


def sentiment_series(symbols):
    p = os.path.join(SIGNALS, "news_sentiment.json")
    out = pd.Series(np.nan, index=symbols)
    if not os.path.exists(p):
        return out, {}
    data = json.load(open(p))
    for sym, items in data.items():
        if sym not in out.index or not items:
            continue
        num = den = 0.0
        for it in items:
            age_d = max(0.0, (NOW - dt.datetime.fromisoformat(it["asof"])).total_seconds() / 86400)
            w = 0.5 ** (age_d / 3) * float(it.get("importance", 1))
            num += w * float(it["score"])
            den += w
        if den > 0.05:
            out[sym] = num / den
    return out, data


def main():
    px, vol, latest, fund, uni, meta = load()
    uni = uni.set_index("symbol")
    syms = [s for s in uni.index if f"{s}.NS" in px.columns]
    P = px[[f"{s}.NS" for s in syms]]
    P.columns = syms
    V = vol.reindex(columns=[f"{s}.NS" for s in syms])
    V.columns = syms
    F = pd.DataFrame({s: fund.get(f"{s}.NS") or {} for s in syms}).T.reindex(syms)
    sector = uni.loc[syms, "industry"].fillna("Other")

    rets = P.pct_change(fill_method=None)
    last = P.iloc[-1]

    def ret(a, b):
        if len(P) <= a:
            return pd.Series(np.nan, index=syms)
        return P.iloc[-1 - b] / P.iloc[-1 - a] - 1

    r12_1, r6_1, r1m, r3m, r1y = ret(252, 21), ret(126, 21), ret(21, 0), ret(63, 0), ret(252, 0)
    ma50, ma200 = P.rolling(50).mean().iloc[-1], P.rolling(200).mean().iloc[-1]
    vol1y = rets.iloc[-252:].std() * np.sqrt(252)
    hi52, lo52 = P.iloc[-252:].max(), P.iloc[-252:].min()
    tv_cr = (P * V).iloc[-61:-1].median() / 1e7

    def f(col):
        return pd.to_numeric(F[col], errors="coerce") if col in F else pd.Series(np.nan, index=syms)

    pe, pb = f("trailingPE"), f("priceToBook")
    ey = (1 / pe).where(pe > 0)
    by = (1 / pb).where(pb > 0)
    de = f("debtToEquity")
    sent, sent_raw = sentiment_series(syms)

    factors = pd.DataFrame({
        "momentum": (0.6 * zscore(r12_1, sector)).add(0.4 * zscore(r6_1, sector), fill_value=np.nan)
                    .fillna(zscore(r6_1, sector)),
        "trend": combine(zscore(last / ma200 - 1, sector),
                         zscore((ma50 > ma200).astype(float).where(ma200.notna()), sector)),
        "quality": combine(zscore(f("returnOnEquity"), sector), zscore(f("operatingMargins"), sector),
                           zscore(f("profitMargins"), sector), zscore(-de, sector)),
        "value": combine(zscore(ey, sector), zscore(by, sector)),
        "growth": combine(zscore(f("revenueGrowth"), sector), zscore(f("earningsGrowth"), sector)),
        "low_vol": zscore(-vol1y, sector),
        "sentiment": sent.clip(-1, 1) * 2,  # ±1 sentiment maps to ±2 sd
    })
    coverage = factors.notna().mul(pd.Series(WEIGHTS)).sum(axis=1)
    composite = factors.fillna(0).mul(pd.Series(WEIGHTS)).sum(axis=1)
    liquid = tv_cr >= MIN_TRADED_VALUE_CR
    pct = composite.rank(pct=True) * 100

    lat = latest.reindex([f"{s}.NS" for s in syms])
    lat.index = syms
    vol_ratio = lat.volume / V.iloc[-21:-1].mean()

    rows = []
    for s in syms:
        fs = factors.loc[s]
        contrib = (fs.fillna(0) * pd.Series(WEIGHTS)).sort_values()
        flags = []
        if pd.notna(lat.chg_pct[s]) and abs(lat.chg_pct[s]) >= 5:
            flags.append(f"{'Up' if lat.chg_pct[s] > 0 else 'Down'} {abs(lat.chg_pct[s]):.1f}% today")
        if pd.notna(vol_ratio[s]) and vol_ratio[s] >= 3:
            flags.append(f"Volume {vol_ratio[s]:.1f}× 20-day average")
        if pd.notna(hi52[s]) and last[s] >= hi52[s] * 0.995:
            flags.append("At 52-week high")
        if pd.notna(lo52[s]) and last[s] <= lo52[s] * 1.005:
            flags.append("At 52-week low")
        if not liquid.get(s, False):
            flags.append("Low liquidity")

        def num(x, nd=2):
            return None if pd.isna(x) else round(float(x), nd)

        rows.append({
            "symbol": s, "company": uni.loc[s, "company"], "sector": sector[s],
            "score": num(composite[s], 3), "percentile": num(pct[s], 1),
            "coverage": num(coverage[s], 2), "liquid": bool(liquid.get(s, False)),
            "factors": {k: num(fs[k], 2) for k in WEIGHTS},
            "top_strengths": [LABELS[k] for k, v in contrib[::-1].items() if v > 0.05][:2],
            "top_weaknesses": [LABELS[k] for k, v in contrib.items() if v < -0.05][:2],
            "price": num(last[s]), "chg_pct": num(lat.chg_pct[s]),
            "ret_1m": num(r1m[s] * 100, 1), "ret_3m": num(r3m[s] * 100, 1), "ret_1y": num(r1y[s] * 100, 1),
            "pe": num(pe[s], 1), "pb": num(pb[s], 1),
            "roe": num(f("returnOnEquity")[s] * 100, 1), "de": num(de[s], 0),
            "rev_growth": num(f("revenueGrowth")[s] * 100, 1),
            "mcap_cr": num(f("marketCap")[s] / 1e7, 0), "vol_1y": num(vol1y[s] * 100, 1),
            "traded_value_cr": num(tv_cr[s], 1), "flags": flags,
        })
    rows.sort(key=lambda r: -(r["score"] or -99))
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    # Market overview
    idx = {}
    for name, t in meta["indices"].items():
        if t in px.columns:
            s = px[t].dropna()
            idx[name] = {"level": round(float(s.iloc[-1]), 2),
                         "chg_pct": round(float(latest.chg_pct.get(t, np.nan)), 2),
                         "ret_1m": round(float(s.iloc[-1] / s.iloc[-22] - 1) * 100, 2),
                         "ret_1y": round(float(s.iloc[-1] / s.iloc[-253] - 1) * 100, 2) if len(s) > 253 else None}
    secperf = pd.DataFrame({"sector": sector, "chg": lat.chg_pct, "r1m": r1m * 100,
                            "score": composite}).groupby("sector").agg(
        chg=("chg", "median"), r1m=("r1m", "median"), score=("score", "mean"), n=("chg", "size"))
    market = {
        "asof_ist": meta["run_at_ist"],
        "price_bar_date": str(latest.bar_date.dropna().iloc[0]),
        "indices": idx,
        "breadth": {"above_200dma_pct": round(float((last > ma200).mean() * 100), 1),
                    "advancers": int((lat.chg_pct > 0).sum()), "decliners": int((lat.chg_pct < 0).sum())},
        "sectors": [{"sector": k, **{c: (None if pd.isna(v) else round(float(v), 2)) for c, v in r.items()}}
                    for k, r in secperf.sort_values("score", ascending=False).iterrows()],
        "top_gainers": [r["symbol"] for r in sorted([r for r in rows if r["chg_pct"] is not None],
                                                    key=lambda r: -r["chg_pct"])[:10]],
        "top_losers": [r["symbol"] for r in sorted([r for r in rows if r["chg_pct"] is not None],
                                                   key=lambda r: r["chg_pct"])[:10]],
        "weights": WEIGHTS, "data_meta": meta,
    }
    json.dump(rows, open(f"{D}/scores.json", "w"), separators=(",", ":"), default=_np)
    json.dump(market, open(f"{D}/market.json", "w"), indent=1, default=_np)
    print(f"Scored {len(rows)} stocks. Top 5:", [r["symbol"] for r in rows[:5]])


if __name__ == "__main__":
    main()
