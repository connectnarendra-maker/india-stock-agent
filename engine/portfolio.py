"""Portfolio analytics against Nifty 50 and Nifty 500.

Run by the agent session, never by GitHub Actions: holdings are passed in as a
local JSON file and are never committed to the repository.

  python engine/portfolio.py <data_dir> <holdings.json> <out.json> [risk_free_pct]

holdings.json: [{"symbol": "INFY", "qty": 10, "avg_price": 1450.5, "buy_date": "2023-04-12"}, ...]
buy_date is optional (used for XIRR; when missing, XIRR is skipped for that lot).

Risk metrics use the CURRENT holdings weights applied to the last 252 trading
days of prices (a "what if I had held today's portfolio for a year" view), which
is the standard way to measure the risk you are carrying now.
"""
import json, sys, datetime as dt
import numpy as np
import pandas as pd
from scipy.optimize import brentq

D, HOLD, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
RF = float(sys.argv[4]) / 100 if len(sys.argv) > 4 else 0.055
TD = 252


def xirr(flows):
    if len(flows) < 2 or not any(a < 0 for _, a in flows) or not any(a > 0 for _, a in flows):
        return None
    t0 = min(d for d, _ in flows)
    f = lambda r: sum(a / (1 + r) ** ((d - t0).days / 365.0) for d, a in flows)
    try:
        return brentq(f, -0.99, 10)
    except ValueError:
        return None


def max_dd(series):
    wealth = (1 + series).cumprod()
    dd = wealth / wealth.cummax() - 1
    trough = dd.idxmin()
    peak = wealth.loc[:trough].idxmax()
    return float(dd.min()), str(peak.date()), str(trough.date())


def stats(p, b, rf):
    """Return, risk and benchmark-relative stats for daily return series p vs b."""
    p, b = p.align(b, join="inner")
    rfd = (1 + rf) ** (1 / TD) - 1
    ann_r = (1 + p).prod() ** (TD / len(p)) - 1
    ann_b = (1 + b).prod() ** (TD / len(b)) - 1
    vol = p.std() * np.sqrt(TD)
    downside = p[p < rfd] - rfd
    dvol = np.sqrt((downside ** 2).sum() / len(p)) * np.sqrt(TD)
    beta = np.cov(p, b)[0, 1] / b.var()
    alpha = ann_r - (rf + beta * (ann_b - rf))
    active = p - b
    te = active.std() * np.sqrt(TD)
    mdd, peak, trough = max_dd(p)
    up, dn = b > 0, b < 0
    var95 = -np.percentile(p, 5)
    cvar95 = -p[p <= -var95].mean()
    return {
        "ann_return_pct": round(ann_r * 100, 2), "benchmark_ann_return_pct": round(ann_b * 100, 2),
        "excess_return_pct": round((ann_r - ann_b) * 100, 2),
        "volatility_pct": round(vol * 100, 2),
        "sharpe": round((ann_r - rf) / vol, 2) if vol else None,
        "sortino": round((ann_r - rf) / dvol, 2) if dvol else None,
        "beta": round(beta, 2), "alpha_pct": round(alpha * 100, 2),
        "correlation": round(float(p.corr(b)), 2),
        "r_squared": round(float(p.corr(b) ** 2), 2),
        "tracking_error_pct": round(te * 100, 2),
        "information_ratio": round((ann_r - ann_b) / te, 2) if te else None,
        "max_drawdown_pct": round(mdd * 100, 2), "max_drawdown_window": [peak, trough],
        "calmar": round(ann_r / abs(mdd), 2) if mdd else None,
        "upside_capture_pct": round(p[up].mean() / b[up].mean() * 100, 1),
        "downside_capture_pct": round(p[dn].mean() / b[dn].mean() * 100, 1),
        "var95_1d_pct": round(var95 * 100, 2), "cvar95_1d_pct": round(cvar95 * 100, 2),
    }


def main():
    hold = json.load(open(HOLD))
    px = pd.read_csv(f"{D}/prices_hist.csv.gz", index_col=0, parse_dates=True)
    latest = pd.read_csv(f"{D}/latest.csv", index_col=0)
    scores = {r["symbol"]: r for r in json.load(open(f"{D}/scores.json"))}
    meta = json.load(open(f"{D}/meta.json"))
    uni = pd.read_csv(f"{D}/universe.csv").set_index("symbol")
    fund = json.load(open(f"{D}/fundamentals.json"))
    today = dt.date.today()

    # Aggregate lots per symbol
    lots = pd.DataFrame(hold)
    lots["symbol"] = lots.symbol.str.upper().str.replace(".NS", "", regex=False)
    lots["cost"] = lots.qty * lots.avg_price
    pos = lots.groupby("symbol").agg(qty=("qty", "sum"), cost=("cost", "sum"))
    pos["avg_price"] = pos.cost / pos.qty
    missing = [s for s in pos.index if f"{s}.NS" not in latest.index or pd.isna(latest.price.get(f"{s}.NS"))]
    pos = pos.drop(index=missing)
    pos["price"] = [latest.price[f"{s}.NS"] for s in pos.index]
    pos["prev_close"] = [latest.prev_close[f"{s}.NS"] for s in pos.index]
    pos["value"] = pos.qty * pos.price
    pos["pnl"] = pos.value - pos.cost
    pos["day_pnl"] = pos.qty * (pos.price - pos.prev_close)
    total = pos.value.sum()
    pos["weight"] = pos.value / total

    # Return series with current weights
    cols = [f"{s}.NS" for s in pos.index]
    R = px[cols].ffill(limit=5).pct_change(fill_method=None).iloc[-TD:]
    R.columns = pos.index
    R = R.dropna(axis=1, thresh=int(TD * 0.6)).fillna(0)
    w = pos.weight.reindex(R.columns).fillna(0)
    w = w / w.sum()
    port = R @ w
    bench = {name: px[t].pct_change(fill_method=None).iloc[-TD:].dropna()
             for name, t in meta["indices"].items() if t in px.columns}

    # Risk contribution (share of portfolio variance)
    cov = R.cov() * TD
    pvar = float(w @ cov @ w)
    rc = (w * (cov @ w)) / pvar
    corr = R.corr().values
    avg_corr = float(corr[np.triu_indices_from(corr, 1)].mean()) if len(w) > 1 else None

    # Sector exposure vs Nifty 500 (approximated by full market-cap weights)
    sector_of = lambda s: uni.industry.get(s, "Other") if s in uni.index else "Other"
    pos["sector"] = [sector_of(s) for s in pos.index]
    mc = pd.Series({s: (fund.get(f"{s}.NS") or {}).get("marketCap") for s in uni.index}, dtype=float)
    idx_sec = (mc.groupby(uni.industry.reindex(mc.index)).sum() / mc.sum() * 100)
    my_sec = pos.groupby("sector").weight.sum() * 100
    sectors = pd.DataFrame({"portfolio_pct": my_sec, "nifty500_pct": idx_sec}).fillna(0)
    sectors["active_pct"] = sectors.portfolio_pct - sectors.nifty500_pct
    sectors = sectors[(sectors.portfolio_pct > 0) | (sectors.nifty500_pct > 1)].sort_values("portfolio_pct", ascending=False)

    # XIRR on lots that carry a buy date, valued at today's price
    flows = []
    for _, l in lots.iterrows():
        if l.symbol in pos.index and isinstance(l.get("buy_date"), str):
            flows.append((dt.date.fromisoformat(l.buy_date), -l.qty * l.avg_price))
            flows.append((today, l.qty * pos.price[l.symbol]))
    xr = xirr(flows)

    hhi = float((pos.weight ** 2).sum())
    wscore = sum(pos.weight[s] * (scores.get(s, {}).get("score") or 0) for s in pos.index)
    wpct = sum(pos.weight[s] * (scores.get(s, {}).get("percentile") or 50) for s in pos.index)

    out = {
        "asof_ist": meta["run_at_ist"], "risk_free_pct": RF * 100,
        "summary": {
            "value": round(total, 2), "cost": round(pos.cost.sum(), 2),
            "pnl": round(pos.pnl.sum(), 2), "pnl_pct": round(pos.pnl.sum() / pos.cost.sum() * 100, 2),
            "day_pnl": round(pos.day_pnl.sum(), 2),
            "day_pnl_pct": round(pos.day_pnl.sum() / (total - pos.day_pnl.sum()) * 100, 2),
            "xirr_pct": round(xr * 100, 2) if xr is not None else None,
            "holdings": int(len(pos)), "effective_n": round(1 / hhi, 1), "hhi": round(hhi, 3),
            "top5_weight_pct": round(pos.weight.nlargest(5).sum() * 100, 1),
            "avg_pairwise_corr": round(avg_corr, 2) if avg_corr is not None else None,
            "weighted_score": round(wscore, 3), "weighted_percentile": round(wpct, 1),
            "unpriced_symbols": missing,
        },
        "vs": {name: stats(port, b, RF) for name, b in bench.items()},
        "holdings": [{
            "symbol": s, "company": uni.company.get(s, s) if s in uni.index else s, "sector": pos.sector[s],
            "qty": float(pos.qty[s]), "avg_price": round(pos.avg_price[s], 2), "price": round(pos.price[s], 2),
            "value": round(pos.value[s], 2), "weight_pct": round(pos.weight[s] * 100, 2),
            "pnl": round(pos.pnl[s], 2), "pnl_pct": round(pos.pnl[s] / pos.cost[s] * 100, 2),
            "day_chg_pct": round((pos.price[s] / pos.prev_close[s] - 1) * 100, 2),
            "risk_contrib_pct": round(float(rc.get(s, np.nan)) * 100, 1) if s in rc else None,
            "score": scores.get(s, {}).get("score"), "percentile": scores.get(s, {}).get("percentile"),
            "rank": scores.get(s, {}).get("rank"), "flags": scores.get(s, {}).get("flags", []),
            "strengths": scores.get(s, {}).get("top_strengths", []),
            "weaknesses": scores.get(s, {}).get("top_weaknesses", []),
        } for s in pos.sort_values("value", ascending=False).index],
        "sectors": [{"sector": k, **{c: round(float(v), 2) for c, v in r.items()}} for k, r in sectors.iterrows()],
        "equity_curve": [{"date": str(d.date()), "portfolio": round(float(v), 4)}
                         for d, v in (1 + port).cumprod().iloc[::5].items()],
        "benchmark_curves": {n: [{"date": str(d.date()), "value": round(float(v), 4)}
                                 for d, v in (1 + b).cumprod().iloc[::5].items()] for n, b in bench.items()},
        "method_note": "Risk metrics apply current weights to the last 252 trading days. "
                       "Nifty 500 sector weights are approximated from full market caps (the index uses free-float).",
    }
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
