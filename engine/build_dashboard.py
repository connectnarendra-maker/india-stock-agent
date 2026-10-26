"""Turn one run's outputs into the dashboard's data files (JSON arrays of rows).

  python engine/build_dashboard.py <data_dir> <state.json> <portfolio_out.json|-> <out_dir>

state.json is the agent's private state (from the dashboard's private store):
  {"news": [...], "alerts": [...], "last_run_ist": "...", "notes": "...", ...}
Writes into out_dir: status, indices, sectors, scores, news, alerts,
pf_summary, pf_vs, pf_holdings, pf_sectors, pf_curve (.json each).
"""
import json, os, sys

D, STATE, PF, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
market = json.load(open(f"{D}/market.json"))
scores = json.load(open(f"{D}/scores.json"))
state = json.load(open(STATE)) if os.path.exists(STATE) else {}
pf = json.load(open(PF)) if PF != "-" and os.path.exists(PF) else None


def w(name, rows):
    json.dump(rows, open(f"{OUT}/{name}.json", "w"), separators=(",", ":"), ensure_ascii=False)


meta = market.get("data_meta", {})
w("status", [{
    "agent_run_ist": state.get("last_run_ist"),
    "market_data_ist": market["asof_ist"],
    "price_bar_date": market["price_bar_date"],
    "fundamentals_ist": (meta.get("refreshed") or {}).get("fundamentals", "")[:16],
    "stocks_scored": len(scores),
    "above_200dma_pct": market["breadth"]["above_200dma_pct"],
    "advancers": market["breadth"]["advancers"],
    "decliners": market["breadth"]["decliners"],
    "risk_free_pct": state.get("risk_free_pct", 5.5),
    "holdings_count": len(state.get("holdings", [])),
    "agent_note": state.get("notes", ""),
}])

names = {"NIFTY50": "Nifty 50", "NIFTY500": "Nifty 500"}
w("indices", [{"index": names.get(k, k), **v} for k, v in market["indices"].items()])
w("sectors", [s for s in market["sectors"] if s.get("n", 0) >= 3])

flat = []
for r in scores:
    flat.append({
        "rank": r["rank"], "symbol": r["symbol"], "company": r["company"], "sector": r["sector"],
        "score": r["score"], "percentile": r["percentile"], "coverage": r["coverage"],
        **{f"f_{k}": v for k, v in r["factors"].items()},
        "strengths": ", ".join(r["top_strengths"]), "weaknesses": ", ".join(r["top_weaknesses"]),
        "price": r["price"], "chg_pct": r["chg_pct"], "ret_1m": r["ret_1m"], "ret_3m": r["ret_3m"],
        "ret_1y": r["ret_1y"], "pe": r["pe"], "pb": r["pb"], "roe": r["roe"], "de": r["de"],
        "rev_growth": r["rev_growth"], "mcap_cr": r["mcap_cr"], "vol_1y": r["vol_1y"],
        "liquid": r["liquid"], "flags": "; ".join(r["flags"]),
    })
w("scores", flat)

w("news", sorted(state.get("news", []), key=lambda n: n.get("time_ist", ""), reverse=True)[:150])
w("alerts", sorted(state.get("alerts", []), key=lambda a: a.get("time_ist", ""), reverse=True)[:100])

if pf:
    s = pf["summary"]
    w("pf_summary", [{k: v for k, v in s.items() if k != "unpriced_symbols"} |
                     {"unpriced": ", ".join(s["unpriced_symbols"]), "risk_free_pct": pf["risk_free_pct"],
                      "asof_ist": pf["asof_ist"]}])
    w("pf_vs", [{"benchmark": names.get(k, k), **{kk: vv for kk, vv in v.items() if kk != "max_drawdown_window"},
                 "dd_from": v["max_drawdown_window"][0], "dd_to": v["max_drawdown_window"][1]}
                for k, v in pf["vs"].items()])
    w("pf_holdings", [h | {"flags": "; ".join(h["flags"]), "strengths": ", ".join(h["strengths"]),
                           "weaknesses": ", ".join(h["weaknesses"])} for h in pf["holdings"]])
    w("pf_sectors", pf["sectors"])
    curve = {p["date"]: {"date": p["date"], "portfolio": p["portfolio"]} for p in pf["equity_curve"]}
    for name, pts in pf["benchmark_curves"].items():
        for p in pts:
            if p["date"] in curve:
                curve[p["date"]][names.get(name, name)] = p["value"]
    w("pf_curve", list(curve.values()))
else:
    for n in ["pf_summary", "pf_vs", "pf_holdings", "pf_sectors", "pf_curve"]:
        w(n, [])
print("Wrote", sorted(os.listdir(OUT)))
