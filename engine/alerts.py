"""Decide which alerts this run should send. Deterministic: same inputs, same alerts.

  python engine/alerts.py <data_dir> <portfolio_out.json|-> <state.json> <out.json>

Writes {"new_alerts": [...], "top10": [...]} where each alert is
  {id, key, time_ist, symbol, type, message, priority}
and `key` is used to send each alert at most once per day (checked against state.alerts).
News alerts are added by the agent itself, not here.
"""
import json, sys, datetime as dt

D, PF, STATE, OUT = sys.argv[1:5]
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
now = dt.datetime.now(IST)
today = now.date().isoformat()
state = json.load(open(STATE))
scores = json.load(open(f"{D}/scores.json"))
market = json.load(open(f"{D}/market.json"))
pf = json.load(open(PF)) if PF != "-" else None
sent = {a.get("key") for a in state.get("alerts", [])}
by_sym = {r["symbol"]: r for r in scores}
out = []


def add(key, symbol, typ, msg, prio):
    k = f"{today}:{key}"
    if k in sent or any(a["key"] == k for a in out):
        return
    out.append({"id": f"{now:%Y%m%d%H%M}-{len(out)}", "key": k, "time_ist": now.isoformat(timespec="minutes"),
                "symbol": symbol, "type": typ, "message": msg, "priority": prio})


# Data freshness during market hours
run_at = dt.datetime.fromisoformat(market["asof_ist"])
if now.weekday() < 5 and 9 <= now.hour <= 16 and (now - run_at).total_seconds() > 3 * 3600:
    add("stale", None, "data", f"Market data is {int((now - run_at).total_seconds() // 3600)}h old; the GitHub feed may have failed.", 2)

prev_top = set(state.get("last_top10", []))
liquid = [r for r in scores if r["liquid"]]
top10 = [r["symbol"] for r in liquid[:10]]
new_in = [s for s in top10 if prev_top and s not in prev_top]
if new_in:
    add("top10:" + ",".join(sorted(new_in)), None, "rank",
        "New in top 10 scores: " + ", ".join(f"{s} ({by_sym[s]['sector']})" for s in new_in), 1)

if pf:
    s = pf["summary"]
    if s["day_pnl_pct"] <= -3:
        add("pf_down3", None, "portfolio", f"Portfolio down {abs(s['day_pnl_pct']):.1f}% today.", 3)
    elif s["day_pnl_pct"] >= 3:
        add("pf_up3", None, "portfolio", f"Portfolio up {s['day_pnl_pct']:.1f}% today.", 2)
    prev_pct = state.get("last_holding_pctl", {})
    for h in pf["holdings"]:
        sym, chg = h["symbol"], h["day_chg_pct"]
        if chg is not None and abs(chg) >= 4:
            d = "up" if chg > 0 else "down"
            add(f"move:{sym}:{d}", sym, "move", f"{sym} {d} {abs(chg):.1f}% today ({h['weight_pct']:.1f}% of portfolio).", 3)
        p, pp = h.get("percentile"), prev_pct.get(sym)
        if p is not None and pp is not None:
            if pp >= 20 > p:
                add(f"pctl_low:{sym}", sym, "rank", f"{sym} score fell to the bottom 20% of the Nifty 500 (percentile {p:.0f}, was {pp:.0f}). Weak on: {', '.join(h['weaknesses']) if isinstance(h['weaknesses'], list) else h['weaknesses'] or 'n/a'}.", 2)
            if pp < 90 <= p:
                add(f"pctl_high:{sym}", sym, "rank", f"{sym} score rose to the top 10% of the Nifty 500 (percentile {p:.0f}).", 1)
        for fl in h.get("flags", []):
            if fl.startswith("At 52-week"):
                add(f"52w:{sym}:{fl}", sym, "move", f"{sym} {fl.lower()}.", 1)

out.sort(key=lambda a: -a["priority"])
json.dump({"new_alerts": out, "top10": top10,
           "holding_pctl": {h["symbol"]: h["percentile"] for h in (pf["holdings"] if pf else []) if h["percentile"] is not None}},
          open(OUT, "w"), indent=1)
print(f"{len(out)} new alert(s)")
for a in out:
    print("-", a["message"])
