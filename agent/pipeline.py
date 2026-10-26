"""Deterministic parts of the hourly agent run.

  python agent/pipeline.py prepare <state.json>
      Clones the latest data branch into work/data, re-scores with current news
      signals, and writes work/focus.json: which stocks to research news for.

  python agent/pipeline.py finish <state.json>
      Reads work/new_news.json (written by the agent after reading the news),
      merges it into signals/news_sentiment.json and the state, re-scores,
      recomputes the portfolio, decides alerts, builds the dashboard files in
      work/dash, and writes work/state_new.json and work/summary.json.

Holdings live only in state.json (the dashboard's private store). They are
written to work/ (git-ignored) for the portfolio calculation and nowhere else.
"""
import json, os, subprocess, sys, datetime as dt

REPO = "https://github.com/connectnarendra-maker/india-stock-agent"
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
NOW = dt.datetime.now(IST)
W = "work"
PY = sys.executable


def sh(*args):
    subprocess.run(args, check=True)


def load(p, default):
    return json.load(open(p)) if os.path.exists(p) else default


def prepare(state_path):
    state = load(state_path, {})
    sh("rm", "-rf", W)
    os.makedirs(W)
    sh("git", "clone", "-q", "--depth", "1", "-b", "data", REPO, f"{W}/data")
    sh(PY, "-W", "ignore", "engine/score.py", f"{W}/data", "signals")
    scores = json.load(open(f"{W}/data/scores.json"))
    meta = json.load(open(f"{W}/data/meta.json"))
    by = {r["symbol"]: r for r in scores}
    hold = {}
    for h in state.get("holdings", []):
        s = h["symbol"].upper().replace(".NS", "")
        hold[s] = hold.get(s, 0) + h["qty"] * (by.get(s, {}).get("price") or h.get("avg_price") or 0)
    holdings = sorted(hold, key=lambda s: -hold[s])
    liquid = [r for r in scores if r["liquid"]]
    top = [r["symbol"] for r in liquid[:10]]
    movers = [r["symbol"] for r in sorted(liquid, key=lambda r: -abs(r["chg_pct"] or 0))
              if abs(r["chg_pct"] or 0) >= 5][:5]
    recent = {n.get("url") for n in state.get("news", [])}
    focus = {
        "now_ist": NOW.isoformat(timespec="minutes"),
        "market_data_ist": meta["run_at_ist"],
        "holdings": [{"symbol": s, "company": by.get(s, {}).get("company", s),
                      "chg_pct": by.get(s, {}).get("chg_pct")} for s in holdings],
        "top_ranked": [{"symbol": s, "company": by[s]["company"], "sector": by[s]["sector"]} for s in top
                       if s not in hold],
        "big_movers": [{"symbol": s, "company": by[s]["company"], "chg_pct": by[s]["chg_pct"]} for s in movers
                       if s not in hold and s not in top],
        "already_logged_urls": sorted(u for u in recent if u),
        "valid_symbols_note": "Use NSE symbols exactly as in work/data/universe.csv; null for market-wide news.",
        "risk_free_pct": state.get("risk_free_pct", 5.5),
        "risk_free_checked": state.get("risk_free_checked"),
    }
    json.dump(focus, open(f"{W}/focus.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in focus.items() if k != "already_logged_urls"}, indent=1))


def finish(state_path):
    state = load(state_path, {})
    new = load(f"{W}/new_news.json", [])
    if isinstance(new, dict):
        rf = new.get("risk_free_pct")
        if rf:
            state["risk_free_pct"] = float(rf)
            state["risk_free_checked"] = NOW.date().isoformat()
        new = new.get("news", [])
    universe = {l.split(",")[0] for l in open(f"{W}/data/universe.csv").read().splitlines()[1:]}

    # Merge news into the state (dedupe by URL, keep 72h, cap 200)
    seen = {n.get("url") for n in state.get("news", [])}
    added = []
    for i, n in enumerate(new):
        if not n.get("headline") or n.get("url") in seen:
            continue
        sym = (n.get("symbol") or "").upper().replace(".NS", "") or None
        if sym and sym not in universe:
            sym = None
        item = {"id": f"{NOW:%Y%m%d%H%M}-{i}", "time_ist": n.get("time_ist") or NOW.isoformat(timespec="minutes"),
                "symbol": sym, "headline": str(n["headline"])[:200], "summary": str(n.get("summary", ""))[:300],
                "source": str(n.get("source", ""))[:60], "url": n.get("url"),
                "sentiment": max(-1.0, min(1.0, float(n.get("sentiment", 0)))),
                "importance": max(0.5, min(2.0, float(n.get("importance", 1))))}
        added.append(item)
        seen.add(item["url"])
    cutoff = (NOW - dt.timedelta(hours=72)).isoformat()
    state["news"] = sorted([n for n in state.get("news", []) + added if n["time_ist"] >= cutoff],
                           key=lambda n: n["time_ist"], reverse=True)[:200]

    # Sentiment signals for the scoring engine (7 days, symbols only, no holdings data)
    sig = load("signals/news_sentiment.json", {})
    for n in added:
        if n["symbol"]:
            sig.setdefault(n["symbol"], []).append({"score": n["sentiment"], "importance": n["importance"],
                                                    "asof": NOW.isoformat(timespec="minutes"),
                                                    "headline": n["headline"][:120]})
    week = (NOW - dt.timedelta(days=7)).isoformat()
    sig = {s: [x for x in v if x["asof"] >= week] for s, v in sig.items()}
    sig = {s: v for s, v in sig.items() if v}
    json.dump(sig, open("signals/news_sentiment.json", "w"), indent=1, ensure_ascii=False)

    # Re-score, portfolio, alerts
    sh(PY, "-W", "ignore", "engine/score.py", f"{W}/data", "signals")
    pf = "-"
    if state.get("holdings"):
        json.dump(state["holdings"], open(f"{W}/holdings.json", "w"))
        sh(PY, "-W", "ignore", "engine/portfolio.py", f"{W}/data", f"{W}/holdings.json", f"{W}/pf.json",
           str(state.get("risk_free_pct", 5.5)))
        pf = f"{W}/pf.json"
    json.dump(state, open(f"{W}/state_tmp.json", "w"))
    sh(PY, "engine/alerts.py", f"{W}/data", pf, f"{W}/state_tmp.json", f"{W}/alerts.json")
    al = json.load(open(f"{W}/alerts.json"))
    new_alerts = al["new_alerts"]

    # News alerts: strong, important news on a holding
    held = {h["symbol"].upper().replace(".NS", "") for h in state.get("holdings", [])}
    sent = {a["key"] for a in state.get("alerts", [])}
    for n in added:
        if n["symbol"] in held and abs(n["sentiment"]) >= 0.6 and n["importance"] >= 1.5:
            key = f"{NOW.date()}:news:{n['url']}"
            if key not in sent:
                tone = "Positive" if n["sentiment"] > 0 else "Negative"
                new_alerts.append({"id": f"{n['id']}-n", "key": key, "time_ist": n["time_ist"], "symbol": n["symbol"],
                                   "type": "news", "message": f"{tone} news on {n['symbol']}: {n['headline']}",
                                   "priority": 3 if n["sentiment"] < 0 else 2, "url": n["url"]})
    new_alerts.sort(key=lambda a: -a["priority"])
    month = (NOW - dt.timedelta(days=30)).isoformat()
    state["alerts"] = [a for a in state.get("alerts", []) + new_alerts if a["time_ist"] >= month][-300:]
    state["last_top10"] = al["top10"]
    if al["holding_pctl"]:
        state["last_holding_pctl"] = al["holding_pctl"]
    state["last_run_ist"] = NOW.isoformat(timespec="minutes")

    json.dump(state, open(f"{W}/state_new.json", "w"), ensure_ascii=False)
    sh(PY, "engine/build_dashboard.py", f"{W}/data", f"{W}/state_new.json", pf, f"{W}/dash")

    # Which dashboard files changed since the last run (by content hash)
    import hashlib
    hashes = {f[:-5]: hashlib.sha256(open(f"{W}/dash/{f}", "rb").read()).hexdigest()
              for f in os.listdir(f"{W}/dash")}
    prev = state.get("file_hashes", {})
    changed = sorted(k for k, h in hashes.items() if prev.get(k) != h)
    state["file_hashes"] = hashes
    json.dump(state, open(f"{W}/state_new.json", "w"), ensure_ascii=False)
    summary = {"news_added": len(added), "alerts": new_alerts, "changed_files": changed,
               "holdings": len(state.get("holdings", [])),
               "market_data_ist": json.load(open(f"{W}/data/meta.json"))["run_at_ist"]}
    json.dump(summary, open(f"{W}/summary.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"prepare": prepare, "finish": finish}[sys.argv[1]](sys.argv[2])
