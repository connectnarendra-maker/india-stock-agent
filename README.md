# India Stock Agent

A research assistant for Indian equities. It ranks the Nifty 500 and analyses a portfolio. It does not give buy or sell recommendations.

**Main branch:** code, plus `signals/` (news sentiment written by the agent, and an optional extra watchlist).
**`data` branch:** rebuilt every hour by GitHub Actions. It holds prices, fundamentals, scores and market overview, and keeps no history.

| File | What it does |
|---|---|
| `engine/fetch.py` | Downloads the Nifty 500 list (NSE), prices (Yahoo Finance) and fundamentals |
| `engine/score.py` | Scores each stock on 7 factors: momentum, trend, quality, value, growth, low volatility and news sentiment. Each is compared within the stock's sector. |
| `engine/portfolio.py` | Compares the portfolio with Nifty 50 and Nifty 500 (return, XIRR, beta, alpha, Sharpe, Sortino, drawdown, VaR, capture ratios, risk contribution, sector tilts). Runs only inside the agent session. Holdings are never committed here. |

The data comes from free sources (NSE files, Yahoo Finance). It can be delayed or occasionally missing. Check anything important before acting on it.
