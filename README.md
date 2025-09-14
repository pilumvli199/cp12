# Crypto Phase-5 Bot

## Features
- Monitors 11 coins (BTC, ETH, SOL, XRP, AAVE, TRX, DOGE, BNB, ADA, LTC, LINK)
- Binance 30m candles + volume fetched live
- GPT-4o-mini advanced analysis (candlesticks, chart patterns, S/R, trendline, volume, OI, long/short ratio)
- Alerts only when CONF >= 65% and bias is BUY/SELL
- Telegram alerts with annotated black & white candlestick chart

## Deploy on Railway
1. Copy `.env.example` → `.env` and fill in your keys.
2. Push repo to GitHub.
3. Create a new Railway project and deploy.
4. Add variables from `.env` into Railway dashboard.
