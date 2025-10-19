import os
import asyncio
import aiohttp
import redis
import json
from datetime import datetime, timedelta
from telegram import Bot
from telegram.error import TelegramError
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import numpy as np

# Configuration
DERIBIT_API = "https://www.deribit.com/api/v2"
DEEPSEEK_API = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Redis connection
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
telegram_bot = Bot(token=TELEGRAM_TOKEN)

# Instruments to monitor
INSTRUMENTS = ["BTC-PERPETUAL", "ETH-PERPETUAL"]

class DeribitDataFetcher:
    """Fetch data from Deribit Public API"""
    
    async def fetch_orderbook(self, session, instrument):
        """Get order book data"""
        url = f"{DERIBIT_API}/public/get_order_book"
        params = {"instrument_name": instrument, "depth": 100}
        
        async with session.get(url, params=params) as response:
            data = await response.json()
            return data.get("result", {})
    
    async def fetch_trades(self, session, instrument):
        """Get recent trades"""
        url = f"{DERIBIT_API}/public/get_last_trades_by_instrument"
        params = {"instrument_name": instrument, "count": 100}
        
        async with session.get(url, params=params) as response:
            data = await response.json()
            return data.get("result", {}).get("trades", [])
    
    async def fetch_candlesticks(self, session, instrument, timeframe="60"):
        """Get OHLCV data"""
        url = f"{DERIBIT_API}/public/get_tradingview_chart_data"
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(hours=48)).timestamp() * 1000)
        
        params = {
            "instrument_name": instrument,
            "start_timestamp": start_time,
            "end_timestamp": end_time,
            "resolution": timeframe
        }
        
        async with session.get(url, params=params) as response:
            data = await response.json()
            return data.get("result", {})
    
    async def fetch_options_summary(self, session, currency="BTC"):
        """Get options chain summary"""
        url = f"{DERIBIT_API}/public/get_book_summary_by_currency"
        params = {"currency": currency, "kind": "option"}
        
        async with session.get(url, params=params) as response:
            data = await response.json()
            return data.get("result", [])
    
    async def fetch_funding_rate(self, session, instrument):
        """Get funding rate"""
        url = f"{DERIBIT_API}/public/get_funding_rate_history"
        params = {"instrument_name": instrument, "count": 10}
        
        async with session.get(url, params=params) as response:
            data = await response.json()
            return data.get("result", [])
    
    async def fetch_ticker(self, session, instrument):
        """Get ticker data"""
        url = f"{DERIBIT_API}/public/ticker"
        params = {"instrument_name": instrument}
        
        async with session.get(url, params=params) as response:
            data = await response.json()
            return data.get("result", {})

class SmartMoneyAnalyzer:
    """Analyze market data using Smart Money Concepts"""
    
    def detect_order_blocks(self, candles):
        """Identify order block zones"""
        order_blocks = []
        
        if not candles or len(candles.get("close", [])) < 5:
            return order_blocks
        
        closes = candles["close"]
        highs = candles["high"]
        lows = candles["low"]
        volumes = candles["volume"]
        
        for i in range(2, len(closes) - 2):
            # Bullish Order Block
            if (closes[i] > closes[i-1] and 
                volumes[i] > np.mean(volumes[max(0, i-10):i]) * 1.5):
                order_blocks.append({
                    "type": "bullish",
                    "zone_low": lows[i-1],
                    "zone_high": highs[i-1],
                    "strength": "high" if volumes[i] > np.mean(volumes) * 2 else "medium"
                })
            
            # Bearish Order Block
            if (closes[i] < closes[i-1] and 
                volumes[i] > np.mean(volumes[max(0, i-10):i]) * 1.5):
                order_blocks.append({
                    "type": "bearish",
                    "zone_low": lows[i-1],
                    "zone_high": highs[i-1],
                    "strength": "high" if volumes[i] > np.mean(volumes) * 2 else "medium"
                })
        
        return order_blocks[-3:]  # Last 3 order blocks
    
    def detect_fvg(self, candles):
        """Detect Fair Value Gaps"""
        fvgs = []
        
        if not candles or len(candles.get("close", [])) < 3:
            return fvgs
        
        highs = candles["high"]
        lows = candles["low"]
        
        for i in range(1, len(highs) - 1):
            # Bullish FVG
            if lows[i+1] > highs[i-1]:
                fvgs.append({
                    "type": "bullish",
                    "gap_low": highs[i-1],
                    "gap_high": lows[i+1],
                    "unfilled": True
                })
            
            # Bearish FVG
            if highs[i+1] < lows[i-1]:
                fvgs.append({
                    "type": "bearish",
                    "gap_low": highs[i+1],
                    "gap_high": lows[i-1],
                    "unfilled": True
                })
        
        return fvgs[-3:]  # Last 3 FVGs
    
    def analyze_liquidity(self, orderbook, trades):
        """Analyze liquidity pools and sweeps"""
        analysis = {
            "bid_liquidity": 0,
            "ask_liquidity": 0,
            "imbalance": 0,
            "large_trades": [],
            "liquidation_likely": False
        }
        
        if not orderbook:
            return analysis
        
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        
        # Calculate liquidity
        bid_volume = sum([bid[1] for bid in bids[:20]])
        ask_volume = sum([ask[1] for ask in asks[:20]])
        
        analysis["bid_liquidity"] = bid_volume
        analysis["ask_liquidity"] = ask_volume
        analysis["imbalance"] = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        
        # Detect large trades (potential whale activity)
        if trades:
            avg_size = np.mean([t["amount"] for t in trades])
            large_trades = [t for t in trades if t["amount"] > avg_size * 3]
            analysis["large_trades"] = len(large_trades)
        
        return analysis
    
    def calculate_pcr(self, options_data):
        """Calculate Put/Call Ratio"""
        put_oi = 0
        call_oi = 0
        
        for opt in options_data:
            if "put" in opt.get("instrument_name", "").lower():
                put_oi += opt.get("open_interest", 0)
            elif "call" in opt.get("instrument_name", "").lower():
                call_oi += opt.get("open_interest", 0)
        
        return put_oi / call_oi if call_oi > 0 else 0
    
    def analyze_market_structure(self, candles):
        """Determine market structure"""
        if not candles or len(candles.get("close", [])) < 10:
            return "neutral"
        
        closes = candles["close"]
        highs = candles["high"]
        lows = candles["low"]
        
        recent_closes = closes[-10:]
        
        # Simple trend detection
        if recent_closes[-1] > recent_closes[0] and recent_closes[-1] > np.mean(recent_closes):
            return "bullish"
        elif recent_closes[-1] < recent_closes[0] and recent_closes[-1] < np.mean(recent_closes):
            return "bearish"
        else:
            return "neutral"

class DeepSeekAI:
    """Interact with DeepSeek V3 API"""
    
    async def analyze_market(self, market_data):
        """Send data to DeepSeek for analysis"""
        
        prompt = f"""You are an expert Smart Money Concepts trader analyzing crypto markets.

Market Data:
{json.dumps(market_data, indent=2)}

Analyze this data using Smart Money Concepts:
1. Order Blocks - Are there valid institutional zones?
2. Fair Value Gaps - Any unfilled imbalances?
3. Liquidity Analysis - Where are stop hunts likely?
4. Market Structure - Bullish/Bearish/Neutral?
5. Put/Call Ratio - Institutional sentiment?
6. Funding Rate - Overleveraged positions?

Provide a trade recommendation ONLY if confidence is above 65%.

Respond in this EXACT JSON format:
{{
  "signal": "BUY" or "SELL" or "NO_TRADE",
  "instrument": "BTC-PERPETUAL" or "ETH-PERPETUAL",
  "entry_price": number,
  "stop_loss": number,
  "target": number,
  "confidence": number (0-100),
  "reasoning": [
    "reason 1",
    "reason 2",
    "reason 3"
  ],
  "risk_level": "LOW" or "MEDIUM" or "HIGH",
  "timeframe": "1H" or "4H"
}}

Be strict - only generate signals when setup is clear and probability is high."""

        headers = {
            "Authorization": f"Bearer {DEEPSEEK_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are an expert crypto trader using Smart Money Concepts."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(DEEPSEEK_API, json=payload, headers=headers) as response:
                    result = await response.json()
                    
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        # Extract JSON from response
                        start = content.find("{")
                        end = content.rfind("}") + 1
                        if start != -1 and end > start:
                            return json.loads(content[start:end])
                    
                    return None
        except Exception as e:
            print(f"DeepSeek API Error: {e}")
            return None

class ChartGenerator:
    """Generate trading charts"""
    
    def create_signal_chart(self, candles, signal_data):
        """Create chart with signal markers"""
        
        if not candles or "close" not in candles:
            return None
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                        gridspec_kw={'height_ratios': [3, 1]})
        
        times = [datetime.fromtimestamp(t/1000) for t in candles["ticks"]]
        
        # Candlestick chart
        for i in range(len(times)):
            color = 'green' if candles["close"][i] >= candles["open"][i] else 'red'
            ax1.plot([times[i], times[i]], 
                    [candles["low"][i], candles["high"][i]], 
                    color=color, linewidth=1)
            ax1.plot([times[i], times[i]], 
                    [candles["open"][i], candles["close"][i]], 
                    color=color, linewidth=4)
        
        # Mark entry, SL, target
        if signal_data.get("signal") != "NO_TRADE":
            entry = signal_data.get("entry_price")
            sl = signal_data.get("stop_loss")
            target = signal_data.get("target")
            
            ax1.axhline(y=entry, color='blue', linestyle='--', label=f'Entry: {entry}')
            ax1.axhline(y=sl, color='red', linestyle='--', label=f'Stop Loss: {sl}')
            ax1.axhline(y=target, color='green', linestyle='--', label=f'Target: {target}')
        
        ax1.set_title(f"{signal_data.get('instrument')} - {signal_data.get('signal')} Signal", 
                     fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Volume chart
        colors = ['green' if candles["close"][i] >= candles["open"][i] else 'red' 
                 for i in range(len(times))]
        ax2.bar(times, candles["volume"], color=colors, alpha=0.5)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_xlabel('Time', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save to BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf

class TelegramNotifier:
    """Send signals to Telegram"""
    
    async def send_signal(self, signal_data, chart_buffer):
        """Send trade signal with chart"""
        
        if signal_data.get("signal") == "NO_TRADE":
            return
        
        # Format message
        message = f"""
🚨 **SMART MONEY SIGNAL** 🚨

**Instrument:** {signal_data['instrument']}
**Signal:** {signal_data['signal']} {'🟢' if signal_data['signal'] == 'BUY' else '🔴'}
**Confidence:** {signal_data['confidence']}%

**Entry:** ${signal_data['entry_price']:,.2f}
**Stop Loss:** ${signal_data['stop_loss']:,.2f}
**Target:** ${signal_data['target']:,.2f}
**Risk:** {signal_data['risk_level']}

**Reasoning:**
{chr(10).join([f"• {r}" for r in signal_data['reasoning']])}

**Timeframe:** {signal_data['timeframe']}
**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC

⚠️ Always manage your risk!
"""
        
        try:
            # Send chart
            if chart_buffer:
                await telegram_bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID,
                    photo=chart_buffer,
                    caption=message,
                    parse_mode='Markdown'
                )
            else:
                await telegram_bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=message,
                    parse_mode='Markdown'
                )
            
            print(f"✅ Signal sent to Telegram: {signal_data['signal']} {signal_data['instrument']}")
            
        except TelegramError as e:
            print(f"❌ Telegram Error: {e}")

class SmartMoneyBot:
    """Main bot orchestrator"""
    
    def __init__(self):
        self.fetcher = DeribitDataFetcher()
        self.analyzer = SmartMoneyAnalyzer()
        self.ai = DeepSeekAI()
        self.chart_gen = ChartGenerator()
        self.notifier = TelegramNotifier()
    
    async def collect_market_data(self, instrument):
        """Collect all necessary data"""
        
        async with aiohttp.ClientSession() as session:
            # Fetch all data concurrently
            orderbook, trades, candles, funding, ticker = await asyncio.gather(
                self.fetcher.fetch_orderbook(session, instrument),
                self.fetcher.fetch_trades(session, instrument),
                self.fetcher.fetch_candlesticks(session, instrument),
                self.fetcher.fetch_funding_rate(session, instrument),
                self.fetcher.fetch_ticker(session, instrument)
            )
            
            # Get options data for currency
            currency = "BTC" if "BTC" in instrument else "ETH"
            options = await self.fetcher.fetch_options_summary(session, currency)
            
            return {
                "orderbook": orderbook,
                "trades": trades,
                "candles": candles,
                "funding": funding,
                "ticker": ticker,
                "options": options
            }
    
    def prepare_analysis_data(self, raw_data, instrument):
        """Prepare data for AI analysis"""
        
        # Analyze using Smart Money concepts
        order_blocks = self.analyzer.detect_order_blocks(raw_data["candles"])
        fvgs = self.analyzer.detect_fvg(raw_data["candles"])
        liquidity = self.analyzer.analyze_liquidity(raw_data["orderbook"], raw_data["trades"])
        pcr = self.analyzer.calculate_pcr(raw_data["options"])
        market_structure = self.analyzer.analyze_market_structure(raw_data["candles"])
        
        # Prepare data for AI
        analysis_data = {
            "instrument": instrument,
            "current_price": raw_data["ticker"].get("last_price", 0),
            "market_structure": market_structure,
            "order_blocks": order_blocks,
            "fair_value_gaps": fvgs,
            "liquidity_analysis": {
                "bid_liquidity": liquidity["bid_liquidity"],
                "ask_liquidity": liquidity["ask_liquidity"],
                "imbalance": liquidity["imbalance"],
                "large_trades_count": liquidity["large_trades"]
            },
            "options_data": {
                "put_call_ratio": round(pcr, 2),
                "sentiment": "bullish" if pcr < 1 else "bearish" if pcr > 1 else "neutral"
            },
            "funding_rate": raw_data["funding"][0]["interest_8h"] if raw_data["funding"] else 0,
            "volume_24h": raw_data["ticker"].get("stats", {}).get("volume", 0)
        }
        
        return analysis_data
    
    async def analyze_and_signal(self, instrument):
        """Complete analysis pipeline"""
        
        print(f"\n{'='*50}")
        print(f"Analyzing {instrument}...")
        print(f"{'='*50}\n")
        
        # Collect data
        raw_data = await self.collect_market_data(instrument)
        
        # Store in Redis (cache for 30 mins)
        cache_key = f"market_data:{instrument}"
        redis_client.setex(cache_key, 1800, json.dumps(raw_data, default=str))
        
        # Prepare analysis
        analysis_data = self.prepare_analysis_data(raw_data, instrument)
        
        print(f"Market Structure: {analysis_data['market_structure']}")
        print(f"PCR: {analysis_data['options_data']['put_call_ratio']}")
        print(f"Order Blocks: {len(analysis_data['order_blocks'])}")
        print(f"FVGs: {len(analysis_data['fair_value_gaps'])}")
        
        # AI analysis
        signal = await self.ai.analyze_market(analysis_data)
        
        if not signal:
            print("❌ No signal generated (AI error)")
            return
        
        print(f"\n🎯 Signal: {signal.get('signal')}")
        print(f"📊 Confidence: {signal.get('confidence')}%")
        
        # Only send if confidence > 65%
        if signal.get("confidence", 0) >= 65 and signal.get("signal") != "NO_TRADE":
            # Generate chart
            chart = self.chart_gen.create_signal_chart(raw_data["candles"], signal)
            
            # Send to Telegram
            await self.notifier.send_signal(signal, chart)
            
            # Store signal in Redis
            signal_key = f"signal:{instrument}:{int(datetime.now().timestamp())}"
            redis_client.setex(signal_key, 86400, json.dumps(signal))  # 24h expiry
        else:
            print("⚠️ Signal confidence too low or NO_TRADE")
    
    async def run_cycle(self):
        """Run one analysis cycle"""
        
        print(f"\n🚀 Starting analysis cycle at {datetime.now()}")
        
        for instrument in INSTRUMENTS:
            try:
                await self.analyze_and_signal(instrument)
                await asyncio.sleep(5)  # Small delay between instruments
            except Exception as e:
                print(f"❌ Error analyzing {instrument}: {e}")
        
        print(f"\n✅ Cycle completed at {datetime.now()}")
        print("⏰ Next run in 30 minutes...\n")
    
    async def start(self):
        """Main bot loop"""
        
        print("="*60)
        print("🤖 SMART MONEY BOT STARTED")
        print("="*60)
        print(f"Monitoring: {', '.join(INSTRUMENTS)}")
        print(f"Analysis Interval: Every 30 minutes")
        print(f"AI Model: DeepSeek V3")
        print("="*60)
        
        # Send startup message
        try:
            await telegram_bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text="🤖 Smart Money Bot is now ACTIVE! 🚀\n\nMonitoring BTC & ETH markets...",
                parse_mode='Markdown'
            )
        except:
            pass
        
        while True:
            try:
                await self.run_cycle()
                await asyncio.sleep(1800)  # 30 minutes
            except KeyboardInterrupt:
                print("\n⚠️ Bot stopped by user")
                break
            except Exception as e:
                print(f"❌ Critical error: {e}")
                await asyncio.sleep(60)  # Wait 1 min before retry

if __name__ == "__main__":
    bot = SmartMoneyBot()
    asyncio.run(bot.start())
