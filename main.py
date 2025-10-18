import asyncio
import aiohttp
import redis
import json
import pandas as pd
import numpy as np
from datetime import datetime
from telegram import Bot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import Dict, List, Tuple, Optional
import ta
import traceback
from dataclasses import dataclass

DERIBIT_API = "https://www.deribit.com/api/v2/public"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    print("Redis connected")
except:
    redis_client = None
    print("Redis not available")

@dataclass
class TradingSignal:
    signal: str
    confidence: float
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    reason: str = ""
    python_score: float = 0
    ai_analysis: Optional[Dict] = None

class DeribitDataFetcher:
    def __init__(self):
        self.session = None
        self.request_count = 0
        self.last_request_time = 0
    
    async def init_session(self):
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'TradingBot/1.0'}
            )
    
    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def rate_limit(self):
        """Simple rate limiting - wait 0.5s between requests"""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < 0.5:
            await asyncio.sleep(0.5 - time_since_last)
        self.last_request_time = asyncio.get_event_loop().time()
    
    async def get_candlestick_data(self, instrument: str, timeframe: str, count: int = 500):
        await self.init_session()
        await self.rate_limit()
        
        resolution = {"15": 15, "30": 30, "60": 60, "240": 240}.get(timeframe, 30)
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = end_time - (count * resolution * 60 * 1000)
        
        params = {
            "instrument_name": instrument,
            "start_timestamp": start_time,
            "end_timestamp": end_time,
            "resolution": resolution
        }
        
        print(f"Fetching {instrument} {timeframe}min data...")
        
        try:
            async with self.session.get(
                f"{DERIBIT_API}/get_tradingview_chart_data",
                params=params
            ) as resp:
                if resp.status != 200:
                    print(f"API returned status {resp.status}")
                    text = await resp.text()
                    print(f"Response: {text[:200]}")
                    return pd.DataFrame()
                
                data = await resp.json()
                
                # Debug output
                if 'error' in data:
                    print(f"API Error: {data['error']}")
                    return pd.DataFrame()
                
                if 'result' not in data:
                    print(f"No result in response: {data}")
                    return pd.DataFrame()
                
                result = data['result']
                
                if result.get('status') != 'ok':
                    print(f"Status not OK: {result.get('status')}")
                    return pd.DataFrame()
                
                # Check if we have data
                if not result.get('ticks') or len(result['ticks']) == 0:
                    print(f"No ticks in response")
                    return pd.DataFrame()
                
                df = pd.DataFrame({
                    'timestamp': result['ticks'],
                    'open': result['open'],
                    'high': result['high'],
                    'low': result['low'],
                    'close': result['close'],
                    'volume': result['volume']
                })
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                print(f"✓ Got {len(df)} candles for {instrument}")
                return df
                
        except asyncio.TimeoutError:
            print(f"Timeout fetching {instrument}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Data fetch error for {instrument}: {e}")
            traceback.print_exc()
            return pd.DataFrame()
    
    async def get_open_interest(self, instrument: str):
        await self.init_session()
        await self.rate_limit()
        
        try:
            async with self.session.get(
                f"{DERIBIT_API}/get_book_summary_by_instrument",
                params={"instrument_name": instrument}
            ) as resp:
                if resp.status != 200:
                    print(f"OI fetch failed: status {resp.status}")
                    return {'open_interest': 0, 'volume_usd': 0, 'last_price': 0, 'funding_rate': 0}
                
                data = await resp.json()
                if 'result' in data and data['result']:
                    r = data['result'][0]
                    return {
                        'open_interest': r.get('open_interest', 0),
                        'volume_usd': r.get('volume_usd', 0),
                        'last_price': r.get('last', 0),
                        'funding_rate': r.get('funding_8h', 0) * 100
                    }
        except Exception as e:
            print(f"OI fetch error: {e}")
        
        return {'open_interest': 0, 'volume_usd': 0, 'last_price': 0, 'funding_rate': 0}
    
    async def get_liquidations(self, currency: str):
        await self.init_session()
        await self.rate_limit()
        
        try:
            async with self.session.get(
                f"{DERIBIT_API}/get_last_trades_by_currency",
                params={"currency": currency, "count": 100, "include_old": False}
            ) as resp:
                if resp.status != 200:
                    return {'long': 0, 'short': 0}
                
                data = await resp.json()
                liq = {'long': 0, 'short': 0}
                
                if 'result' in data and 'trades' in data['result']:
                    for t in data['result']['trades']:
                        if t.get('liquidation'):
                            liq['long' if t['direction'] == 'sell' else 'short'] += t['amount']
                
                return liq
        except Exception as e:
            print(f"Liquidation fetch error: {e}")
        
        return {'long': 0, 'short': 0}

class TechnicalAnalyzer:
    @staticmethod
    def calculate_indicators(df):
        if len(df) < 200:
            print(f"⚠ Only {len(df)} candles, need 200+ for all indicators")
            if len(df) < 50:
                return df
        
        try:
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            
            if len(df) >= 200:
                df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
            
            print(f"✓ Indicators calculated")
        except Exception as e:
            print(f"Indicator calculation error: {e}")
        
        return df

class MarketAnalyzer:
    def __init__(self):
        self.fetcher = DeribitDataFetcher()
    
    async def analyze(self, coin: str):
        await self.fetcher.init_session()
        instrument = f"{coin}-PERPETUAL"
        
        try:
            print(f"\n{'='*50}")
            print(f"Analyzing {coin}")
            print(f"{'='*50}")
            
            # Fetch with delays
            df_30m = await self.fetcher.get_candlestick_data(instrument, "30", 500)
            await asyncio.sleep(1)
            
            df_1h = await self.fetcher.get_candlestick_data(instrument, "60", 300)
            await asyncio.sleep(1)
            
            df_4h = await self.fetcher.get_candlestick_data(instrument, "240", 200)
            await asyncio.sleep(1)
            
            # Check if we got data
            if df_30m.empty:
                raise Exception(f"No 30m data for {instrument}")
            if df_1h.empty:
                raise Exception(f"No 1h data for {instrument}")
            if df_4h.empty:
                raise Exception(f"No 4h data for {instrument}")
            
            print(f"✓ All timeframes loaded")
            
            # Fetch additional data
            oi = await self.fetcher.get_open_interest(instrument)
            liq = await self.fetcher.get_liquidations(coin)
            
            # Calculate indicators
            df_30m = TechnicalAnalyzer.calculate_indicators(df_30m)
            df_1h = TechnicalAnalyzer.calculate_indicators(df_1h)
            df_4h = TechnicalAnalyzer.calculate_indicators(df_4h)
            
            # Determine trends
            trend_4h = "BULLISH" if df_4h['ema_20'].iloc[-1] > df_4h['ema_50'].iloc[-1] else "BEARISH"
            trend_1h = "BULLISH" if df_1h['ema_20'].iloc[-1] > df_1h['ema_50'].iloc[-1] else "BEARISH"
            
            price = float(df_30m['close'].iloc[-1])
            rsi = float(df_30m['rsi'].iloc[-1]) if not pd.isna(df_30m['rsi'].iloc[-1]) else 50
            
            print(f"✓ Analysis complete: ${price:.2f}, RSI: {rsi:.1f}")
            
            return {
                'coin': coin,
                'price': price,
                'trend_4h': trend_4h,
                'trend_1h': trend_1h,
                'rsi': rsi,
                'oi': oi['open_interest'],
                'liq': liq,
                'df': df_30m
            }
            
        except Exception as e:
            print(f"❌ Analysis failed for {coin}: {e}")
            raise
        finally:
            await self.fetcher.close_session()

class AIAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def analyze(self, analysis: Dict):
        score = 0
        reasons = []
        signal = "NO_TRADE"
        
        # Trend analysis
        if analysis['trend_4h'] == "BULLISH" and analysis['trend_1h'] == "BULLISH":
            score += 50
            signal = "LONG"
            reasons.append("Bullish trend alignment")
        elif analysis['trend_4h'] == "BEARISH" and analysis['trend_1h'] == "BEARISH":
            score += 50
            signal = "SHORT"
            reasons.append("Bearish trend alignment")
        
        # RSI analysis
        rsi = analysis['rsi']
        if signal == "LONG" and 30 < rsi < 50:
            score += 20
            reasons.append("RSI favorable for long")
        elif signal == "SHORT" and 50 < rsi < 70:
            score += 20
            reasons.append("RSI favorable for short")
        elif rsi > 75 or rsi < 25:
            score -= 20
            reasons.append("RSI extreme zone")
        
        # Liquidation analysis
        liq = analysis['liq']
        total = liq['long'] + liq['short']
        if total > 0:
            if signal == "LONG" and (liq['long'] / total) > 0.7:
                score -= 15
                reasons.append("Heavy long liquidations")
            elif signal == "SHORT" and (liq['short'] / total) > 0.7:
                score -= 15
                reasons.append("Heavy short liquidations")
        
        score = max(0, min(100, score))
        
        if score < 60:
            signal = "NO_TRADE"
            reasons.append("Confidence below threshold")
        
        # Calculate targets
        price = analysis['price']
        entry = price
        
        if signal == "LONG":
            sl = price * 0.97
            tp1 = price * 1.04
            tp2 = price * 1.08
        elif signal == "SHORT":
            sl = price * 1.03
            tp1 = price * 0.96
            tp2 = price * 0.92
        else:
            sl = tp1 = tp2 = None
        
        return TradingSignal(
            signal=signal,
            confidence=score,
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            reason=', '.join(reasons),
            python_score=score,
            ai_analysis={'confidence': score}
        )

class ChartGenerator:
    @staticmethod
    def create_chart(df, analysis, signal, coin):
        try:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{coin} 30min', 'RSI'),
                vertical_spacing=0.05
            )
            
            df_plot = df.tail(100)
            
            # Candlestick
            fig.add_trace(go.Candlestick(
                x=df_plot['timestamp'],
                open=df_plot['open'],
                high=df_plot['high'],
                low=df_plot['low'],
                close=df_plot['close'],
                name='Price'
            ), row=1, col=1)
            
            # EMAs
            if 'ema_20' in df_plot.columns:
                fig.add_trace(go.Scatter(
                    x=df_plot['timestamp'],
                    y=df_plot['ema_20'],
                    name='EMA20',
                    line=dict(color='blue', width=1)
                ), row=1, col=1)
            
            if 'ema_50' in df_plot.columns:
                fig.add_trace(go.Scatter(
                    x=df_plot['timestamp'],
                    y=df_plot['ema_50'],
                    name='EMA50',
                    line=dict(color='orange', width=1)
                ), row=1, col=1)
            
            # Trading levels
            if signal.signal in ['LONG', 'SHORT'] and signal.entry:
                times = [df_plot['timestamp'].iloc[0], df_plot['timestamp'].iloc[-1]]
                
                fig.add_trace(go.Scatter(
                    x=times, y=[signal.entry, signal.entry],
                    mode='lines', name='Entry',
                    line=dict(color='blue', width=2, dash='dot')
                ), row=1, col=1)
                
                if signal.sl:
                    fig.add_trace(go.Scatter(
                        x=times, y=[signal.sl, signal.sl],
                        mode='lines', name='SL',
                        line=dict(color='red', width=2)
                    ), row=1, col=1)
                
                if signal.tp1:
                    fig.add_trace(go.Scatter(
                        x=times, y=[signal.tp1, signal.tp1],
                        mode='lines', name='TP1',
                        line=dict(color='green', width=2)
                    ), row=1, col=1)
            
            # RSI
            if 'rsi' in df_plot.columns:
                fig.add_trace(go.Scatter(
                    x=df_plot['timestamp'],
                    y=df_plot['rsi'],
                    name='RSI',
                    line=dict(color='purple')
                ), row=2, col=1)
                
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            
            # Title
            sig_text = "🟢 LONG" if signal.signal == "LONG" else "🔴 SHORT" if signal.signal == "SHORT" else "⚪ NO TRADE"
            fig.update_layout(
                title=f"{coin} - {sig_text} | Confidence: {signal.confidence:.0f}%",
                xaxis_rangeslider_visible=False,
                height=800,
                template='plotly_white',
                showlegend=True
            )
            
            filename = f"/tmp/{coin}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.write_image(filename, width=1400, height=800)
            print(f"✓ Chart saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"Chart generation error: {e}")
            traceback.print_exc()
            return None

class TradingBot:
    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token=token)
        self.chat_id = chat_id
        self.analyzer = MarketAnalyzer()
        self.ai = AIAnalyzer(DEEPSEEK_API_KEY)
        self.chart = ChartGenerator()
    
    async def send_signal(self, coin: str, analysis: Dict, signal: TradingSignal, chart_path: str):
        emoji = "🟢" if signal.signal == "LONG" else "🔴" if signal.signal == "SHORT" else "⚪"
        
        msg = f"""{emoji} **{coin} SIGNAL** {emoji}

**Signal:** {signal.signal}
**Confidence:** {signal.confidence:.0f}%

**Market Data:**
Price: ${analysis['price']:.2f}
Trend 4H: {analysis['trend_4h']}
Trend 1H: {analysis['trend_1h']}
RSI: {analysis['rsi']:.1f}
Open Interest: {analysis['oi']:,.0f}
"""
        
        if signal.entry:
            msg += f"""
**Trading Levels:**
Entry: ${signal.entry:.2f}
Stop Loss: ${signal.sl:.2f}
Target 1: ${signal.tp1:.2f}
Target 2: ${signal.tp2:.2f}
"""
        
        msg += f"\n**Analysis:** {signal.reason}\n\n_{datetime.now().strftime('%Y-%m-%d %H:%M')} UTC_"
        
        try:
            if chart_path and os.path.exists(chart_path):
                with open(chart_path, 'rb') as photo:
                    await self.bot.send_photo(
                        chat_id=self.chat_id,
                        photo=photo,
                        caption=msg,
                        parse_mode='Markdown'
                    )
                print(f"✓ Signal with chart sent for {coin}")
            else:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=msg,
                    parse_mode='Markdown'
                )
                print(f"✓ Signal sent for {coin} (no chart)")
                
        except Exception as e:
            print(f"❌ Send error: {e}")
            traceback.print_exc()
    
    async def analyze_and_signal(self, coin: str):
        try:
            analysis = await self.analyzer.analyze(coin)
            signal = await self.ai.analyze(analysis)
            
            print(f"\n📊 {coin} Result: {signal.signal} | Confidence: {signal.confidence:.0f}%")
            
            # Generate chart
            chart_path = None
            if signal.confidence >= 60 and signal.signal in ['LONG', 'SHORT']:
                chart_path = self.chart.create_chart(analysis['df'], analysis, signal, coin)
                await self.send_signal(coin, analysis, signal, chart_path)
            else:
                print(f"⚠ No signal sent (confidence: {signal.confidence:.0f}%)")
            
            # Cleanup
            if chart_path and os.path.exists(chart_path):
                os.remove(chart_path)
                
        except Exception as e:
            print(f"❌ {coin} error: {e}")
            traceback.print_exc()
    
    async def run(self):
        print("\n" + "="*50)
        print("🤖 TRADING BOT STARTED")
        print("="*50)
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text="🤖 *Trading Bot Started!*\n\nMonitoring BTC and ETH...",
                parse_mode='Markdown'
            )
        except Exception as e:
            print(f"⚠ Could not send startup message: {e}")
        
        cycle = 0
        while True:
            try:
                cycle += 1
                print(f"\n{'='*50}")
                print(f"Cycle #{cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*50}")
                
                await self.analyze_and_signal("BTC")
                await asyncio.sleep(5)
                
                await self.analyze_and_signal("ETH")
                
                print(f"\n⏳ Waiting 30 minutes until next cycle...")
                await asyncio.sleep(1800)  # 30 minutes
                
            except KeyboardInterrupt:
                print("\n🛑 Shutting down...")
                break
            except Exception as e:
                print(f"❌ Cycle error: {e}")
                traceback.print_exc()
                print("⏳ Waiting 60 seconds before retry...")
                await asyncio.sleep(60)

async def main():
    bot = TradingBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        traceback.print_exc()
