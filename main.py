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
    print("✓ Redis connected")
except:
    redis_client = None
    print("⚠ Redis not available")

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
    ai_score: float = 0
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
        
        resolution = {"15": 15, "30": 30, "60": 60, "180": 180}.get(timeframe, 30)
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
                
                if 'error' in data:
                    print(f"API Error: {data['error']}")
                    return pd.DataFrame()
                
                if 'result' not in data:
                    print(f"No result in response")
                    return pd.DataFrame()
                
                result = data['result']
                
                if result.get('status') != 'ok':
                    print(f"Status not OK: {result.get('status')}")
                    return pd.DataFrame()
                
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
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_high'] = bb.bollinger_hband()
            df['bb_mid'] = bb.bollinger_mavg()
            df['bb_low'] = bb.bollinger_lband()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
            print(f"✓ Indicators calculated")
        except Exception as e:
            print(f"Indicator calculation error: {e}")
        
        return df
    
    @staticmethod
    def get_pattern_analysis(df):
        """Detect candlestick patterns and market structure"""
        try:
            last_5 = df.tail(5)
            patterns = []
            
            # Price action analysis
            recent_high = df.tail(20)['high'].max()
            recent_low = df.tail(20)['low'].min()
            current_price = df['close'].iloc[-1]
            
            # Volume analysis
            avg_volume = df['volume'].tail(20).mean()
            current_volume = df['volume'].iloc[-1]
            volume_spike = current_volume > (avg_volume * 1.5)
            
            # Momentum
            price_change_5 = ((df['close'].iloc[-1] / df['close'].iloc[-5]) - 1) * 100
            
            return {
                'recent_high': float(recent_high),
                'recent_low': float(recent_low),
                'price_position': float((current_price - recent_low) / (recent_high - recent_low) * 100),
                'volume_spike': volume_spike,
                'volume_ratio': float(current_volume / avg_volume),
                'momentum_5candles': float(price_change_5),
                'consolidating': abs(price_change_5) < 1
            }
        except:
            return {}

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
            df_15m = await self.fetcher.get_candlestick_data(instrument, "15", 500)
            await asyncio.sleep(1)
            
            df_1h = await self.fetcher.get_candlestick_data(instrument, "60", 300)
            await asyncio.sleep(1)
            
            df_3h = await self.fetcher.get_candlestick_data(instrument, "180", 200)
            await asyncio.sleep(1)
            
            # Check if we got data
            if df_15m.empty or df_1h.empty or df_3h.empty:
                raise Exception(f"Missing data for {instrument}")
            
            print(f"✓ All timeframes loaded")
            
            # Fetch additional data
            oi = await self.fetcher.get_open_interest(instrument)
            liq = await self.fetcher.get_liquidations(coin)
            
            # Calculate indicators
            df_15m = TechnicalAnalyzer.calculate_indicators(df_15m)
            df_1h = TechnicalAnalyzer.calculate_indicators(df_1h)
            df_3h = TechnicalAnalyzer.calculate_indicators(df_3h)
            
            # Get pattern analysis
            patterns = TechnicalAnalyzer.get_pattern_analysis(df_15m)
            
            # Determine trends
            trend_3h = "BULLISH" if df_3h['ema_20'].iloc[-1] > df_3h['ema_50'].iloc[-1] else "BEARISH"
            trend_1h = "BULLISH" if df_1h['ema_20'].iloc[-1] > df_1h['ema_50'].iloc[-1] else "BEARISH"
            
            price = float(df_15m['close'].iloc[-1])
            rsi_15m = float(df_15m['rsi'].iloc[-1]) if not pd.isna(df_15m['rsi'].iloc[-1]) else 50
            rsi_1h = float(df_1h['rsi'].iloc[-1]) if not pd.isna(df_1h['rsi'].iloc[-1]) else 50
            
            macd_15m = float(df_15m['macd'].iloc[-1] - df_15m['macd_signal'].iloc[-1]) if 'macd' in df_15m.columns else 0
            
            print(f"✓ Analysis complete: ${price:.2f}, RSI: {rsi_15m:.1f}")
            
            return {
                'coin': coin,
                'price': price,
                'trend_3h': trend_3h,
                'trend_1h': trend_1h,
                'rsi_15m': rsi_15m,
                'rsi_1h': rsi_1h,
                'macd_histogram': macd_15m,
                'oi': oi['open_interest'],
                'volume_24h': oi['volume_usd'],
                'funding_rate': oi['funding_rate'],
                'liq': liq,
                'patterns': patterns,
                'df_15m': df_15m,
                'df_1h': df_1h,
                'df_3h': df_3h
            }
            
        except Exception as e:
            print(f"❌ Analysis failed for {coin}: {e}")
            raise
        finally:
            await self.fetcher.close_session()

class AIAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = None
    
    async def init_session(self):
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            )
    
    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    def build_analysis_prompt(self, analysis: Dict) -> str:
        """Build detailed prompt for DeepSeek V3"""
        
        # Get recent price action from last 10 candles
        df = analysis['df_15m'].tail(10)
        price_action = []
        for i, row in df.iterrows():
            change = ((row['close'] - row['open']) / row['open']) * 100
            price_action.append(f"${row['close']:.2f} ({change:+.2f}%)")
        
        prompt = f"""You are an expert cryptocurrency trading analyst. Analyze this {analysis['coin']} market data and provide trading signals.

MARKET DATA:
- Current Price: ${analysis['price']:.2f}
- Trend 3H: {analysis['trend_3h']}
- Trend 1H: {analysis['trend_1h']}
- RSI 15m: {analysis['rsi_15m']:.1f}
- RSI 1H: {analysis['rsi_1h']:.1f}
- MACD Histogram: {analysis['macd_histogram']:.4f}
- 24H Volume: ${analysis['volume_24h']:,.0f}
- Open Interest: {analysis['oi']:,.0f}
- Funding Rate: {analysis['funding_rate']:.4f}%
- Long Liquidations: {analysis['liq']['long']:.2f}
- Short Liquidations: {analysis['liq']['short']:.2f}

PRICE ACTION (Last 10 candles):
{', '.join(price_action)}

PATTERN ANALYSIS:
- Price Position in Range: {analysis['patterns'].get('price_position', 0):.1f}%
- Volume Spike: {analysis['patterns'].get('volume_spike', False)}
- Volume Ratio: {analysis['patterns'].get('volume_ratio', 1):.2f}x
- 5-Candle Momentum: {analysis['patterns'].get('momentum_5candles', 0):.2f}%
- Consolidating: {analysis['patterns'].get('consolidating', False)}

ANALYSIS REQUIREMENTS:
1. Consider multi-timeframe trend alignment
2. Analyze RSI divergences and overbought/oversold conditions
3. Evaluate volume profile and liquidation imbalances
4. Assess market structure (support/resistance)
5. Consider funding rate implications
6. Identify entry timing based on momentum

Respond ONLY with valid JSON in this exact format:
{{
    "signal": "LONG" or "SHORT" or "NO_TRADE",
    "confidence": 0-100,
    "reasoning": "brief explanation",
    "key_factors": ["factor1", "factor2", "factor3"],
    "risk_level": "LOW" or "MEDIUM" or "HIGH",
    "entry_timing": "IMMEDIATE" or "WAIT_FOR_PULLBACK" or "WAIT_FOR_BREAKOUT"
}}"""
        
        return prompt
    
    async def call_deepseek(self, prompt: str) -> Optional[Dict]:
        """Call DeepSeek V3 API"""
        try:
            await self.init_session()
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional cryptocurrency trading analyst. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            print("🤖 Calling DeepSeek V3 API...")
            
            async with self.session.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload
            ) as resp:
                if resp.status != 200:
                    print(f"❌ DeepSeek API error: {resp.status}")
                    return None
                
                data = await resp.json()
                
                if 'choices' not in data or not data['choices']:
                    print("❌ No response from DeepSeek")
                    return None
                
                content = data['choices'][0]['message']['content'].strip()
                
                # Extract JSON from response
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()
                
                ai_response = json.loads(content)
                print(f"✓ DeepSeek analysis received: {ai_response['signal']} ({ai_response['confidence']}%)")
                
                return ai_response
                
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse DeepSeek response: {e}")
            return None
        except Exception as e:
            print(f"❌ DeepSeek API call failed: {e}")
            traceback.print_exc()
            return None
    
    async def analyze(self, analysis: Dict):
        """Combined Python + AI analysis"""
        
        # Python-based analysis
        python_score = 0
        reasons = []
        signal = "NO_TRADE"
        
        # Trend analysis
        if analysis['trend_3h'] == "BULLISH" and analysis['trend_1h'] == "BULLISH":
            python_score += 40
            signal = "LONG"
            reasons.append("Bullish trend alignment")
        elif analysis['trend_3h'] == "BEARISH" and analysis['trend_1h'] == "BEARISH":
            python_score += 40
            signal = "SHORT"
            reasons.append("Bearish trend alignment")
        
        # RSI analysis
        rsi = analysis['rsi_15m']
        if signal == "LONG" and 30 < rsi < 50:
            python_score += 15
            reasons.append("RSI favorable")
        elif signal == "SHORT" and 50 < rsi < 70:
            python_score += 15
            reasons.append("RSI favorable")
        elif rsi > 75 or rsi < 25:
            python_score -= 15
            reasons.append("RSI extreme")
        
        # MACD momentum
        if signal == "LONG" and analysis['macd_histogram'] > 0:
            python_score += 10
            reasons.append("MACD bullish")
        elif signal == "SHORT" and analysis['macd_histogram'] < 0:
            python_score += 10
            reasons.append("MACD bearish")
        
        # Volume confirmation
        if analysis['patterns'].get('volume_spike'):
            python_score += 10
            reasons.append("Volume spike")
        
        # Liquidation analysis
        liq = analysis['liq']
        total = liq['long'] + liq['short']
        if total > 0:
            if signal == "LONG" and (liq['long'] / total) > 0.7:
                python_score -= 15
                reasons.append("Heavy long liq")
            elif signal == "SHORT" and (liq['short'] / total) > 0.7:
                python_score -= 15
                reasons.append("Heavy short liq")
        
        python_score = max(0, min(100, python_score))
        
        # AI Analysis
        ai_score = python_score  # Default fallback
        ai_response = None
        
        if self.api_key and self.api_key != "YOUR_DEEPSEEK_KEY":
            try:
                prompt = self.build_analysis_prompt(analysis)
                ai_response = await self.call_deepseek(prompt)
                
                if ai_response:
                    ai_signal = ai_response.get('signal', signal)
                    ai_score = ai_response.get('confidence', python_score)
                    
                    # Combine signals - if they agree, boost confidence
                    if ai_signal == signal:
                        # Both agree - weighted average favoring AI
                        final_score = (python_score * 0.3) + (ai_score * 0.7)
                        reasons.append(f"AI confirms: {ai_response.get('reasoning', 'Analysis aligned')}")
                    else:
                        # Disagreement - be more conservative
                        final_score = min(python_score, ai_score) * 0.7
                        signal = "NO_TRADE"
                        reasons.append(f"AI disagrees: {ai_signal}")
                    
                    # Add AI key factors
                    if 'key_factors' in ai_response:
                        reasons.extend(ai_response['key_factors'][:2])
                else:
                    final_score = python_score
                    reasons.append("AI unavailable")
            except Exception as e:
                print(f"⚠ AI analysis failed: {e}")
                final_score = python_score
                reasons.append("AI error")
        else:
            final_score = python_score
            reasons.append("AI not configured")
        
        # Final decision threshold
        if final_score < 60:
            signal = "NO_TRADE"
        
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
        
        await self.close_session()
        
        return TradingSignal(
            signal=signal,
            confidence=final_score,
            entry=entry,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            reason=', '.join(reasons[:4]),  # Limit to 4 reasons
            python_score=python_score,
            ai_score=ai_score,
            ai_analysis=ai_response
        )

class ChartGenerator:
    @staticmethod
    def create_chart(df, analysis, signal, coin):
        try:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{coin} 15min', 'RSI'),
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
            
            # Bollinger Bands
            if 'bb_high' in df_plot.columns:
                fig.add_trace(go.Scatter(
                    x=df_plot['timestamp'],
                    y=df_plot['bb_high'],
                    name='BB Upper',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df_plot['timestamp'],
                    y=df_plot['bb_low'],
                    name='BB Lower',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(128,128,128,0.1)',
                    showlegend=False
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
                title=f"{coin} - {sig_text} | Confidence: {signal.confidence:.0f}% (Py:{signal.python_score:.0f}% AI:{signal.ai_score:.0f}%)",
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
_Python: {signal.python_score:.0f}% | AI: {signal.ai_score:.0f}%_

**Market Data:**
Price: ${analysis['price']:.2f}
Trend 3H: {analysis['trend_3h']}
Trend 1H: {analysis['trend_1h']}
RSI 15m: {analysis['rsi_15m']:.1f}
RSI 1H: {analysis['rsi_1h']:.1f}
Open Interest: {analysis['oi']:,.0f}
24H Volume: ${analysis['volume_24h']:,.0f}
Funding Rate: {analysis['funding_rate']:.4f}%
"""
