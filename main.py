"""
Deribit Crypto Trading Bot - Complete Implementation
BTC & ETH Analysis with Smart Money Concepts + DeepSeek V3 AI
FIXED: Uses pandas-ta instead of talib (no system dependencies)
"""

import asyncio
import aiohttp
import redis
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from telegram import Bot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import Dict, List, Tuple, Optional
import ta  # Using 'ta' library - stable technical analysis
import traceback
from dataclasses import dataclass

# ==================== CONFIG ====================
DERIBIT_API = "https://www.deribit.com/api/v2/public"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

# Redis Connection with error handling
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    print("✅ Redis connected successfully")
except Exception as e:
    print(f"⚠️ Redis connection failed: {e}")
    redis_client = None

# ==================== DATA CLASSES ====================
@dataclass
class TradingSignal:
    signal: str  # LONG/SHORT/NO_TRADE
    confidence: float
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp1: Optional[float] = None
    tp2: Optional[float] = None
    reason: str = ""
    python_score: float = 0
    ai_analysis: Optional[Dict] = None

# ==================== DATA FETCHER ====================
class DeribitDataFetcher:
    def __init__(self):
        self.session = None
    
    async def init_session(self):
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_candlestick_data(self, instrument: str, timeframe: str, count: int = 500) -> pd.DataFrame:
        """Fetch OHLCV candlestick data"""
        await self.init_session()
        
        resolution_map = {"15": 15, "30": 30, "60": 60, "240": 240}
        resolution = resolution_map.get(timeframe, 30)
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = end_time - (count * resolution * 60 * 1000)
        
        url = f"{DERIBIT_API}/get_tradingview_chart_data"
        params = {
            "instrument_name": instrument,
            "start_timestamp": start_time,
            "end_timestamp": end_time,
            "resolution": resolution
        }
        
        try:
            async with self.session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get('result', {}).get('status') == 'ok':
                    result = data['result']
                    df = pd.DataFrame({
                        'timestamp': result['ticks'],
                        'open': result['open'],
                        'high': result['high'],
                        'low': result['low'],
                        'close': result['close'],
                        'volume': result['volume']
                    })
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
        except Exception as e:
            print(f"Error fetching {timeframe}m data: {e}")
        
        return pd.DataFrame()
    
    async def get_open_interest(self, instrument: str) -> Dict:
        """Fetch current Open Interest"""
        await self.init_session()
        
        url = f"{DERIBIT_API}/get_book_summary_by_instrument"
        params = {"instrument_name": instrument}
        
        try:
            async with self.session.get(url, params=params) as resp:
                data = await resp.json()
                if 'result' in data and len(data['result']) > 0:
                    result = data['result'][0]
                    return {
                        'open_interest': result.get('open_interest', 0),
                        'volume_usd': result.get('volume_usd', 0),
                        'last_price': result.get('last', 0),
                        'funding_rate': result.get('funding_8h', 0) * 100
                    }
        except Exception as e:
            print(f"Error fetching OI: {e}")
        
        return {'open_interest': 0, 'volume_usd': 0, 'last_price': 0, 'funding_rate': 0}
    
    async def get_liquidations(self, currency: str) -> Dict:
        """Get liquidation data"""
        await self.init_session()
        
        url = f"{DERIBIT_API}/get_last_trades_by_currency"
        params = {
            "currency": currency,
            "count": 100,
            "include_old": False
        }
        
        try:
            async with self.session.get(url, params=params) as resp:
                data = await resp.json()
                liquidations = {'long': 0, 'short': 0}
                
                if 'result' in data and 'trades' in data['result']:
                    for trade in data['result']['trades']:
                        if trade.get('liquidation'):
                            if trade['direction'] == 'sell':
                                liquidations['long'] += trade['amount']
                            else:
                                liquidations['short'] += trade['amount']
                
                return liquidations
        except Exception as e:
            print(f"Error fetching liquidations: {e}")
        
        return {'long': 0, 'short': 0}

# ==================== TECHNICAL ANALYSIS ====================
class TechnicalAnalyzer:
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[List, List]:
        """Calculate support and resistance levels"""
        if len(df) < window * 2:
            return [], []
        
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == highs.iloc[i]:
                resistance_levels.append(float(df['high'].iloc[i]))
            if df['low'].iloc[i] == lows.iloc[i]:
                support_levels.append(float(df['low'].iloc[i]))
        
        resistance_levels = TechnicalAnalyzer._cluster_levels(resistance_levels)[:3]
        support_levels = TechnicalAnalyzer._cluster_levels(support_levels)[:3]
        
        return support_levels, resistance_levels
    
    @staticmethod
    def _cluster_levels(levels: List[float], threshold: float = 0.005) -> List[float]:
        """Cluster nearby price levels"""
        if not levels:
            return []
        
        levels = sorted(set(levels))
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold:
                current_cluster.append(level)
            else:
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        
        if current_cluster:
            clustered.append(sum(current_cluster) / len(current_cluster))
        
        return clustered
    
    @staticmethod
    def detect_order_blocks(df: pd.DataFrame) -> Dict:
        """Detect bullish and bearish order blocks"""
        order_blocks = {'bullish': [], 'bearish': []}
        
        if len(df) < 5:
            return order_blocks
        
        for i in range(2, len(df) - 2):
            if (df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i] > df['high'].iloc[i-1] and
                df['volume'].iloc[i] > df['volume'].iloc[i-1] * 1.3):
                
                order_blocks['bullish'].append({
                    'index': i,
                    'high': float(df['high'].iloc[i-1]),
                    'low': float(df['low'].iloc[i-1]),
                    'price': float(df['close'].iloc[i])
                })
            
            if (df['close'].iloc[i] < df['open'].iloc[i] and
                df['close'].iloc[i] < df['low'].iloc[i-1] and
                df['volume'].iloc[i] > df['volume'].iloc[i-1] * 1.3):
                
                order_blocks['bearish'].append({
                    'index': i,
                    'high': float(df['high'].iloc[i-1]),
                    'low': float(df['low'].iloc[i-1]),
                    'price': float(df['close'].iloc[i])
                })
        
        return order_blocks
    
    @staticmethod
    def detect_fvg(df: pd.DataFrame) -> Dict:
        """Detect Fair Value Gaps"""
        fvgs = {'bullish': [], 'bearish': []}
        
        if len(df) < 3:
            return fvgs
        
        for i in range(1, len(df) - 1):
            if df['low'].iloc[i+1] > df['high'].iloc[i-1]:
                fvgs['bullish'].append({
                    'index': i,
                    'top': float(df['low'].iloc[i+1]),
                    'bottom': float(df['high'].iloc[i-1]),
                    'size': float(df['low'].iloc[i+1] - df['high'].iloc[i-1])
                })
            
            if df['high'].iloc[i+1] < df['low'].iloc[i-1]:
                fvgs['bearish'].append({
                    'index': i,
                    'top': float(df['low'].iloc[i-1]),
                    'bottom': float(df['high'].iloc[i+1]),
                    'size': float(df['low'].iloc[i-1] - df['high'].iloc[i+1])
                })
        
        return fvgs
    
    @staticmethod
    def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 20) -> Dict:
        """Detect liquidity sweeps"""
        sweeps = {'bullish': [], 'bearish': []}
        
        if len(df) < lookback + 5:
            return sweeps
        
        for i in range(lookback, len(df)):
            recent_high = df['high'].iloc[i-lookback:i].max()
            recent_low = df['low'].iloc[i-lookback:i].min()
            
            if (df['low'].iloc[i] < recent_low * 0.999 and
                df['close'].iloc[i] > df['open'].iloc[i] and
                df['close'].iloc[i] > recent_low):
                
                sweeps['bullish'].append({
                    'index': i,
                    'sweep_price': float(df['low'].iloc[i]),
                    'recovery_price': float(df['close'].iloc[i]),
                    'strength': float((df['close'].iloc[i] - df['low'].iloc[i]) / df['low'].iloc[i] * 100)
                })
            
            if (df['high'].iloc[i] > recent_high * 1.001 and
                df['close'].iloc[i] < df['open'].iloc[i] and
                df['close'].iloc[i] < recent_high):
                
                sweeps['bearish'].append({
                    'index': i,
                    'sweep_price': float(df['high'].iloc[i]),
                    'recovery_price': float(df['close'].iloc[i]),
                    'strength': float((df['high'].iloc[i] - df['close'].iloc[i]) / df['high'].iloc[i] * 100)
                })
        
        return sweeps
    
    @staticmethod
    def calculate_cvd(df: pd.DataFrame) -> pd.Series:
        """Calculate Cumulative Volume Delta"""
        delta = np.where(df['close'] >= df['open'], df['volume'], -df['volume'])
        cvd = pd.Series(delta).cumsum()
        return cvd
    
    @staticmethod
    def detect_bos_choch(df: pd.DataFrame) -> Dict:
        """Detect Break of Structure and Change of Character"""
        signals = {'bos': [], 'choch': []}
        
        if len(df) < 20:
            return signals
        
        window = 5
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(df) - window):
            if df['high'].iloc[i] == highs.iloc[i]:
                swing_highs.append({'index': i, 'price': float(df['high'].iloc[i])})
            if df['low'].iloc[i] == lows.iloc[i]:
                swing_lows.append({'index': i, 'price': float(df['low'].iloc[i])})
        
        if len(swing_highs) >= 2:
            last_high = swing_highs[-1]['price']
            prev_high = swing_highs[-2]['price']
            
            if df['close'].iloc[-1] > last_high and last_high > prev_high:
                signals['bos'].append({
                    'type': 'bullish',
                    'price': last_high,
                    'strength': 'strong'
                })
        
        if len(swing_lows) >= 2:
            last_low = swing_lows[-1]['price']
            prev_low = swing_lows[-2]['price']
            
            if df['close'].iloc[-1] < last_low and last_low < prev_low:
                signals['bos'].append({
                    'type': 'bearish',
                    'price': last_low,
                    'strength': 'strong'
                })
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 1:
            if df['close'].iloc[-1] < swing_lows[-1]['price']:
                signals['choch'].append({
                    'type': 'bearish_reversal',
                    'price': swing_lows[-1]['price']
                })
        
        return signals
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using ta library"""
        if len(df) < 200:
            print(f"⚠️ Insufficient data for indicators: {len(df)} candles")
            return df
        
        try:
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # MACD
            macd_indicator = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['macd'] = macd_indicator.macd()
            df['macd_signal'] = macd_indicator.macd_signal()
            df['macd_hist'] = macd_indicator.macd_diff()
            
            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb_indicator.bollinger_hband()
            df['bb_middle'] = bb_indicator.bollinger_mavg()
            df['bb_lower'] = bb_indicator.bollinger_lband()
            
            # ATR
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            
            # EMAs
            df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
            
            # Volume SMA
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
        
        return df

# ==================== REDIS MANAGER ====================
class RedisManager:
    @staticmethod
    def store_analysis(coin: str, timeframe: str, data: Dict):
        """Store analysis data in Redis"""
        if not redis_client:
            return
        
        try:
            key = f"analysis:{coin}:{timeframe}:{int(datetime.now().timestamp())}"
            redis_client.setex(key, 86400 * 7, json.dumps(data, default=str))
            
            latest_key = f"latest:{coin}:{timeframe}"
            redis_client.set(latest_key, json.dumps(data, default=str))
        except Exception as e:
            print(f"Redis store error: {e}")
    
    @staticmethod
    def get_previous_analysis(coin: str, timeframe: str) -> Dict:
        """Get previous analysis"""
        if not redis_client:
            return {}
        
        try:
            key = f"latest:{coin}:{timeframe}"
            data = redis_client.get(key)
            return json.loads(data) if data else {}
        except Exception as e:
            print(f"Redis get error: {e}")
            return {}
    
    @staticmethod
    def manage_memory():
        """Auto-delete old data"""
        if not redis_client:
            return
        
        try:
            info = redis_client.info('memory')
            used_memory = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            
            if max_memory > 0:
                usage_percent = (used_memory / max_memory) * 100
                
                if usage_percent > 50:
                    keys = redis_client.keys("analysis:*")
                    if keys:
                        keys_to_delete = sorted(keys)[:len(keys)//5]
                        if keys_to_delete:
                            redis_client.delete(*keys_to_delete)
                            print(f"🗑️ Deleted {len(keys_to_delete)} old keys")
        except Exception as e:
            print(f"Memory management error: {e}")

# ==================== MARKET ANALYZER ====================
class MarketAnalyzer:
    def __init__(self):
        self.fetcher = DeribitDataFetcher()
        self.analyzer = TechnicalAnalyzer()
    
    async def analyze_multi_timeframe(self, coin: str) -> Dict:
        """Multi-timeframe analysis"""
        await self.fetcher.init_session()
        
        instrument = f"{coin}-PERPETUAL"
        
        try:
            print(f"  📊 Fetching {coin} data...")
            df_30m = await self.fetcher.get_candlestick_data(instrument, "30", 500)
            await asyncio.sleep(0.5)
            df_1h = await self.fetcher.get_candlestick_data(instrument, "60", 300)
            await asyncio.sleep(0.5)
            df_4h = await self.fetcher.get_candlestick_data(instrument, "240", 200)
            
            if df_30m.empty or df_1h.empty or df_4h.empty:
                raise Exception("Failed to fetch candlestick data")
            
            print(f"  💰 Fetching OI & liquidations...")
            oi_data = await self.fetcher.get_open_interest(instrument)
            liq_data = await self.fetcher.get_liquidations(coin)
            
            print(f"  🔬 Calculating indicators...")
            df_30m = self.analyzer.calculate_indicators(df_30m)
            df_1h = self.analyzer.calculate_indicators(df_1h)
            df_4h = self.analyzer.calculate_indicators(df_4h)
            
            print(f"  🎯 Running SMC analysis...")
            support_30m, resistance_30m = self.analyzer.calculate_support_resistance(df_30m)
            order_blocks = self.analyzer.detect_order_blocks(df_30m)
            fvg = self.analyzer.detect_fvg(df_30m)
            liquidity_sweeps = self.analyzer.detect_liquidity_sweeps(df_30m)
            bos_choch = self.analyzer.detect_bos_choch(df_30m)
            cvd = self.analyzer.calculate_cvd(df_30m)
            
            # OI Analysis
            prev_oi = RedisManager.get_previous_analysis(coin, "30")
            oi_change = 0
            oi_sentiment = "NEUTRAL"
            
            if prev_oi and 'open_interest' in prev_oi:
                prev_oi_val = prev_oi['open_interest']
                curr_oi_val = oi_data['open_interest']
                
                if prev_oi_val > 0:
                    oi_change = ((curr_oi_val - prev_oi_val) / prev_oi_val) * 100
                    price_change = ((df_30m['close'].iloc[-1] - df_30m['close'].iloc[-2]) / df_30m['close'].iloc[-2]) * 100
                    
                    if oi_change > 0 and price_change > 0:
                        oi_sentiment = "STRONG BULLISH"
                    elif oi_change > 0 and price_change < 0:
                        oi_sentiment = "STRONG BEARISH"
                    elif oi_change < 0 and price_change > 0:
                        oi_sentiment = "WEAK BULLISH"
                    elif oi_change < 0 and price_change < 0:
                        oi_sentiment = "WEAK BEARISH"
            
            trend_4h = "BULLISH" if df_4h['ema_20'].iloc[-1] > df_4h['ema_50'].iloc[-1] else "BEARISH"
            trend_1h = "BULLISH" if df_1h['ema_20'].iloc[-1] > df_1h['ema_50'].iloc[-1] else "BEARISH"
            
            current_price = float(df_30m['close'].iloc[-1])
            high_swing = float(df_30m['high'].tail(50).max())
            low_swing = float(df_30m['low'].tail(50).min())
            mid_point = (high_swing + low_swing) / 2
            
            price_position = "PREMIUM" if current_price > mid_point else "DISCOUNT"
            
            analysis = {
                'coin': coin,
                'timestamp': datetime.now().isoformat(),
                'price': current_price,
                'open_interest': oi_data.get('open_interest', 0),
                'oi_change_percent': round(oi_change, 2),
                'oi_sentiment': oi_sentiment,
                'volume_24h': oi_data.get('volume_usd', 0),
                'funding_rate': round(oi_data.get('funding_rate', 0), 4),
                'liquidations': liq_data,
                'trend_4h': trend_4h,
                'trend_1h': trend_1h,
                'rsi_30m': float(df_30m['rsi'].iloc[-1]) if not pd.isna(df_30m['rsi'].iloc[-1]) else 50,
                'support_levels': support_30m,
                'resistance_levels': resistance_30m,
                'order_blocks': {
                    'bullish_count': len(order_blocks['bullish']),
                    'bearish_count': len(order_blocks['bearish']),
                    'bullish': order_blocks['bullish'][-3:],
                    'bearish': order_blocks['bearish'][-3:]
                },
                'fvg': {
                    'bullish_count': len(fvg['bullish']),
                    'bearish_count': len(fvg['bearish']),
                    'bullish': fvg['bullish'][-3:],
                    'bearish': fvg['bearish'][-3:]
                },
                'liquidity_sweeps': {
                    'bullish': liquidity_sweeps['bullish'][-2:],
                    'bearish': liquidity_sweeps['bearish'][-2:]
                },
                'bos_choch': bos_choch,
                'cvd_trend': 'BULLISH' if cvd.iloc[-1] > cvd.iloc[-10] else 'BEARISH',
                'price_position': price_position,
                'confluence_zones': self._find_confluence(support_30m, resistance_30m, order_blocks, fvg, current_price)
            }
            
            RedisManager.store_analysis(coin, "30", analysis)
            compact_data = self._prepare_compact_data(df_30m, analysis)
            
            return {
                'analysis': analysis,
                'compact_data': compact_data,
                'df_30m': df_30m
            }
            
        except Exception as e:
            print(f"❌ Analysis error for {coin}: {e}")
            traceback.print_exc()
            raise
        
        finally:
            await self.fetcher.close_session()
    
    def _find_confluence(self, support, resistance, order_blocks, fvg, current_price) -> List:
        """Find confluence zones"""
        zones = []
        
        for sup in support:
            factors = ['Support']
            
            for ob in order_blocks['bullish']:
                if abs(sup - ob['price']) / sup < 0.015:
                    factors.append('Bullish OB')
                    break
            
            for gap in fvg['bullish']:
                if sup >= gap['bottom'] and sup <= gap['top']:
                    factors.append('Bullish FVG')
                    break
            
            if len(factors) > 1:
                zones.append({
                    'type': 'BEARISH_CONFLUENCE',
                    'price': res,
                    'factors': factors,
                    'distance_percent': round(((res - current_price) / current_price) * 100, 2)
                })
        
        return zones
    
    def _prepare_compact_data(self, df: pd.DataFrame, analysis: Dict) -> str:
        """Prepare compact data for AI"""
        recent = df.tail(50)
        
        compact = {
            'candles': [
                {
                    'o': round(float(row['open']), 1),
                    'h': round(float(row['high']), 1),
                    'l': round(float(row['low']), 1),
                    'c': round(float(row['close']), 1),
                    'v': int(row['volume'])
                }
                for _, row in recent.iterrows()
            ],
            'price': analysis['price'],
            'oi_sent': analysis['oi_sentiment'],
            'trend_4h': analysis['trend_4h'],
            'trend_1h': analysis['trend_1h'],
            'rsi': round(analysis['rsi_30m'], 1),
            'support': [round(x, 1) for x in analysis['support_levels']],
            'resistance': [round(x, 1) for x in analysis['resistance_levels']],
            'ob_bull': analysis['order_blocks']['bullish_count'],
            'ob_bear': analysis['order_blocks']['bearish_count'],
            'fvg_bull': analysis['fvg']['bullish_count'],
            'fvg_bear': analysis['fvg']['bearish_count'],
            'cvd': analysis['cvd_trend'],
            'position': analysis['price_position'],
            'liq': analysis['liquidations']
        }
        
        return json.dumps(compact, separators=(',', ':'))

# ==================== AI ANALYZER ====================
class AIAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = DEEPSEEK_API_URL
    
    async def analyze_market(self, compact_data: str, python_analysis: Dict) -> TradingSignal:
        """Send data to DeepSeek V3 for AI analysis"""
        
        python_signal = self._python_pre_analysis(python_analysis)
        
        if python_signal.confidence < 40:
            print(f"  ⚠️ Low Python confidence: {python_signal.confidence}%")
            return python_signal
        
        prompt = self._create_analysis_prompt(compact_data, python_analysis, python_signal)
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "deepseek-chat",
                    "messages": [
                        {
                            "role": "system",
                            "content": """You are an expert crypto trader specializing in Smart Money Concepts. 
Analyze market data and provide clear trading signals. Focus on:
- Order Blocks, FVG, Liquidity Sweeps, BOS/ChOCh
- OI analysis (strong vs weak moves)
- Identify traps vs real breakouts/breakdowns
- Premium/Discount zones for optimal entry

Return ONLY valid JSON:
{
  "signal": "LONG" or "SHORT" or "NO_TRADE",
  "confidence": 0-100,
  "entry": price_number,
  "sl": price_number,
  "tp1": price_number,
  "tp2": price_number,
  "reason": "brief 2-3 sentence explanation"
}"""
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 400,
                    "response_format": {"type": "json_object"}
                }
                
                async with session.post(self.api_url, headers=headers, json=payload, timeout=30) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        ai_response = json.loads(result['choices'][0]['message']['content'])
                        
                        if not all(k in ai_response for k in ['signal', 'confidence']):
                            print("  ⚠️ Invalid AI response format")
                            return python_signal
                        
                        final_confidence = (python_signal.confidence * 0.4 + ai_response['confidence'] * 0.6)
                        
                        return TradingSignal(
                            signal=ai_response['signal'],
                            confidence=round(final_confidence, 2),
                            entry=ai_response.get('entry'),
                            sl=ai_response.get('sl'),
                            tp1=ai_response.get('tp1'),
                            tp2=ai_response.get('tp2'),
                            reason=ai_response.get('reason', ''),
                            python_score=python_signal.confidence,
                            ai_analysis=ai_response
                        )
                    else:
                        print(f"  ❌ AI API Error: {resp.status}")
                        return python_signal
                        
        except asyncio.TimeoutError:
            print("  ⏱️ AI API timeout")
            return python_signal
        except Exception as e:
            print(f"  ❌ AI Analysis Error: {e}")
            return python_signal
    
    def _python_pre_analysis(self, analysis: Dict) -> TradingSignal:
        """Python-based pre-filtering logic"""
        score = 0
        reasons = []
        signal_type = "NO_TRADE"
        
        if analysis['trend_4h'] == "BULLISH" and analysis['trend_1h'] == "BULLISH":
            score += 40
            signal_type = "LONG"
            reasons.append("HTF+LTF Bullish")
        elif analysis['trend_4h'] == "BEARISH" and analysis['trend_1h'] == "BEARISH":
            score += 40
            signal_type = "SHORT"
            reasons.append("HTF+LTF Bearish")
        else:
            reasons.append("No trend alignment")
            score += 10
        
        oi_sent = analysis['oi_sentiment']
        if oi_sent == "STRONG BULLISH" and signal_type == "LONG":
            score += 30
            reasons.append("Strong Bullish OI")
        elif oi_sent == "STRONG BEARISH" and signal_type == "SHORT":
            score += 30
            reasons.append("Strong Bearish OI")
        elif "WEAK" in oi_sent:
            score -= 15
            reasons.append("Weak OI")
        
        if analysis['price_position'] == "DISCOUNT" and signal_type == "LONG":
            score += 15
            reasons.append("Discount zone")
        elif analysis['price_position'] == "PREMIUM" and signal_type == "SHORT":
            score += 15
            reasons.append("Premium zone")
        
        rsi = analysis['rsi_30m']
        if signal_type == "LONG" and 30 < rsi < 50:
            score += 10
            reasons.append("RSI buy zone")
        elif signal_type == "SHORT" and 50 < rsi < 70:
            score += 10
            reasons.append("RSI sell zone")
        elif rsi > 75 or rsi < 25:
            score -= 15
            reasons.append("RSI extreme")
        
        smc_score = 0
        if analysis['order_blocks']['bullish_count'] > 0 and signal_type == "LONG":
            smc_score += 8
            reasons.append("Bullish OB")
        elif analysis['order_blocks']['bearish_count'] > 0 and signal_type == "SHORT":
            smc_score += 8
            reasons.append("Bearish OB")
        
        if analysis['fvg']['bullish_count'] > 0 and signal_type == "LONG":
            smc_score += 7
            reasons.append("Bullish FVG")
        elif analysis['fvg']['bearish_count'] > 0 and signal_type == "SHORT":
            smc_score += 7
            reasons.append("Bearish FVG")
        
        if analysis['bos_choch'].get('bos'):
            for bos in analysis['bos_choch']['bos']:
                if bos['type'] == 'bullish' and signal_type == "LONG":
                    smc_score += 5
                    reasons.append("Bullish BOS")
                elif bos['type'] == 'bearish' and signal_type == "SHORT":
                    smc_score += 5
                    reasons.append("Bearish BOS")
        
        score += smc_score
        
        confluence_bonus = 0
        for zone in analysis['confluence_zones']:
            if abs(zone['distance_percent']) < 2:
                if zone['type'] == 'BULLISH_CONFLUENCE' and signal_type == "LONG":
                    confluence_bonus = 15
                    reasons.append(f"Confluence @{zone['price']:.0f}")
                    break
                elif zone['type'] == 'BEARISH_CONFLUENCE' and signal_type == "SHORT":
                    confluence_bonus = 15
                    reasons.append(f"Confluence @{zone['price']:.0f}")
                    break
        
        score += confluence_bonus
        
        liq = analysis['liquidations']
        total_liq = liq['long'] + liq['short']
        
        if total_liq > 0:
            long_pct = (liq['long'] / total_liq) * 100
            
            if signal_type == "LONG" and long_pct > 65:
                score -= 20
                reasons.append("High long liq")
            elif signal_type == "SHORT" and long_pct < 35:
                score -= 20
                reasons.append("High short liq")
        
        if analysis['cvd_trend'] == "BULLISH" and signal_type == "LONG":
            score += 10
            reasons.append("CVD bullish")
        elif analysis['cvd_trend'] == "BEARISH" and signal_type == "SHORT":
            score += 10
            reasons.append("CVD bearish")
        
        score = min(100, max(0, score))
        
        return TradingSignal(
            signal=signal_type if score >= 60 else 'NO_TRADE',
            confidence=score,
            reason=', '.join(reasons[:5])
        )
    
    def _create_analysis_prompt(self, compact_data: str, analysis: Dict, python_signal: TradingSignal) -> str:
        """Create optimized prompt for AI"""
        
        prompt = f"""Analyze {analysis['coin']} market:

CANDLE DATA (50x30min): {compact_data}

PYTHON ANALYSIS:
Signal: {python_signal.signal} | Confidence: {python_signal.confidence}%
Factors: {python_signal.reason}

SMC CONTEXT:
OB: {analysis['order_blocks']['bullish_count']}B/{analysis['order_blocks']['bearish_count']}B
FVG: {analysis['fvg']['bullish_count']}B/{analysis['fvg']['bearish_count']}B
Position: {analysis['price_position']} | CVD: {analysis['cvd_trend']}

CONFLUENCE: {json.dumps(analysis['confluence_zones'][:2])}

Is this a trap or genuine move? Optimal entry zone? OI support? Provide JSON signal."""
        
        return prompt

# ==================== CHART GENERATOR ====================
class ChartGenerator:
    @staticmethod
    def create_signal_chart(df: pd.DataFrame, analysis: Dict, signal: TradingSignal, coin: str) -> str:
        """Generate trading chart with signal"""
        
        try:
            fig = make_subplots(
                rows=3, cols=1,
                row_heights=[0.6, 0.2, 0.2],
                subplot_titles=(f'{coin} - 30min', 'Volume', 'RSI'),
                vertical_spacing=0.05
            )
            
            df_plot = df.tail(100).copy()
            
            fig.add_trace(
                go.Candlestick(
                    x=df_plot['timestamp'],
                    open=df_plot['open'],
                    high=df_plot['high'],
                    low=df_plot['low'],
                    close=df_plot['close'],
                    name='Price',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ),
                row=1, col=1
            )
            
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['ema_20'], name='EMA20', line=dict(color='#2196F3', width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['ema_50'], name='EMA50', line=dict(color='#FF9800', width=1.5)), row=1, col=1)
            
            current_time = df_plot['timestamp'].iloc[-1]
            start_time = df_plot['timestamp'].iloc[0]
            
            for sup in analysis['support_levels'][:3]:
                fig.add_trace(go.Scatter(x=[start_time, current_time], y=[sup, sup], mode='lines', name=f'S:{sup:.0f}', line=dict(color='#4CAF50', width=2, dash='dash')), row=1, col=1)
            
            for res in analysis['resistance_levels'][:3]:
                fig.add_trace(go.Scatter(x=[start_time, current_time], y=[res, res], mode='lines', name=f'R:{res:.0f}', line=dict(color='#F44336', width=2, dash='dash')), row=1, col=1)
            
            if signal.signal in ['LONG', 'SHORT'] and signal.entry:
                fig.add_trace(go.Scatter(x=[start_time, current_time], y=[signal.entry, signal.entry], mode='lines', name='ENTRY', line=dict(color='#2196F3', width=3)), row=1, col=1)
                
                if signal.sl:
                    fig.add_trace(go.Scatter(x=[start_time, current_time], y=[signal.sl, signal.sl], mode='lines', name='SL', line=dict(color='#F44336', width=2.5)), row=1, col=1)
                if signal.tp1:
                    fig.add_trace(go.Scatter(x=[start_time, current_time], y=[signal.tp1, signal.tp1], mode='lines', name='TP1', line=dict(color='#8BC34A', width=2.5)), row=1, col=1)
                if signal.tp2:
                    fig.add_trace(go.Scatter(x=[start_time, current_time], y=[signal.tp2, signal.tp2], mode='lines', name='TP2', line=dict(color='#CDDC39', width=2, dash='dot')), row=1, col=1)
            
            colors = ['#ef5350' if df_plot['close'].iloc[i] < df_plot['open'].iloc[i] else '#26a69a' for i in range(len(df_plot))]
            fig.add_trace(go.Bar(x=df_plot['timestamp'], y=df_plot['volume'], name='Volume', marker_color=colors, showlegend=False, opacity=0.7), row=2, col=1)
            
            fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['rsi'], name='RSI', line=dict(color='#9C27B0', width=2), fill='tozeroy', showlegend=False), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#F44336", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#4CAF50", row=3, col=1)
            
            signal_emoji = "🟢 LONG" if signal.signal == "LONG" else "🔴 SHORT" if signal.signal == "SHORT" else "⚪ NO TRADE"
            
            fig.update_layout(
                title=f"{coin} Signal - {signal_emoji} | {signal.confidence}%",
                xaxis_rangeslider_visible=False,
                height=900,
                width=1400,
                template='plotly_white',
                showlegend=True,
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)'),
                margin=dict(l=60, r=60, t=80, b=60)
            )
            
            filename = f"/tmp/{coin}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            fig.write_image(filename, width=1400, height=900, scale=2)
            
            return filename
            
        except Exception as e:
            print(f"❌ Chart error: {e}")
            return None

# ==================== TELEGRAM BOT ====================
class TradingBot:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.bot = Bot(token=token)
        self.market_analyzer = MarketAnalyzer()
        self.ai_analyzer = AIAnalyzer(DEEPSEEK_API_KEY)
        self.chart_gen = ChartGenerator()
    
    async def send_signal(self, coin: str, analysis: Dict, signal: TradingSignal, chart_path: str):
        """Send trading signal to Telegram"""
        
        signal_emoji = "🟢" if signal.signal == "LONG" else "🔴" if signal.signal == "SHORT" else "⚪"
        rr_ratio = self._calculate_rr(signal)
        
        message = f"""{signal_emoji} **{coin} SIGNAL** {signal_emoji}

**Signal:** {signal.signal}
**Confidence:** {signal.confidence}%

**Market:**
Price: ${analysis['price']:.2f}
Trend 4H: {analysis['trend_4h']} | 1H: {analysis['trend_1h']}
RSI: {analysis['rsi_30m']:.1f} | {analysis['price_position']}

**OI:**
{analysis['open_interest']:,.0f} ({analysis['oi_change_percent']:+.2f}%)
Sentiment: {analysis['oi_sentiment']}
"""
        
        if signal.signal in ['LONG', 'SHORT'] and signal.entry:
            message += f"""
**Targets:**
Entry: ${signal.entry:.2f}
SL: ${signal.sl:.2f}
TP1: ${signal.tp1:.2f}
TP2: ${signal.tp2:.2f}
R:R: 1:{rr_ratio:.2f}
"""
        
        message += f"""
**SMC:**
OB: {analysis['order_blocks']['bullish_count']}🟢/{analysis['order_blocks']['bearish_count']}🔴
FVG: {analysis['fvg']['bullish_count']}🟢/{analysis['fvg']['bearish_count']}🔴
CVD: {analysis['cvd_trend']}

**Reason:** {signal.reason}

**Scores:** Py:{signal.python_score:.0f}% | AI:{signal.ai_analysis['confidence']:.0f}%
{datetime.now().strftime('%Y-%m-%d %H:%M')} UTC
"""
        
        try:
            if chart_path and os.path.exists(chart_path):
                with open(chart_path, 'rb') as photo:
                    await self.bot.send_photo(chat_id=self.chat_id, photo=photo, caption=message, parse_mode='Markdown')
            else:
                await self.bot.send_message(chat_id=self.chat_id, text=message, parse_mode='Markdown')
            
            print(f"  ✅ Signal sent for {coin}")
            
        except Exception as e:
            print(f"  ❌ Error sending: {e}")
    
    def _calculate_rr(self, signal: TradingSignal) -> float:
        if not (signal.entry and signal.sl and signal.tp1):
            return 0
        risk = abs(signal.entry - signal.sl)
        reward = abs(signal.tp1 - signal.entry)
        return reward / risk if risk > 0 else 0
    
    async def analyze_and_signal(self, coin: str):
        """Main analysis"""
        try:
            print(f"\n{'='*60}\n🔍 Analyzing {coin}...\n{'='*60}")
            
            result = await self.market_analyzer.analyze_multi_timeframe(coin)
            analysis = result['analysis']
            compact_data = result['compact_data']
            df = result['df_30m']
            
            print(f"  ✅ Data fetched")
            print(f"  🤖 Sending to DeepSeek V3...")
            
            signal = await self.ai_analyzer.analyze_market(compact_data, analysis)
            print(f"  📊 Signal: {signal.signal} | {signal.confidence}%")
            
            print(f"  📈 Generating chart...")
            chart_path = self.chart_gen.create_signal_chart(df, analysis, signal, coin)
            
            if signal.confidence >= 60 and signal.signal in ['LONG', 'SHORT']:
                print(f"  📤 Sending alert...")
                await self.send_signal(coin, analysis, signal, chart_path)
                print(f"  ✅ {coin} ALERT SENT!")
            else:
                print(f"  ⚠️ No trade (Low confidence)")
            
            if chart_path and os.path.exists(chart_path):
                os.remove(chart_path)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            traceback.print_exc()
    
    async def scheduled_analysis(self):
        """Run every 30 minutes"""
        print("\n" + "="*60)
        print("🚀 DERIBIT BOT STARTED")
        print("="*60)
        print(f"📡 Redis: {'Connected' if redis_client else 'Disabled'}")
        print(f"💬 Telegram: {self.chat_id}")
        print(f"⏰ Interval: 30 minutes")
        print(f"🪙 Coins: BTC, ETH")
        print("="*60 + "\n")
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text="🤖 *Bot Started!*\n\nAnalyzing BTC & ETH every 30min\nSignals sent when confidence ≥ 60%",
                parse_mode='Markdown'
            )
        except:
            pass
        
        while True:
            try:
                cycle_start = datetime.now()
                print(f"\n{'🔄 '*30}\nCycle - {cycle_start.strftime('%H:%M:%S')}\n{'🔄 '*30}\n")
                
                await self.analyze_and_signal("BTC")
                await asyncio.sleep(5)
                await self.analyze_and_signal("ETH")
                
                RedisManager.manage_memory()
                
                duration = (datetime.now() - cycle_start).total_seconds()
                print(f"\n{'✅ '*30}\nCompleted in {duration:.1f}s\nNext in 30min\n{'✅ '*30}\n")
                
                await asyncio.sleep(1800)
                
            except KeyboardInterrupt:
                print("\n👋 Stopped")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                await asyncio.sleep(60)

# ==================== MAIN ====================
async def main():
    bot = TradingBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    await bot.scheduled_analysis()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal: {e}")
        traceback.print_exc():
                zones.append({
                    'type': 'BULLISH_CONFLUENCE',
                    'price': sup,
                    'factors': factors,
                    'distance_percent': round(((sup - current_price) / current_price) * 100, 2)
                })
        
        for res in resistance:
            factors = ['Resistance']
            
            for ob in order_blocks['bearish']:
                if abs(res - ob['price']) / res < 0.015:
                    factors.append('Bearish OB')
                    break
            
            for gap in fvg['bearish']:
                if res >= gap['bottom'] and res <= gap['top']:
                    factors.append('Bearish FVG')
                    break
            
            if len(factors) > 1
