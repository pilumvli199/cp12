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
except Exception as e:
    print(f"Redis connection failed: {e}")
    redis_client = None

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
    
    async def init_session(self):
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_candlestick_data(self, instrument: str, timeframe: str, count: int = 500) -> pd.DataFrame:
        await self.init_session()
        resolution_map = {"15": 15, "30": 30, "60": 60, "240": 240}
        resolution = resolution_map.get(timeframe, 30)
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = end_time - (count * resolution * 60 * 1000)
        url = f"{DERIBIT_API}/get_tradingview_chart_data"
        params = {"instrument_name": instrument, "start_timestamp": start_time, "end_timestamp": end_time, "resolution": resolution}
        try:
            async with self.session.get(url, params=params) as resp:
                data = await resp.json()
                if data.get('result', {}).get('status') == 'ok':
                    result = data['result']
                    df = pd.DataFrame({'timestamp': result['ticks'], 'open': result['open'], 'high': result['high'], 'low': result['low'], 'close': result['close'], 'volume': result['volume']})
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    return df
        except Exception as e:
            print(f"Error fetching data: {e}")
        return pd.DataFrame()
    
    async def get_open_interest(self, instrument: str) -> Dict:
        await self.init_session()
        url = f"{DERIBIT_API}/get_book_summary_by_instrument"
        params = {"instrument_name": instrument}
        try:
            async with self.session.get(url, params=params) as resp:
                data = await resp.json()
                if 'result' in data and len(data['result']) > 0:
                    result = data['result'][0]
                    return {'open_interest': result.get('open_interest', 0), 'volume_usd': result.get('volume_usd', 0), 'last_price': result.get('last', 0), 'funding_rate': result.get('funding_8h', 0) * 100}
        except Exception as e:
            print(f"Error fetching OI: {e}")
        return {'open_interest': 0, 'volume_usd': 0, 'last_price': 0, 'funding_rate': 0}
    
    async def get_liquidations(self, currency: str) -> Dict:
        await self.init_session()
        url = f"{DERIBIT_API}/get_last_trades_by_currency"
        params = {"currency": currency, "count": 100, "include_old": False}
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

class TechnicalAnalyzer:
    @staticmethod
    def calculate_support_resistance(df: pd.DataFrame, window: int = 20) -> Tuple[List, List]:
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
        order_blocks = {'bullish': [], 'bearish': []}
        if len(df) < 5:
            return order_blocks
        for i in range(2, len(df) - 2):
            if (df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i] > df['high'].iloc[i-1] and df['volume'].iloc[i] > df['volume'].iloc[i-1] * 1.3):
                order_blocks['bullish'].append({'index': i, 'high': float(df['high'].iloc[i-1]), 'low': float(df['low'].iloc[i-1]), 'price': float(df['close'].iloc[i])})
            if (df['close'].iloc[i] < df['open'].iloc[i] and df['close'].iloc[i] < df['low'].iloc[i-1] and df['volume'].iloc[i] > df['volume'].iloc[i-1] * 1.3):
                order_blocks['bearish'].append({'index': i, 'high': float(df['high'].iloc[i-1]), 'low': float(df['low'].iloc[i-1]), 'price': float(df['close'].iloc[i])})
        return order_blocks
    
    @staticmethod
    def detect_fvg(df: pd.DataFrame) -> Dict:
        fvgs = {'bullish': [], 'bearish': []}
        if len(df) < 3:
            return fvgs
        for i in range(1, len(df) - 1):
            if df['low'].iloc[i+1] > df['high'].iloc[i-1]:
                fvgs['bullish'].append({'index': i, 'top': float(df['low'].iloc[i+1]), 'bottom': float(df['high'].iloc[i-1]), 'size': float(df['low'].iloc[i+1] - df['high'].iloc[i-1])})
            if df['high'].iloc[i+1] < df['low'].iloc[i-1]:
                fvgs['bearish'].append({'index': i, 'top': float(df['low'].iloc[i-1]), 'bottom': float(df['high'].iloc[i+1]), 'size': float(df['low'].iloc[i-1] - df['high'].iloc[i+1])})
        return fvgs
    
    @staticmethod
    def detect_liquidity_sweeps(df: pd.DataFrame, lookback: int = 20) -> Dict:
        sweeps = {'bullish': [], 'bearish': []}
        if len(df) < lookback + 5:
            return sweeps
        for i in range(lookback, len(df)):
            recent_high = df['high'].iloc[i-lookback:i].max()
            recent_low = df['low'].iloc[i-lookback:i].min()
            if (df['low'].iloc[i] < recent_low * 0.999 and df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i] > recent_low):
                sweeps['bullish'].append({'index': i, 'sweep_price': float(df['low'].iloc[i]), 'recovery_price': float(df['close'].iloc[i]), 'strength': float((df['close'].iloc[i] - df['low'].iloc[i]) / df['low'].iloc[i] * 100)})
            if (df['high'].iloc[i] > recent_high * 1.001 and df['close'].iloc[i] < df['open'].iloc[i] and df['close'].iloc[i] < recent_high):
                sweeps['bearish'].append({'index': i, 'sweep_price': float(df['high'].iloc[i]), 'recovery_price': float(df['close'].iloc[i]), 'strength': float((df['high'].iloc[i] - df['close'].iloc[i]) / df['high'].iloc[i] * 100)})
        return sweeps
    
    @staticmethod
    def calculate_cvd(df: pd.DataFrame) -> pd.Series:
        delta = np.where(df['close'] >= df['open'], df['volume'], -df['volume'])
        cvd = pd.Series(delta).cumsum()
        return cvd
    
    @staticmethod
    def detect_bos_choch(df: pd.DataFrame) -> Dict:
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
                signals['bos'].append({'type': 'bullish', 'price': last_high, 'strength': 'strong'})
        if len(swing_lows) >= 2:
            last_low = swing_lows[-1]['price']
            prev_low = swing_lows[-2]['price']
            if df['close'].iloc[-1] < last_low and last_low < prev_low:
                signals['bos'].append({'type': 'bearish', 'price': last_low, 'strength': 'strong'})
        if len(swing_highs) >= 2 and len(swing_lows) >= 1:
            if df['close'].iloc[-1] < swing_lows[-1]['price']:
                signals['choch'].append({'type': 'bearish_reversal', 'price': swing_lows[-1]['price']})
        return signals
    
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) < 200:
            return df
        try:
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            macd_ind = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['macd'] = macd_ind.macd()
            df['macd_signal'] = macd_ind.macd_signal()
            df['macd_hist'] = macd_ind.macd_diff()
            bb_ind = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb_ind.bollinger_hband()
            df['bb_middle'] = bb_ind.bollinger_mavg()
            df['bb_lower'] = bb_ind.bollinger_lband()
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
            df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            df['ema_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
        except Exception as e:
            print(f"Indicator error: {e}")
        return df

class RedisManager:
    @staticmethod
    def store_analysis(coin: str, timeframe: str, data: Dict):
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
        if not redis_client:
            return {}
        try:
            key = f"latest:{coin}:{timeframe}"
            data = redis_client.get(key)
            return json.loads(data) if data else {}
        except Exception as e:
            return {}
    
    @staticmethod
    def manage_memory():
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
        except Exception as e:
            print(f"Memory error: {e}")

class MarketAnalyzer:
    def __init__(self):
        self.fetcher = DeribitDataFetcher()
        self.analyzer = TechnicalAnalyzer()
    
    async def analyze_multi_timeframe(self, coin: str) -> Dict:
        await self.fetcher.init_session()
        instrument = f"{coin}-PERPETUAL"
        try:
            df_30m = await self.fetcher.get_candlestick_data(instrument, "30", 500)
            await asyncio.sleep(0.5)
            df_1h = await self.fetcher.get_candlestick_data(instrument, "60", 300)
            await asyncio.sleep(0.5)
            df_4h = await self.fetcher.get_candlestick_data(instrument, "240", 200)
            if df_30m.empty or df_1h.empty or df_4h.empty:
                raise Exception("Failed to fetch data")
            oi_data = await self.fetcher.get_open_interest(instrument)
            liq_data = await self.fetcher.get_liquidations(coin)
            df_30m = self.analyzer.calculate_indicators(df_30m)
            df_1h = self.analyzer.calculate_indicators(df_1h)
            df_4h = self.analyzer.calculate_indicators(df_4h)
            support_30m, resistance_30m = self.analyzer.calculate_support_resistance(df_30m)
            order_blocks = self.analyzer.detect_order_blocks(df_30m)
            fvg = self.analyzer.detect_fvg(df_30m)
            liquidity_sweeps = self.analyzer.detect_liquidity_sweeps(df_30m)
            bos_choch = self.analyzer.detect_bos_choch(df_30m)
            cvd = self.analyzer.calculate_cvd(df_30m)
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
            return {'analysis': analysis, 'compact_data': compact_data, 'df_30m': df_30m}
        except Exception as e:
            print(f"Analysis error: {e}")
            raise
        finally:
            await self.fetcher.close_session()
    
    def _find_confluence(self, support, resistance, order_blocks, fvg, current_price) -> List:
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
                zones.append({'type': 'BULLISH_CONFLUENCE', 'price': sup, 'factors': factors, 'distance_percent': round(((sup - current_price) / current_price) * 100, 2)})
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
            if len(factors) > 1:
                zones.append({'type': 'BEARISH_CONFLUENCE', 'price': res, 'factors': factors, 'distance_percent': round(((res - current_price) / current_price) * 100, 2)})
        return zones
    
    def _prepare_compact_data(self, df: pd.DataFrame, analysis: Dict) -> str:
        recent = df.tail(50)
        compact = {
            'candles': [{'o': round(float(row['open']), 1), 'h': round(float(row['high']), 1), 'l': round(float(row['low']), 1), 'c': round(float(row['close']), 1), 'v': int(row['volume'])} for _, row in recent.iterrows()],
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

class AIAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = DEEPSEEK_API_URL
    
    async def analyze_market(self, compact_data: str, python_analysis: Dict) -> TradingSignal:
        python_signal = self._python_pre_analysis(python_analysis)
        if python_signal.confidence < 40:
            return python_signal
        prompt = f"""Analyze {python_analysis['coin']}: {compact_data}
PYTHON: {python_signal.signal} ({python_signal.confidence}%)
Provide JSON: {{"signal":"LONG/SHORT/NO_TRADE","confidence":0-100,"entry":price,"sl":price,"tp1":price,"tp2":price,"reason":"text"}}"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                payload = {"model": "deepseek-chat", "messages": [{"role": "system", "content": "Expert crypto trader. Return JSON signal."}, {"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 400, "response_format": {"type": "json_object"}}
                async with session.post(self.api_url, headers=headers, json=payload, timeout=30) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        ai_response = json.loads(result['choices'][0]['message']['content'])
                        if not all(k in ai_response for k in ['signal', 'confidence']):
                            return python_signal
                        final_confidence = (python_signal.confidence * 0.4 + ai_response['confidence'] * 0.6)
                        return TradingSignal(signal=ai_response['signal'], confidence=round(final_confidence, 2), entry=ai_response.get('entry'), sl=ai_response.get('sl'), tp1=ai_response.get('tp1'), tp2=ai_response.get('tp2'), reason=ai_response.get('reason', ''), python_score=python_signal.confidence, ai_analysis=ai_response)
                    else:
                        return python_signal
        except Exception as e:
            print(f"AI error: {e}")
            return python_signal
    
    def _python_pre_analysis(self, analysis: Dict) -> TradingSignal:
        score = 0
        reasons = []
        signal_type = "NO_TRADE"
        if analysis['trend_4h'] == "BULLISH" and analysis['trend_1h'] == "BULLISH":
            score += 40
            signal_type = "LONG"
            reasons.append("Trend aligned")
        elif analysis['trend_4h'] == "BEARISH" and analysis['trend_1h'] == "BEARISH":
            score += 40
            signal_type = "SHORT"
            reasons.append("Trend aligned")
        else:
            score += 10
        oi_sent = analysis['oi_sentiment']
        if oi_sent == "STRONG BULLISH" and signal_type == "LONG":
            score += 30
            reasons.append("Strong OI")
        elif oi_sent == "STRONG BEARISH" and signal_type == "SHORT":
            score += 30
            reasons.append("Strong OI")
        elif "WEAK" in oi_sent:
            score -= 15
        if analysis['price_position'] == "DISCOUNT" and signal_type == "LONG":
            score += 15
            reasons.append("Discount zone")
        elif analysis['price_position'] == "PREMIUM" and signal_type == "SHORT":
            score += 15
            reasons.append("Premium zone")
        rsi = analysis['rsi_30m']
        if signal_type == "LONG" and 30 < rsi < 50:
            score += 10
        elif signal_type == "SHORT" and 50 < rsi < 70:
            score += 10
        elif rsi > 75 or rsi < 25:
            score -= 15
        if analysis['order_blocks']['bullish_count'] > 0 and signal_type == "LONG":
            score += 8
            reasons.append("OB")
        elif analysis['order_blocks']['bearish_count'] > 0 and signal_type == "SHORT":
            score += 8
            reasons.append("OB")
        if analysis['fvg']['bullish_count'] > 0 and signal_type == "LONG":
            score += 7
        elif analysis['fvg']['bearish_count'] > 0 and signal_type == "SHORT":
