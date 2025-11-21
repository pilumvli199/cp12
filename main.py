#!/usr/bin/env python3
"""
DELTA EXCHANGE EXACT REPLICA BOT
================================
1. Dynamic Strike Steps (Matches Website exactly)
2. Range: ATM -12 to ATM +12 (25 Rows)
3. Data: IV, OI, Volume (USD)
"""

import os
import asyncio
import requests
from datetime import datetime
from telegram import Bot
import logging

# ==================== CONFIGURATION ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
BASE_URL = "https://api.india.delta.exchange"

# ==================== DELTA API CLIENT ====================
class DeltaExchangeClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        self.product_map = {}

    def get_products(self):
        """Fetch Expiry Dates Mapping"""
        try:
            url = f"{BASE_URL}/v2/products"
            params = {'contract_types': 'call_options,put_options', 'states': 'live'}
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json().get('result', [])
                new_map = {}
                for p in data:
                    settlement = p.get('settlement_time')
                    if settlement:
                        try:
                            dt = datetime.strptime(settlement, '%Y-%m-%dT%H:%M:%SZ')
                            new_map[p['symbol']] = dt
                        except:
                            continue
                self.product_map = new_map
                return self.product_map
            return {}
        except:
            return {}

    def get_tickers(self):
        try:
            url = f"{BASE_URL}/v2/tickers"
            response = self.session.get(url, timeout=10)
            return response.json().get('result', []) if response.status_code == 200 else []
        except:
            return []

    def get_market_data(self, underlying='BTC'):
        try:
            # 1. Fetch Data
            product_map = self.get_products() or self.product_map
            all_tickers = self.get_tickers()
            if not all_tickers: return None

            # 2. Spot Price
            perp_symbol = f"{underlying}USD"
            spot_ticker = next((t for t in all_tickers if t['symbol'] == perp_symbol), None)
            spot_price = float(spot_ticker.get('mark_price') or 0) if spot_ticker else 0

            # 3. Filter Tickers for Nearest Expiry
            options = []
            now = datetime.now()
            
            # Temporary list to find nearest expiry
            valid_expiries = set()
            
            for t in all_tickers:
                sym = t.get('symbol')
                if sym in product_map:
                    exp_dt = product_map[sym]
                    if exp_dt > now:
                        valid_expiries.add(exp_dt)

            if not valid_expiries: return None
            nearest_exp = sorted(list(valid_expiries))[0]
            exp_str = nearest_exp.strftime('%d-%m-%Y')

            # 4. Collect relevant options
            calls = {}
            puts = {}
            available_strikes = set()

            for t in all_tickers:
                sym = t.get('symbol')
                if sym not in product_map: continue
                if product_map[sym] != nearest_exp: continue

                try:
                    strike = float(t.get('strike_price', 0))
                    if strike == 0: continue
                    
                    available_strikes.add(strike)

                    # Extract Data
                    greeks = t.get('greeks', {}) or {}
                    iv = float(greeks.get('implied_volatility', 0) or 0)
                    if 0 < iv < 5: iv *= 100  # Fix percentage

                    # Turnover (Volume in $)
                    usd_vol = float(t.get('turnover', 0) or 0)
                    if usd_vol == 0:
                        usd_vol = float(t.get('volume',0)) * float(t.get('mark_price',0))

                    data = {
                        'ltp': float(t.get('mark_price', 0)),
                        'oi': float(t.get('oi', 0)),
                        'iv': iv,
                        'vol': usd_vol
                    }

                    if 'C' in sym or '_C_' in sym:
                        calls[strike] = data
                    elif 'P' in sym or '_P_' in sym:
                        puts[strike] = data
                except:
                    continue

            # 5. SMART SORTING (The Fix)
            # Instead of calculating math, we take ACTUAL strikes from API
            sorted_strikes = sorted(list(available_strikes))
            
            if not sorted_strikes: return None

            # Find index of strike closest to Spot Price
            closest_strike = min(sorted_strikes, key=lambda x: abs(x - spot_price))
            mid_index = sorted_strikes.index(closest_strike)

            # Slice list to show range (e.g., 12 above, 12 below)
            start_idx = max(0, mid_index - 10)
            end_idx = min(len(sorted_strikes), mid_index + 11)
            
            selected_strikes = sorted_strikes[start_idx:end_idx]

            # Calculate PCR based on visible range or total? Using Total for accuracy
            total_put_oi = sum(p['oi'] for p in puts.values())
            total_call_oi = sum(c['oi'] for c in calls.values())
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0

            return {
                'u': underlying, 
                'spot': spot_price, 
                'atm': closest_strike, 
                'exp': exp_str,
                'pcr': pcr,
                'strikes': selected_strikes,
                'calls': calls,
                'puts': puts
            }

        except Exception as e:
            logger.error(f"Logic Error: {e}")
            return None

# ==================== FORMATTER ====================
class TelegramFormatter:
    @staticmethod
    def fmt_num(num, is_price=False):
        if num is None: return "-"
        if num == 0: return "0"
        
        if is_price:
            # For prices < 100, show decimals, else int
            return f"{num:.1f}" if num < 500 else f"{num:.0f}"
            
        if num >= 1000000: return f"{num/1000000:.1f}m"
        if num >= 1000: return f"{num/1000:.0f}k"
        return f"{num:.0f}"

    def generate_message(self, data):
        if not data: return None
        
        u, spot, atm = data['u'], data['spot'], data['atm']
        strikes = data['strikes']
        
        trend = "Neutral ⚖️"
        if data['pcr'] > 1.0: trend = "Bullish 🟢"
        elif data['pcr'] < 0.65: trend = "Bearish 🔴"

        # Header
        msg = f"📊 <b>{u} LIVE CHAIN</b>\n"
        msg += f"🎯 Spot: <b>{spot:,.0f}</b>\n"
        msg += f"⚖️ PCR: <b>{data['pcr']:.2f}</b> ({trend})\n"
        msg += f"📅 Exp: <b>{data['exp']}</b>\n"
        
        msg += "─" * 32 + "\n"
        # Header matching screenshot logic roughly
        # IV | OI | LTP || LTP | OI | IV
        msg += "<b> CALLS (IV/OI)  |  PUTS (OI/IV)</b>\n"
        msg += "<code> IV   OI   LTP | LTP   OI   IV </code>\n"
        msg += "─" * 32 + "\n"

        for k in strikes:
            c = data['calls'].get(k, {})
            p = data['puts'].get(k, {})
            
            # Calls Data
            c_iv = f"{int(c.get('iv', 0))}" if c.get('iv') else "-"
            c_oi = self.fmt_num(c.get('oi', 0))
            c_ltp = self.fmt_num(c.get('ltp', 0), is_price=True)
            
            # Puts Data
            p_ltp = self.fmt_num(p.get('ltp', 0), is_price=True)
            p_oi = self.fmt_num(p.get('oi', 0))
            p_iv = f"{int(p.get('iv', 0))}" if p.get('iv') else "-"

            # ATM Marker
            marker = "🔹" if k == atm else "  "
            
            # Strike Label (Keep clean: 78000 -> 78k, 2600 -> 2600)
            if k >= 10000: # BTC
                st_lbl = f"{k/1000:.1f}k"
            else: # ETH
                st_lbl = f"{k:.0f}"

            # Row Construction
            # Space allocation: 
            # IV(3) OI(4) LTP(5) | LTP(5) OI(4) IV(3)
            row = f"{c_iv:>3} {c_oi:>4} {c_ltp:>5}|{p_ltp:<5} {p_oi:<4} {p_iv:<3}"
            
            msg += f"<code>{row}</code>{marker}<b>{st_lbl}</b>\n"

        msg += "─" * 32 + "\n"
        return msg

# ==================== RUNNER ====================
async def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN not set")
        return

    client = DeltaExchangeClient()
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    fmt = TelegramFormatter()
    
    client.get_products()
    logger.info("🚀 Exact Match Bot Started...")
    
    while True:
        try:
            # Fetch BTC
            data_btc = client.get_market_data('BTC')
            if data_btc:
                await bot.send_message(TELEGRAM_CHAT_ID, fmt.generate_message(data_btc), parse_mode='HTML')
                logger.info("✅ BTC Update")

            await asyncio.sleep(3)

            # Fetch ETH
            data_eth = client.get_market_data('ETH')
            if data_eth:
                await bot.send_message(TELEGRAM_CHAT_ID, fmt.generate_message(data_eth), parse_mode='HTML')
                logger.info("✅ ETH Update")

        except Exception as e:
            logger.error(f"⚠️ Main Loop Error: {e}")
        
        logger.info("💤 Sleeping 60s...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
