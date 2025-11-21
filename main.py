#!/usr/bin/env python3
"""
DELTA EXCHANGE OPTION CHAIN BOT (RAILWAY FIX)
=============================================
1. Fetches Product Metadata (For Expiry Dates)
2. Fetches Live Tickers (For Price/Vol)
3. Merges them accurately
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

# ENV Variables
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
        # Cache for product metadata to save API calls
        self.product_map = {}
        self.last_product_fetch = None

    def get_products(self):
        """Fetch static product data (Expiries)"""
        # Refresh cache every 1 hour
        if self.product_map and self.last_product_fetch:
            if (datetime.now() - self.last_product_fetch).total_seconds() < 3600:
                return self.product_map

        try:
            url = f"{BASE_URL}/v2/products"
            params = {'contract_types': 'call_options,put_options', 'states': 'live'}
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('result', [])
                # Map Symbol -> Expiry Date Object
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
                self.last_product_fetch = datetime.now()
                logger.info(f"📚 Loaded {len(new_map)} products (Metadata)")
                return self.product_map
            return {}
        except Exception as e:
            logger.error(f"❌ Product Fetch Error: {e}")
            return {}

    def get_tickers(self):
        """Fetch live prices"""
        try:
            url = f"{BASE_URL}/v2/tickers"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                return response.json().get('result', [])
            return []
        except Exception as e:
            logger.error(f"❌ Ticker Fetch Error: {e}")
            return []

    def get_option_chain_data(self, underlying='BTC'):
        try:
            # 1. Ensure we have metadata (Expiry dates)
            product_map = self.get_products()
            if not product_map:
                logger.error("❌ Could not fetch Product Metadata")
                return None

            # 2. Fetch Live Tickers
            all_tickers = self.get_tickers()
            if not all_tickers: 
                logger.error("❌ Could not fetch Tickers")
                return None

            # 3. Find Spot Price
            perp_symbol = f"{underlying}USD"
            spot_ticker = next((t for t in all_tickers if t['symbol'] == perp_symbol), None)
            if not spot_ticker: 
                logger.warning(f"⚠️ Spot symbol {perp_symbol} not found")
                return None
            
            spot_price = float(spot_ticker.get('mark_price') or 0)
            logger.info(f"💰 {underlying} Spot: {spot_price}")

            # 4. Merge Ticker Data with Expiry Data
            options_data = []
            now = datetime.now()

            for t in all_tickers:
                sym = t.get('symbol')
                # Check if this symbol exists in our Options Product Map
                if sym in product_map:
                    expiry_dt = product_map[sym]
                    # Only take future expiries
                    if expiry_dt > now:
                        t['expiry_dt'] = expiry_dt # Inject expiry into ticker
                        options_data.append(t)

            logger.info(f"✅ Found {len(options_data)} valid future options for {underlying}")

            if not options_data:
                return None

            # 5. Find Nearest Expiry
            # Get unique dates
            unique_dates = sorted(list(set(o['expiry_dt'] for o in options_data)))
            if not unique_dates: return None
            
            nearest_expiry_dt = unique_dates[0]
            nearest_expiry_str = nearest_expiry_dt.strftime('%d-%m-%Y')
            
            logger.info(f"📅 Selected Expiry: {nearest_expiry_str}")

            # 6. Organize Strikes
            calls, puts = {}, {}
            
            for opt in options_data:
                if opt['expiry_dt'] != nearest_expiry_dt:
                    continue
                
                try:
                    # Note: v2/tickers usually has strike_price? If not, need to parse or get from product
                    # But usually tickers have it. If fails, we might need product specs.
                    strike = float(opt.get('strike_price', 0))
                    if strike == 0: continue # Skip invalid strikes

                    # Contract Type Detection
                    c_type = opt.get('contract_type', '').lower()
                    # Fallback if contract_type is missing in ticker, guess by symbol usually ends in C or P
                    is_call = 'call' in c_type or opt['symbol'].endswith('_C') or '-C' in opt['symbol']
                    is_put = 'put' in c_type or opt['symbol'].endswith('_P') or '-P' in opt['symbol']

                    data = {
                        'symbol': opt['symbol'],
                        'ltp': float(opt.get('mark_price', 0) or 0),
                        'oi': float(opt.get('oi', 0) or 0),
                        'volume': float(opt.get('volume', 0) or 0)
                    }
                    
                    if is_call: calls[strike] = data
                    elif is_put: puts[strike] = data

                except Exception as e:
                    continue

            # 7. Filter Range around ATM
            atm = self.round_to_strike(spot_price, underlying)
            strikes = self.get_strike_range(atm, underlying, count=5)
            
            return {
                'u': underlying, 'spot': spot_price, 'atm': atm, 'exp': nearest_expiry_str,
                'calls': {k: calls[k] for k in strikes if k in calls},
                'puts': {k: puts[k] for k in strikes if k in puts}
            }

        except Exception as e:
            logger.error(f"❌ Logic Error: {e}")
            return None

    def round_to_strike(self, price, underlying):
        step = 1000 if underlying == 'BTC' else 100
        return round(price / step) * step

    def get_strike_range(self, atm, underlying, count=5):
        step = 1000 if underlying == 'BTC' else 100
        return [atm + (i * step) for i in range(-count, count + 1)]

# ==================== FORMATTER ====================
class TelegramFormatter:
    @staticmethod
    def num_fmt(num):
        if num >= 1000000: return f"{num/1000000:.1f}M"
        if num >= 1000: return f"{num/1000:.0f}K"
        return f"{num:.0f}"

    def generate_message(self, data):
        if not data: return None
        
        u, spot, atm, exp = data['u'], data['spot'], data['atm'], data['exp']
        calls, puts = data['calls'], data['puts']

        msg = f"🔔 <b>{u} CHAIN (Vol & OI)</b>\n"
        msg += f"🎯 Spot: <b>{spot:,.0f}</b> | ATM: <b>{atm:,.0f}</b>\n"
        msg += f"📅 Exp: <b>{exp}</b>\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "<b>  CALLS (Vol/OI)   |   PUTS (Vol/OI)</b>\n"
        msg += "<code> Vol   OI  LTP | LTP  OI   Vol </code>\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━\n"

        strikes = sorted(set(list(calls.keys()) + list(puts.keys())))
        
        for k in strikes:
            c = calls.get(k, {})
            p = puts.get(k, {})
            
            c_v = self.num_fmt(c.get('volume', 0))
            c_o = self.num_fmt(c.get('oi', 0))
            c_l = f"{c.get('ltp',0):.0f}" if c.get('ltp') else "-"
            
            p_l = f"{p.get('ltp',0):.0f}" if p.get('ltp') else "-"
            p_o = self.num_fmt(p.get('oi', 0))
            p_v = self.num_fmt(p.get('volume', 0))

            marker = "🔹" if k == atm else "  "
            strike_lbl = f"{k/1000:.1f}k" if u == 'BTC' else f"{k:.0f}"

            row = f"{c_v:>3} {c_o:>3} {c_l:>4}|{p_l:<4} {p_o:<3} {p_v:<3}"
            msg += f"<code>{row}</code> {marker}<b>{strike_lbl}</b>\n"

        msg += "━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "<i>Values in K (Thousands) or M (Millions)</i>"
        return msg

# ==================== RUNNER ====================
async def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN not set")
        return

    client = DeltaExchangeClient()
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    fmt = TelegramFormatter()

    logger.info("🚀 Bot Started (Railway Optimized)...")
    
    # Initial fetch
    client.get_products()
    
    while True:
        try:
            # BTC
            data = client.get_option_chain_data('BTC')
            if data:
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID, 
                    text=fmt.generate_message(data), 
                    parse_mode='HTML'
                )
                logger.info("✅ BTC Update Sent")
            else:
                logger.warning("⚠️ Still no data for BTC (Check Logs above)")

            await asyncio.sleep(5)

            # ETH
            data = client.get_option_chain_data('ETH')
            if data:
                await bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID, 
                    text=fmt.generate_message(data), 
                    parse_mode='HTML'
                )
                logger.info("✅ ETH Update Sent")
            
        except Exception as e:
            logger.error(f"⚠️ Main Loop Error: {e}")
        
        logger.info("💤 Sleeping 60s...")
        await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
