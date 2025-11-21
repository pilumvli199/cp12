#!/usr/bin/env python3
"""
DELTA EXCHANGE OPTION CHAIN BOT (OPTIMIZED)
===========================================
Efficient Bulk Data Fetching
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

# Environment Variables
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Delta Exchange API Base URL
BASE_URL = "https://api.india.delta.exchange"

# ==================== DELTA API CLIENT ====================
class DeltaExchangeClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

    def get_bulk_tickers(self):
        """Fetch ALL tickers in ONE request to avoid rate limits"""
        try:
            url = f"{BASE_URL}/v2/tickers"
            # No parameters means fetch everything (Spot, Futures, Options)
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data.get('result', [])
            return []
        except Exception as e:
            logger.error(f"❌ Bulk Ticker Fetch Error: {e}")
            return []

    def parse_expiry(self, symbol_info):
        """Extract expiry date object from symbol info safely"""
        try:
            # Symbol format usually: C-BTC-211125-90000
            # But relying on settlement_time is safer if available
            settlement = symbol_info.get('settlement_time')
            if settlement:
                return datetime.strptime(settlement, '%Y-%m-%dT%H:%M:%SZ')
        except:
            pass
        return None

    def get_option_chain_data(self, underlying='BTC'):
        """Get chain data by processing bulk tickers locally"""
        try:
            # 1. Get ALL data in one go
            all_tickers = self.get_bulk_tickers()
            if not all_tickers:
                return None

            logger.info(f"📦 Processed {len(all_tickers)} total market tickers")

            # 2. Find Spot Price (Mark Price of Perpetual)
            perp_symbol = f"{underlying}USD" # e.g. BTCUSD
            spot_ticker = next((t for t in all_tickers if t['symbol'] == perp_symbol), None)
            
            if not spot_ticker:
                logger.error(f"❌ Spot price not found for {underlying}")
                return None

            spot_price = float(spot_ticker.get('mark_price') or spot_ticker.get('close', 0))
            logger.info(f"💰 {underlying} Spot: {spot_price}")

            # 3. Filter only Options for this Underlying
            # Option symbols usually look like "C-BTC-..." or contain contract_type check
            options_tickers = [
                t for t in all_tickers 
                if t.get('contract_type') in ['call_options', 'put_options'] 
                and t.get('underlying_asset_symbol') == underlying
            ]

            if not options_tickers:
                return None

            # 4. Find Nearest Expiry
            now = datetime.now()
            expiry_map = {} # Date string -> Date Object

            for opt in options_tickers:
                exp_dt = self.parse_expiry(opt)
                # Adjust for IST (roughly check if future)
                if exp_dt and exp_dt > now:
                    date_str = exp_dt.strftime('%d-%m-%Y')
                    expiry_map[date_str] = exp_dt

            if not expiry_map:
                return None

            # Sort expiries and pick nearest
            sorted_expiries = sorted(expiry_map.items(), key=lambda x: x[1])
            nearest_expiry_str = sorted_expiries[0][0]
            
            logger.info(f"📅 Selected Expiry: {nearest_expiry_str}")

            # 5. Filter options for this specific expiry
            active_options = []
            for opt in options_tickers:
                exp_dt = self.parse_expiry(opt)
                if exp_dt and exp_dt.strftime('%d-%m-%Y') == nearest_expiry_str:
                    active_options.append(opt)

            # 6. Organize into Strikes
            calls = {}
            puts = {}
            
            for opt in active_options:
                try:
                    strike = float(opt.get('strike_price', 0))
                    quotes = opt.get('quotes', {})
                    
                    data = {
                        'symbol': opt['symbol'],
                        'strike': strike,
                        'bid': float(quotes.get('best_bid', 0) or 0),
                        'ask': float(quotes.get('best_ask', 0) or 0),
                        'oi': float(opt.get('oi', 0) or 0),
                        'volume': float(opt.get('volume', 0) or 0)
                    }

                    if opt.get('contract_type') == 'call_options':
                        calls[strike] = data
                    else:
                        puts[strike] = data
                except:
                    continue

            # 7. Calculate ATM and Range
            atm_strike = self.round_to_strike(spot_price, underlying)
            
            # Filter relevant strikes (e.g., 5 up, 5 down)
            relevant_strikes = self.get_strike_range(atm_strike, underlying, count=6)
            
            final_calls = {k: v for k, v in calls.items() if k in relevant_strikes}
            final_puts = {k: v for k, v in puts.items() if k in relevant_strikes}

            return {
                'underlying': underlying,
                'spot_price': spot_price,
                'atm_strike': atm_strike,
                'expiry_date': nearest_expiry_str,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'calls': final_calls,
                'puts': final_puts
            }

        except Exception as e:
            logger.error(f"❌ Chain Logic Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def round_to_strike(self, price, underlying):
        step = 1000 if underlying == 'BTC' else 100
        return round(price / step) * step

    def get_strike_range(self, atm, underlying, count=6):
        step = 1000 if underlying == 'BTC' else 100
        strikes = []
        for i in range(-count, count + 1):
            strikes.append(atm + (i * step))
        return strikes

# ==================== TELEGRAM FORMATTER ====================
class TelegramFormatter:
    @staticmethod
    def format_large(num):
        if num >= 1000: return f"{num/1000:.1f}K"
        return f"{num:.0f}"

    def generate_message(self, data):
        if not data: return "❌ Data fetch failed."

        u, spot, atm = data['underlying'], data['spot_price'], data['atm_strike']
        exp, time = data['expiry_date'], data['timestamp']
        calls, puts = data['calls'], data['puts']

        # Header
        msg = f"🔔 <b>{u} OPTION CHAIN</b>\n\n"
        msg += f"💰 Spot: <b>${spot:,.2f}</b>\n"
        msg += f"🎯 ATM: <b>${atm:,.0f}</b>\n"
        msg += f"📅 Exp: <b>{exp}</b>\n"
        msg += f"🕒 {time}\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += "<b>  CALLS       |    PUTS</b>\n"
        msg += "<code>Bid  Ask  OI | Bid  Ask  OI</code>\n"
        msg += "━━━━━━━━━━━━━━━━━━━━━━\n"

        # Table Rows
        strikes = sorted(set(list(calls.keys()) + list(puts.keys())))
        
        for k in strikes:
            c = calls.get(k, {})
            p = puts.get(k, {})
            
            # Formatting
            c_b = f"{c.get('bid',0):.0f}" if c.get('bid') else "-"
            c_a = f"{c.get('ask',0):.0f}" if c.get('ask') else "-"
            c_oi = self.format_large(c.get('oi', 0))
            
            p_b = f"{p.get('bid',0):.0f}" if p.get('bid') else "-"
            p_a = f"{p.get('ask',0):.0f}" if p.get('ask') else "-"
            p_oi = self.format_large(p.get('oi', 0))

            # Strike Label
            st_lbl = f"{k/1000:.0f}K" if u == 'BTC' else f"{k:.0f}"
            marker = "🎯" if k == atm else "  "

            row = f"{c_b:>3} {c_a:>3} {c_oi:>3}|{p_b:>3} {p_a:>3} {p_oi:>3} {marker}{st_lbl}"
            msg += f"<code>{row}</code>\n"

        # Summary
        c_tot = sum(c.get('oi',0) for c in calls.values())
        p_tot = sum(p.get('oi',0) for p in puts.values())
        pcr = p_tot / c_tot if c_tot > 0 else 0

        msg += "\n━━━━━━━━━━━━━━━━━━━━━━\n"
        msg += f"📊 <b>PCR: {pcr:.2f}</b> | Call: {self.format_large(c_tot)} | Put: {self.format_large(p_tot)}"
        
        return msg

# ==================== BOT RUNNER ====================
class BotRunner:
    def __init__(self):
        self.client = DeltaExchangeClient()
        self.bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.fmt = TelegramFormatter()

    async def send_update(self, underlying):
        data = self.client.get_option_chain_data(underlying)
        if data:
            msg = self.fmt.generate_message(data)
            try:
                await self.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID, 
                    text=msg, 
                    parse_mode='HTML'
                )
                logger.info(f"✅ Sent {underlying} update")
            except Exception as e:
                logger.error(f"⚠️ Telegram Error: {e}")
        else:
            logger.warning(f"⚠️ No data generated for {underlying}")

    async def run(self):
        logger.info("🚀 Bot Started...")
        await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="🤖 <b>Delta Bot Active</b>\nFetching Complete Chain...", parse_mode='HTML')
        
        while True:
            await self.send_update('BTC')
            await asyncio.sleep(5) # Short gap
            await self.send_update('ETH')
            
            logger.info("💤 Sleeping 60s...")
            await asyncio.sleep(60)

if __name__ == "__main__":
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("❌ ENV Variables Missing!")
        exit()
        
    try:
        runner = BotRunner()
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        print("🛑 Stopped")
