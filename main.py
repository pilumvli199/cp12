#!/usr/bin/env python3
"""
DELTA EXCHANGE OPTION CHAIN BOT
================================
Complete option chain data with proper API usage
Auto nearest expiry selection
"""

import os
import time
import asyncio
import requests
from datetime import datetime, timedelta
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
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    def get_ticker(self, symbol):
        """Get ticker for a symbol"""
        try:
            url = f"{BASE_URL}/v2/tickers/{symbol}"
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                json_data = response.json()
                if json_data.get('success'):
                    return json_data.get('result', {})
            return None
        except Exception as e:
            logger.error(f"❌ Ticker error {symbol}: {e}")
            return None
    
    def get_all_products(self, underlying='BTC'):
        """Get all products for expiry detection"""
        try:
            url = f"{BASE_URL}/v2/products"
            params = {
                'contract_types': 'call_options,put_options',
                'states': 'live',
                'page_size': 500
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code != 200:
                return []
            
            json_data = response.json()
            if not json_data.get('success'):
                return []
            
            products = json_data.get('result', [])
            
            # Filter by underlying
            filtered = [p for p in products if underlying in p.get('symbol', '')]
            
            logger.info(f"📦 Found {len(filtered)} option products for {underlying}")
            return filtered
            
        except Exception as e:
            logger.error(f"❌ Error getting products: {e}")
            return []
    
    def get_nearest_active_expiry(self, underlying='BTC'):
        """Get nearest active expiry from products"""
        try:
            products = self.get_all_products(underlying)
            
            if not products:
                return None
            
            # Extract unique expiries
            expiries = set()
            now = datetime.now()
            
            for product in products:
                settlement_time = product.get('settlement_time')
                if settlement_time:
                    try:
                        # Format: "2025-11-17T12:00:00Z"
                        expiry_dt = datetime.strptime(settlement_time, '%Y-%m-%dT%H:%M:%SZ')
                        # Set to 5:30 PM IST cutoff
                        expiry_dt = expiry_dt.replace(hour=17, minute=30)
                        
                        if expiry_dt > now:
                            expiry_date = expiry_dt.strftime('%d-%m-%Y')
                            expiries.add(expiry_date)
                    except:
                        continue
            
            if not expiries:
                return None
            
            # Get nearest
            sorted_expiries = sorted(list(expiries), key=lambda x: datetime.strptime(x, '%d-%m-%Y'))
            nearest = sorted_expiries[0] if sorted_expiries else None
            
            logger.info(f"✅ Nearest expiry for {underlying}: {nearest}")
            return nearest
            
        except Exception as e:
            logger.error(f"❌ Error finding expiry: {e}")
            return None
    
    def get_complete_option_chain(self, underlying='BTC'):
        """Get complete option chain using products + tickers API"""
        try:
            # Get spot price
            perp_symbol = f"{underlying}USD"
            ticker = self.get_ticker(perp_symbol)
            
            if not ticker:
                return None
            
            spot_price = float(ticker.get('mark_price', 0) or ticker.get('close', 0))
            
            if spot_price == 0:
                return None
            
            # Get nearest expiry
            expiry_date = self.get_nearest_active_expiry(underlying)
            
            if not expiry_date:
                logger.error(f"❌ No active expiry for {underlying}")
                return None
            
            # Get all option products for this expiry
            all_products = self.get_all_products(underlying)
            
            # Filter by expiry
            expiry_dt = datetime.strptime(expiry_date, '%d-%m-%Y')
            
            option_symbols = []
            for product in all_products:
                settlement_time = product.get('settlement_time')
                if settlement_time:
                    try:
                        prod_expiry = datetime.strptime(settlement_time, '%Y-%m-%dT%H:%M:%SZ')
                        prod_expiry_str = prod_expiry.strftime('%d-%m-%Y')
                        
                        if prod_expiry_str == expiry_date:
                            option_symbols.append(product.get('symbol'))
                    except:
                        continue
            
            logger.info(f"📊 Found {len(option_symbols)} options for {expiry_date}")
            
            # Calculate ATM
            atm_strike = self.round_to_strike(spot_price, underlying)
            strike_range = self.get_strike_range(atm_strike, underlying, count=8)
            
            # Fetch ticker data for each option
            calls = {}
            puts = {}
            
            for symbol in option_symbols:
                try:
                    opt_ticker = self.get_ticker(symbol)
                    
                    if not opt_ticker:
                        continue
                    
                    strike_price = opt_ticker.get('strike_price')
                    if not strike_price:
                        continue
                    
                    strike = float(strike_price)
                    
                    if strike not in strike_range:
                        continue
                    
                    is_call = 'C-' in symbol
                    
                    quotes = opt_ticker.get('quotes', {})
                    
                    option_data = {
                        'symbol': symbol,
                        'strike': strike,
                        'mark_price': self.safe_float(opt_ticker.get('mark_price')),
                        'bid': self.safe_float(quotes.get('best_bid')),
                        'ask': self.safe_float(quotes.get('best_ask')),
                        'oi': self.safe_float(opt_ticker.get('oi')),
                        'volume': self.safe_float(opt_ticker.get('volume')),
                    }
                    
                    if is_call:
                        calls[strike] = option_data
                    else:
                        puts[strike] = option_data
                    
                except Exception as e:
                    continue
            
            logger.info(f"✅ Calls: {len(calls)}, Puts: {len(puts)}")
            
            return {
                'underlying': underlying,
                'spot_price': spot_price,
                'atm_strike': atm_strike,
                'expiry_date': expiry_date,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'calls': calls,
                'puts': puts
            }
            
        except Exception as e:
            logger.error(f"❌ Option chain error: {e}")
            return None
    
    def safe_float(self, value, default=0):
        try:
            if value is None:
                return default
            return float(value)
        except:
            return default
    
    def round_to_strike(self, price, underlying):
        if underlying == 'BTC':
            return round(price / 1000) * 1000
        else:
            return round(price / 100) * 100
    
    def get_strike_range(self, atm_strike, underlying, count=8):
        if underlying == 'BTC':
            step = 1000
        else:
            step = 100
        
        strikes = []
        for i in range(-count, count + 1):
            strikes.append(atm_strike + (i * step))
        return strikes

# ==================== TELEGRAM FORMATTER ====================
class TelegramFormatter:
    @staticmethod
    def format_large_number(num):
        if num >= 1000:
            return f"{num/1000:.1f}K"
        return f"{num:.0f}"
    
    def format_option_chain(self, chain_data):
        if not chain_data:
            return "❌ No data available"
        
        underlying = chain_data['underlying']
        spot = chain_data['spot_price']
        atm = chain_data['atm_strike']
        expiry = chain_data['expiry_date']
        timestamp = chain_data['timestamp']
        calls = chain_data['calls']
        puts = chain_data['puts']
        
        # Time to expiry
        try:
            expiry_dt = datetime.strptime(expiry, '%d-%m-%Y').replace(hour=17, minute=30)
            delta = expiry_dt - datetime.now()
            
            if delta.total_seconds() < 0:
                tte = "EXPIRED"
            else:
                days = delta.days
                hours = delta.seconds // 3600
                minutes = (delta.seconds % 3600) // 60
                if days > 0:
                    tte = f"{days}d {hours}h"
                else:
                    tte = f"{hours}h {minutes}m"
        except:
            tte = "N/A"
        
        # Header
        message = f"""
🔔 <b>{underlying} OPTION CHAIN</b>

💰 Spot: <b>${spot:,.2f}</b>
🎯 ATM: <b>${atm:,.0f}</b>
📅 Expiry: <b>{expiry}</b>
⏰ TTE: <b>{tte}</b>
🕐 {timestamp}

━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Get all strikes
        all_strikes = sorted(set(list(calls.keys()) + list(puts.keys())))
        
        if not all_strikes:
            message += "\n⚠️ No options data\n"
            return message
        
        # Table
        message += "\n<b>   CALLS         |    PUTS</b>\n"
        message += "<code>Bid Ask  OI  | Bid Ask  OI</code>\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for strike in all_strikes:
            call = calls.get(strike, {})
            put = puts.get(strike, {})
            
            # Strike display
            if underlying == 'BTC':
                strike_str = f"${strike/1000:.0f}K"
            else:
                strike_str = f"${strike:,.0f}"
            
            # ATM marker
            marker = "🎯" if strike == atm else "  "
            
            # Call data
            if call and (call.get('bid') > 0 or call.get('ask') > 0):
                c_bid = f"{call.get('bid', 0):.0f}" if call.get('bid', 0) > 0 else "-"
                c_ask = f"{call.get('ask', 0):.0f}" if call.get('ask', 0) > 0 else "-"
                c_oi = self.format_large_number(call.get('oi', 0))
                call_line = f"{c_bid:>3} {c_ask:>3} {c_oi:>4}"
            else:
                call_line = " -   -    -"
            
            # Put data  
            if put and (put.get('bid') > 0 or put.get('ask') > 0):
                p_bid = f"{put.get('bid', 0):.0f}" if put.get('bid', 0) > 0 else "-"
                p_ask = f"{put.get('ask', 0):.0f}" if put.get('ask', 0) > 0 else "-"
                p_oi = self.format_large_number(put.get('oi', 0))
                put_line = f"{p_bid:>3} {p_ask:>3} {p_oi:>4}"
            else:
                put_line = " -   -    -"
            
            message += f"<code>{call_line} | {put_line}</code> {marker}{strike_str}\n"
        
        message += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # Summary
        total_call_oi = sum(c.get('oi', 0) for c in calls.values())
        total_put_oi = sum(p.get('oi', 0) for p in puts.values())
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        message += f"\n📊 <b>Summary</b>\n"
        message += f"Call OI: {self.format_large_number(total_call_oi)}\n"
        message += f"Put OI: {self.format_large_number(total_put_oi)}\n"
        message += f"PCR: <b>{pcr:.2f}</b>\n"
        
        message += "\n⚡ <i>Delta Exchange India</i>"
        
        return message

# ==================== MAIN BOT ====================
class DeltaOptionBot:
    def __init__(self):
        self.client = DeltaExchangeClient()
        self.telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.formatter = TelegramFormatter()
    
    async def send_telegram_message(self, message):
        try:
            await self.telegram_bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=message,
                parse_mode='HTML'
            )
            logger.info("✅ Sent")
        except Exception as e:
            logger.error(f"❌ Telegram: {e}")
    
    async def fetch_and_send(self, underlying):
        try:
            logger.info(f"🔍 Fetching {underlying}...")
            
            chain_data = self.client.get_complete_option_chain(underlying)
            
            if chain_data:
                message = self.formatter.format_option_chain(chain_data)
                await self.send_telegram_message(message)
            else:
                logger.warning(f"⚠️ No data for {underlying}")
                
        except Exception as e:
            logger.error(f"❌ Error: {e}")
    
    async def run(self):
        logger.info("="*50)
        logger.info("🚀 DELTA OPTION BOT - COMPLETE DATA")
        logger.info("="*50)
        
        startup = """
🚀 <b>DELTA OPTION BOT v2</b>

📊 BTC & ETH Option Chains
⏱️ Every 1 minute
📡 Auto nearest expiry
✅ Complete data fetch

━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ Status: 🟢 <b>ACTIVE</b>
"""
        await self.send_telegram_message(startup)
        
        while True:
            try:
                await self.fetch_and_send('BTC')
                await asyncio.sleep(5)
                
                await self.fetch_and_send('ETH')
                
                logger.info("⏳ Waiting 1 min...\n")
                await asyncio.sleep(55)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopped")
                break
            except Exception as e:
                logger.error(f"❌ Main: {e}")
                await asyncio.sleep(60)

if __name__ == "__main__":
    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        logger.error("❌ Missing env!")
        exit(1)
    
    try:
        bot = DeltaOptionBot()
        asyncio.run(bot.run())
    except Exception as e:
        logger.error(f"💥 Fatal: {e}")
