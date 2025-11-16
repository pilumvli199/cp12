#!/usr/bin/env python3
"""
DELTA EXCHANGE OPTION CHAIN BOT
================================
Professional Option Chain Data with Bid/Ask/Mark/OI/Volume
Telegram Alert System - Every 1 Minute
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
    
    def get_option_chain_detailed(self, underlying='BTC', expiry_date=None):
        """Get detailed option chain with all data"""
        try:
            # Get perpetual futures price
            perp_symbol = f"{underlying}USD"
            ticker = self.get_ticker(perp_symbol)
            
            if not ticker:
                logger.warning(f"⚠️ No ticker for {perp_symbol}")
                return None
            
            spot_price = float(ticker.get('mark_price', 0) or ticker.get('close', 0))
            
            if spot_price == 0:
                logger.warning(f"⚠️ Price is 0 for {underlying}")
                return None
            
            # Auto-calculate next Friday if no expiry provided
            if not expiry_date:
                today = datetime.now()
                days_until_friday = (4 - today.weekday()) % 7
                if days_until_friday == 0:
                    days_until_friday = 7
                next_friday = today + timedelta(days=days_until_friday)
                expiry_date = next_friday.strftime('%d-%m-%Y')
            
            # Fetch option chain
            url = f"{BASE_URL}/v2/tickers"
            params = {
                'contract_types': 'call_options,put_options',
                'underlying_asset_symbols': underlying,
                'expiry_date': expiry_date
            }
            
            logger.info(f"📡 Fetching {underlying} options for {expiry_date}")
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            
            if response.status_code != 200:
                logger.error(f"❌ Options API failed: {response.status_code}")
                return None
            
            json_data = response.json()
            
            if not json_data.get('success'):
                logger.error(f"❌ Options API unsuccessful")
                return None
            
            options_data = json_data.get('result', [])
            
            if not options_data:
                logger.warning(f"⚠️ No options for {underlying} on {expiry_date}")
                return None
            
            # Calculate ATM
            atm_strike = self.round_to_strike(spot_price, underlying)
            
            # Get strike range (ATM ±10 strikes)
            strike_range = self.get_strike_range(atm_strike, underlying, count=10)
            
            # Parse options data
            calls = {}
            puts = {}
            
            for opt in options_data:
                try:
                    strike = opt.get('strike_price')
                    if not strike:
                        continue
                    
                    strike_float = float(strike)
                    if strike_float not in strike_range:
                        continue
                    
                    symbol = opt.get('symbol', '')
                    is_call = 'C-' in symbol
                    
                    quotes = opt.get('quotes', {})
                    
                    option_info = {
                        'symbol': symbol,
                        'strike': strike_float,
                        'mark_price': self.safe_float(opt.get('mark_price')),
                        'bid': self.safe_float(quotes.get('best_bid')),
                        'ask': self.safe_float(quotes.get('best_ask')),
                        'bid_qty': self.safe_float(quotes.get('bid_size')),
                        'ask_qty': self.safe_float(quotes.get('ask_size')),
                        'oi': self.safe_float(opt.get('oi')),
                        'volume': self.safe_float(opt.get('volume')),
                        'bid_iv': self.safe_float(quotes.get('bid_iv')),
                        'ask_iv': self.safe_float(quotes.get('ask_iv')),
                    }
                    
                    if is_call:
                        calls[strike_float] = option_info
                    else:
                        puts[strike_float] = option_info
                        
                except Exception as e:
                    continue
            
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
            logger.error(f"❌ Option chain error for {underlying}: {e}")
            return None
    
    def safe_float(self, value, default=0):
        """Safely convert to float"""
        try:
            if value is None:
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def round_to_strike(self, price, underlying):
        """Round price to nearest strike"""
        if underlying == 'BTC':
            return round(price / 1000) * 1000
        else:  # ETH
            return round(price / 100) * 100
    
    def get_strike_range(self, atm_strike, underlying, count=10):
        """Get ±count strikes around ATM"""
        if underlying == 'BTC':
            step = 1000
        else:  # ETH
            step = 100
        
        strikes = []
        for i in range(-count, count + 1):
            strikes.append(atm_strike + (i * step))
        return strikes

# ==================== TELEGRAM FORMATTER ====================
class TelegramFormatter:
    @staticmethod
    def format_large_number(num):
        """Format large numbers (K for thousands)"""
        if num >= 1000:
            return f"{num/1000:.1f}K"
        return f"{num:.0f}"
    
    def format_detailed_option_chain(self, chain_data):
        """Format detailed option chain like Delta Exchange UI"""
        if not chain_data:
            return "❌ No data available"
        
        underlying = chain_data['underlying']
        spot = chain_data['spot_price']
        atm = chain_data['atm_strike']
        expiry = chain_data['expiry_date']
        timestamp = chain_data['timestamp']
        calls = chain_data['calls']
        puts = chain_data['puts']
        
        # Calculate time to expiry
        try:
            expiry_dt = datetime.strptime(expiry, '%d-%m-%Y')
            now = datetime.now()
            delta = expiry_dt - now
            days = delta.days
            hours = delta.seconds // 3600
            tte = f"{days}d:{hours}h"
        except:
            tte = "N/A"
        
        # Header
        message = f"""
🔔 <b>{underlying} OPTION CHAIN</b>

💰 Spot: <b>${spot:,.2f}</b>
🎯 ATM: <b>${atm:,.0f}</b>
📅 Expiry: {expiry}
⏰ TTE: {tte}
🕐 {timestamp}

━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Get all strikes
        all_strikes = sorted(set(list(calls.keys()) + list(puts.keys())))
        
        if not all_strikes:
            message += "\n⚠️ No options data available\n"
            return message
        
        # Table format
        message += "\n<b>CALLS                    |   PUTS</b>\n"
        message += "<code>Strike  | Bid   Ask  OI     | Bid   Ask  OI</code>\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for strike in all_strikes:
            call = calls.get(strike, {})
            put = puts.get(strike, {})
            
            # ATM indicator
            if strike == atm:
                indicator = "🎯"
            else:
                indicator = "  "
            
            # Format strike
            strike_str = f"${strike:,.0f}"
            
            # Call side
            if call:
                call_bid = f"${call.get('bid', 0):.1f}"
                call_ask = f"${call.get('ask', 0):.1f}"
                call_oi = self.format_large_number(call.get('oi', 0))
                call_line = f"{call_bid:>6} {call_ask:>6} {call_oi:>5}"
            else:
                call_line = "   -      -     -  "
            
            # Put side
            if put:
                put_bid = f"${put.get('bid', 0):.1f}"
                put_ask = f"${put.get('ask', 0):.1f}"
                put_oi = self.format_large_number(put.get('oi', 0))
                put_line = f"{put_bid:>6} {put_ask:>6} {put_oi:>5}"
            else:
                put_line = "   -      -     -  "
            
            message += f"<code>{indicator}{strike_str:>8} | {call_line} | {put_line}</code>\n"
        
        message += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # Summary stats
        total_call_oi = sum(c.get('oi', 0) for c in calls.values())
        total_put_oi = sum(p.get('oi', 0) for p in puts.values())
        
        message += f"\n📊 <b>Summary</b>\n"
        message += f"Call OI: {self.format_large_number(total_call_oi)}\n"
        message += f"Put OI: {self.format_large_number(total_put_oi)}\n"
        message += f"PCR: {(total_put_oi/total_call_oi if total_call_oi > 0 else 0):.2f}\n"
        
        message += "\n⚡ <i>Delta Exchange India</i>"
        
        return message

# ==================== MAIN BOT ====================
class DeltaOptionBot:
    def __init__(self):
        self.client = DeltaExchangeClient()
        self.telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.formatter = TelegramFormatter()
    
    async def send_telegram_message(self, message):
        """Send message to Telegram"""
        try:
            max_length = 4096
            if len(message) > max_length:
                parts = [message[i:i+max_length] for i in range(0, len(message), max_length)]
                for part in parts:
                    await self.telegram_bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=part,
                        parse_mode='HTML'
                    )
                    await asyncio.sleep(1)
            else:
                await self.telegram_bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=message,
                    parse_mode='HTML'
                )
            logger.info("✅ Message sent")
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")
    
    async def fetch_and_send_data(self, underlying):
        """Fetch detailed option chain and send to Telegram"""
        try:
            logger.info(f"🔍 Fetching {underlying}...")
            
            chain_data = self.client.get_option_chain_detailed(underlying)
            
            if chain_data:
                message = self.formatter.format_detailed_option_chain(chain_data)
                await self.send_telegram_message(message)
            else:
                logger.warning(f"⚠️ No data for {underlying}")
                await self.send_telegram_message(f"⚠️ No option data for {underlying}")
                
        except Exception as e:
            logger.error(f"❌ Error for {underlying}: {e}")
    
    async def run(self):
        """Main loop"""
        logger.info("="*50)
        logger.info("🚀 DELTA OPTION CHAIN BOT STARTED")
        logger.info("="*50)
        
        startup_msg = """
🚀 <b>DELTA OPTION CHAIN BOT</b>

📊 Assets: BTC & ETH
⏱️ Interval: 1 minute
📈 Data: Full Option Chain
   • Bid/Ask Prices
   • Open Interest
   • Strike-wise Analysis
   • PCR Ratio

━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ Status: 🟢 <b>ACTIVE</b>
"""
        await self.send_telegram_message(startup_msg)
        
        while True:
            try:
                # BTC Option Chain
                await self.fetch_and_send_data('BTC')
                await asyncio.sleep(5)
                
                # ETH Option Chain
                await self.fetch_and_send_data('ETH')
                
                # Wait 1 minute
                logger.info("⏳ Waiting 1 minute...\n")
                await asyncio.sleep(55)
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped")
                break
            except Exception as e:
                logger.error(f"❌ Main error: {e}")
                await asyncio.sleep(60)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    if not all([TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        logger.error("❌ Missing env vars!")
        exit(1)
    
    try:
        bot = DeltaOptionBot()
        asyncio.run(bot.run())
    except Exception as e:
        logger.error(f"💥 Fatal: {e}")
