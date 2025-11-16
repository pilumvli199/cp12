#!/usr/bin/env python3
"""
DELTA EXCHANGE TEST BOT
========================
BTC/ETH LTP + Option Chain Data every 1 minute
Telegram Alert System
"""

import os
import time
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
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    def get_ticker(self, symbol):
        """Get ticker for a symbol with timeout"""
        try:
            url = f"{BASE_URL}/v2/tickers/{symbol}"
            response = requests.get(url, headers=self.headers, timeout=5)
            
            if response.status_code == 200:
                json_data = response.json()
                
                if not json_data.get('success'):
                    logger.error(f"❌ Ticker API unsuccessful for {symbol}")
                    return None
                
                data = json_data.get('result', {})
                if not data:
                    logger.error(f"❌ No data for {symbol}")
                    return None
                
                return {
                    'symbol': data.get('symbol', symbol),
                    'mark_price': float(data.get('mark_price', 0)),
                    'spot_price': float(data.get('spot_price', 0)),
                    'close': float(data.get('close', 0)),
                    'volume': float(data.get('volume', 0)),
                    'turnover_usd': float(data.get('turnover_usd', 0)),
                    'oi': data.get('oi', 0)
                }
            else:
                logger.error(f"❌ Ticker failed {symbol}: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"⏱️ Timeout for {symbol}")
            return None
        except Exception as e:
            logger.error(f"❌ Ticker error {symbol}: {e}")
            return None
    
    def get_option_chain_simple(self, underlying='BTC', expiry_date=None):
        """Get option chain using tickers API - faster method"""
        try:
            # Get perpetual futures price first
            perp_symbol = f"{underlying}USD"
            ticker = self.get_ticker(perp_symbol)
            
            if not ticker:
                logger.warning(f"⚠️ No ticker for {perp_symbol}")
                return None
            
            # Use mark_price or close price
            spot_price = ticker.get('mark_price') or ticker.get('close', 0)
            
            if spot_price == 0:
                logger.warning(f"⚠️ Price is 0 for {underlying}")
                return None
            
            # If no expiry provided, get tickers for all options
            if not expiry_date:
                # Get current or nearest weekly expiry (Friday)
                from datetime import datetime, timedelta
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
            
            logger.info(f"📡 Fetching options for {underlying} expiry {expiry_date}")
            
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
                logger.warning(f"⚠️ No options data for {underlying} on {expiry_date}")
                return None
            
            # Calculate ATM
            atm_strike = self.round_to_strike(spot_price, underlying)
            
            # Filter options near ATM (±5 strikes for telegram message limit)
            strike_range = self.get_strike_range(atm_strike, underlying, count=5)
            
            options = []
            for opt in options_data:
                strike = opt.get('strike_price')
                if not strike:
                    continue
                
                strike_float = float(strike)
                if strike_float not in strike_range:
                    continue
                
                option_data = {
                    'symbol': opt.get('symbol', ''),
                    'strike': strike_float,
                    'type': 'CE' if 'C-' in opt.get('symbol', '') else 'PE',
                    'mark_price': float(opt.get('mark_price', 0)),
                    'open_interest': float(opt.get('oi', 0)),
                    'volume': float(opt.get('volume', 0)),
                    'best_bid': float(opt.get('quotes', {}).get('best_bid', 0)),
                    'best_ask': float(opt.get('quotes', {}).get('best_ask', 0)),
                }
                options.append(option_data)
            
            return {
                'underlying': underlying,
                'spot_price': spot_price,
                'atm_strike': atm_strike,
                'expiry_date': expiry_date,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'options': sorted(options, key=lambda x: (x['strike'], x['type']))
            }
            
        except requests.exceptions.Timeout:
            logger.error(f"⏱️ Timeout fetching options for {underlying}")
            return None
        except Exception as e:
            logger.error(f"❌ Option chain error for {underlying}: {e}")
            return None
    
    def round_to_strike(self, price, underlying):
        """Round price to nearest strike"""
        if underlying == 'BTC':
            return round(price / 1000) * 1000
        else:  # ETH
            return round(price / 100) * 100
    
    def get_strike_range(self, atm_strike, underlying, count=5):
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
    def format_option_chain_message(chain_data):
        """Format option chain data for Telegram"""
        if not chain_data:
            return "❌ No data available"
        
        underlying = chain_data['underlying']
        spot = chain_data['spot_price']
        atm = chain_data['atm_strike']
        expiry = chain_data.get('expiry_date', 'N/A')
        timestamp = chain_data['timestamp']
        options = chain_data['options']
        
        # Header
        message = f"""
🔔 <b>{underlying} OPTION CHAIN</b>

📊 Spot: <b>${spot:,.2f}</b>
🎯 ATM: <b>${atm:,.0f}</b>
📅 Expiry: {expiry}
🕐 {timestamp}

━━━━━━━━━━━━━━━━━━━━
"""
        
        if not options:
            message += "\n⚠️ No options available\n"
            return message
        
        # Separate CE and PE
        ce_options = [opt for opt in options if opt['type'] == 'CE']
        pe_options = [opt for opt in options if opt['type'] == 'PE']
        
        # Get unique strikes
        strikes = sorted(list(set([opt['strike'] for opt in options])))
        
        # Calls
        message += "\n📈 <b>CALLS (CE)</b>\n━━━━━━━━━━━━━━━━━━━━\n"
        
        for strike in strikes:
            ce = next((opt for opt in ce_options if opt['strike'] == strike), None)
            if ce:
                mark = "🎯 " if strike == atm else ""
                message += f"\n{mark}<b>Strike ${ce['strike']:,.0f}</b>\n"
                message += f"Premium: ${ce['mark_price']:.2f} | "
                message += f"OI: {ce['open_interest']:,.0f}\n"
        
        # Puts
        message += "\n\n📉 <b>PUTS (PE)</b>\n━━━━━━━━━━━━━━━━━━━━\n"
        
        for strike in strikes:
            pe = next((opt for opt in pe_options if opt['strike'] == strike), None)
            if pe:
                mark = "🎯 " if strike == atm else ""
                message += f"\n{mark}<b>Strike ${pe['strike']:,.0f}</b>\n"
                message += f"Premium: ${pe['mark_price']:.2f} | "
                message += f"OI: {pe['open_interest']:,.0f}\n"
        
        message += "\n━━━━━━━━━━━━━━━━━━━━\n⚡ Delta Exchange India"
        
        return message

# ==================== MAIN BOT ====================
class DeltaTestBot:
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
        """Fetch option chain and send to Telegram"""
        try:
            logger.info(f"🔍 Fetching {underlying}...")
            
            chain_data = self.client.get_option_chain_simple(underlying)
            
            if chain_data:
                message = self.formatter.format_option_chain_message(chain_data)
                await self.send_telegram_message(message)
            else:
                logger.warning(f"⚠️ No data for {underlying}")
                await self.send_telegram_message(f"⚠️ No data available for {underlying}")
                
        except Exception as e:
            logger.error(f"❌ Error for {underlying}: {e}")
    
    async def run(self):
        """Main loop"""
        logger.info("="*50)
        logger.info("🚀 DELTA BOT STARTED")
        logger.info("="*50)
        
        startup_msg = """
🚀 <b>DELTA BOT STARTED</b>

📊 Monitoring: BTC & ETH
⏱️ Interval: 1 minute
📡 Data: Option Chain (ATM ±5)

━━━━━━━━━━━━━━━━━━━━
⚡ Status: 🟢 <b>ACTIVE</b>
"""
        await self.send_telegram_message(startup_msg)
        
        while True:
            try:
                await self.fetch_and_send_data('BTC')
                await asyncio.sleep(5)
                
                await self.fetch_and_send_data('ETH')
                
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
        bot = DeltaTestBot()
        asyncio.run(bot.run())
    except Exception as e:
        logger.error(f"💥 Fatal: {e}")
