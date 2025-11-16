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
DELTA_API_KEY = os.getenv('DELTA_API_KEY')
DELTA_API_SECRET = os.getenv('DELTA_API_SECRET')
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Delta Exchange API Base URL
BASE_URL = "https://api.india.delta.exchange"

# ==================== DELTA API CLIENT ====================
class DeltaExchangeClient:
    def __init__(self, api_key, api_secret):
        self.api_key = api_key
        self.api_secret = api_secret
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    
    def get_products(self):
        """Get all available products"""
        try:
            url = f"{BASE_URL}/v2/products"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                return response.json().get('result', [])
            else:
                logger.error(f"❌ Products API failed: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"❌ Get products error: {e}")
            return []
    
    def get_ticker(self, symbol):
        """Get LTP for a symbol"""
        try:
            url = f"{BASE_URL}/v2/tickers/{symbol}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('result', {})
                return {
                    'symbol': data.get('symbol'),
                    'mark_price': float(data.get('mark_price', 0)),
                    'spot_price': float(data.get('spot_price', 0)),
                    'volume': float(data.get('volume', 0)),
                    'turnover_usd': float(data.get('turnover_usd', 0))
                }
            else:
                logger.error(f"❌ Ticker API failed: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"❌ Get ticker error: {e}")
            return None
    
    def get_option_chain(self, underlying='BTC'):
        """Get option chain for BTC/ETH"""
        try:
            # Get spot price first
            spot_symbol = f"{underlying}USDT"
            ticker = self.get_ticker(spot_symbol)
            
            if not ticker:
                return None
            
            spot_price = ticker['spot_price']
            atm_strike = self.round_to_strike(spot_price, underlying)
            
            # Get all products
            all_products = self.get_products()
            
            # Filter options near ATM (±10 strikes)
            options = []
            strike_range = self.get_strike_range(atm_strike, underlying)
            
            for product in all_products:
                if product.get('contract_type') != 'call_options' and product.get('contract_type') != 'put_options':
                    continue
                
                # Check if product is for our underlying
                if underlying not in product.get('symbol', ''):
                    continue
                
                strike = product.get('strike_price')
                if strike and float(strike) in strike_range:
                    # Get orderbook for this option
                    orderbook = self.get_orderbook(product['symbol'])
                    
                    option_data = {
                        'symbol': product['symbol'],
                        'strike': float(strike),
                        'type': 'CE' if product['contract_type'] == 'call_options' else 'PE',
                        'expiry': product.get('expiry_time', 'N/A'),
                        'mark_price': float(product.get('mark_price', 0)),
                        'open_interest': float(product.get('open_interest', 0)),
                        'volume': float(product.get('volume', 0)),
                        'best_bid': orderbook.get('best_bid', 0) if orderbook else 0,
                        'best_ask': orderbook.get('best_ask', 0) if orderbook else 0
                    }
                    options.append(option_data)
            
            return {
                'underlying': underlying,
                'spot_price': spot_price,
                'atm_strike': atm_strike,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'options': sorted(options, key=lambda x: (x['strike'], x['type']))
            }
            
        except Exception as e:
            logger.error(f"❌ Get option chain error: {e}")
            return None
    
    def get_orderbook(self, symbol):
        """Get orderbook for a symbol"""
        try:
            url = f"{BASE_URL}/v2/l2orderbook/{symbol}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json().get('result', {})
                buy = data.get('buy', [])
                sell = data.get('sell', [])
                
                return {
                    'best_bid': float(buy[0]['price']) if buy else 0,
                    'best_ask': float(sell[0]['price']) if sell else 0
                }
            return None
        except Exception as e:
            logger.error(f"❌ Orderbook error: {e}")
            return None
    
    def round_to_strike(self, price, underlying):
        """Round price to nearest strike"""
        if underlying == 'BTC':
            # BTC strikes in 1000s (e.g., 95000, 96000)
            return round(price / 1000) * 1000
        else:  # ETH
            # ETH strikes in 100s (e.g., 3400, 3500)
            return round(price / 100) * 100
    
    def get_strike_range(self, atm_strike, underlying):
        """Get ±10 strikes around ATM"""
        if underlying == 'BTC':
            step = 1000
        else:  # ETH
            step = 100
        
        strikes = []
        for i in range(-10, 11):
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
        timestamp = chain_data['timestamp']
        options = chain_data['options']
        
        # Header
        message = f"""
🔔 {underlying} OPTION CHAIN UPDATE

📊 Spot Price: ${spot:,.2f}
🎯 ATM Strike: ${atm:,.0f}
🕐 Time: {timestamp}

━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        # Separate CE and PE
        ce_options = [opt for opt in options if opt['type'] == 'CE']
        pe_options = [opt for opt in options if opt['type'] == 'PE']
        
        # Group by strike
        strikes = sorted(list(set([opt['strike'] for opt in options])))
        
        message += "\n📈 CALL OPTIONS (CE):\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        for strike in strikes:
            ce = next((opt for opt in ce_options if opt['strike'] == strike), None)
            
            if ce:
                mark = "🎯" if strike == atm else "  "
                message += f"\n{mark} Strike: ${ce['strike']:,.0f}\n"
                message += f"   Premium: ${ce['mark_price']:.2f}\n"
                message += f"   OI: {ce['open_interest']:,.0f}\n"
                message += f"   Vol: {ce['volume']:,.0f}\n"
                message += f"   Bid/Ask: ${ce['best_bid']:.2f}/${ce['best_ask']:.2f}\n"
        
        message += "\n\n📉 PUT OPTIONS (PE):\n"
        message += "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        for strike in strikes:
            pe = next((opt for opt in pe_options if opt['strike'] == strike), None)
            
            if pe:
                mark = "🎯" if strike == atm else "  "
                message += f"\n{mark} Strike: ${pe['strike']:,.0f}\n"
                message += f"   Premium: ${pe['mark_price']:.2f}\n"
                message += f"   OI: {pe['open_interest']:,.0f}\n"
                message += f"   Vol: {pe['volume']:,.0f}\n"
                message += f"   Bid/Ask: ${pe['best_bid']:.2f}/${pe['best_ask']:.2f}\n"
        
        message += "\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
        message += "⚡ Delta Exchange India\n"
        
        return message

# ==================== MAIN BOT ====================
class DeltaTestBot:
    def __init__(self):
        self.client = DeltaExchangeClient(DELTA_API_KEY, DELTA_API_SECRET)
        self.telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)
        self.formatter = TelegramFormatter()
    
    async def send_telegram_message(self, message):
        """Send message to Telegram"""
        try:
            # Split long messages
            max_length = 4096
            if len(message) > max_length:
                # Send in parts
                parts = [message[i:i+max_length] for i in range(0, len(message), max_length)]
                for part in parts:
                    await self.telegram_bot.send_message(
                        chat_id=TELEGRAM_CHAT_ID,
                        text=part,
                        parse_mode='HTML'
                    )
                    await asyncio.sleep(1)  # Avoid rate limit
            else:
                await self.telegram_bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=message,
                    parse_mode='HTML'
                )
            logger.info("✅ Message sent to Telegram")
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")
    
    async def fetch_and_send_data(self, underlying):
        """Fetch option chain and send to Telegram"""
        try:
            logger.info(f"🔍 Fetching {underlying} data...")
            
            # Get option chain
            chain_data = self.client.get_option_chain(underlying)
            
            if chain_data:
                # Format message
                message = self.formatter.format_option_chain_message(chain_data)
                
                # Send to Telegram
                await self.send_telegram_message(message)
            else:
                logger.warning(f"⚠️ No data for {underlying}")
                
        except Exception as e:
            logger.error(f"❌ Fetch error: {e}")
    
    async def run(self):
        """Main loop - fetch every 1 minute"""
        logger.info("="*50)
        logger.info("🚀 DELTA EXCHANGE TEST BOT STARTED")
        logger.info("="*50)
        
        # Send startup message
        startup_msg = """
🚀 DELTA TEST BOT STARTED

📊 Monitoring: BTC & ETH
⏱️ Interval: 1 minute
📡 Data: LTP + Option Chain (ATM ±10)

━━━━━━━━━━━━━━━━━━━━━━━━
⚡ Status: 🟢 ACTIVE
"""
        await self.send_telegram_message(startup_msg)
        
        while True:
            try:
                # Fetch BTC data
                await self.fetch_and_send_data('BTC')
                await asyncio.sleep(5)  # Wait 5 sec between BTC and ETH
                
                # Fetch ETH data
                await self.fetch_and_send_data('ETH')
                
                # Wait 1 minute before next cycle
                logger.info("⏳ Waiting 1 minute for next update...\n")
                await asyncio.sleep(55)  # 55 + 5 = 60 seconds total
                
            except KeyboardInterrupt:
                logger.info("🛑 Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Main loop error: {e}")
                await asyncio.sleep(60)

# ==================== ENTRY POINT ====================
if __name__ == "__main__":
    # Validate environment variables
    if not all([DELTA_API_KEY, DELTA_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
        logger.error("❌ Missing environment variables!")
        logger.error("Required: DELTA_API_KEY, DELTA_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID")
        exit(1)
    
    try:
        bot = DeltaTestBot()
        asyncio.run(bot.run())
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
