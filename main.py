#!/usr/bin/env python3
"""
DELTA EXCHANGE FINAL BOT (IV FIX)
=================================
1. Fixed Missing IV issue (Decimal vs Percentage)
2. 31 Strikes (Deep Analysis)
3. Corrected Layout
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
        self.product_map = {}
        self.last_product_fetch = None

    def get_products(self):
        if self.product_map and self.last_product_fetch:
            if (datetime.now() - self.last_product_fetch).total_seconds() < 3600:
                return self.product_map

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
                self.last_product_fetch = datetime.now()
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

    def get_analysis_data(self, underlying='BTC'):
        try:
            product_map = self.get_products()
            all_tickers = self.get_tickers()
            if not product_map or not all_tickers: return None

            # 1. Spot Price
            perp_symbol = f"{underlying}USD"
            spot_ticker = next((t for t in all_tickers if t['symbol'] == perp_symbol), None)
            spot_price = float(spot_ticker.get('mark_price') or 0) if spot_ticker else 0

            # 2. Filter Expiry
            options_data = []
            now = datetime.now()
            for t in all_tickers:
                if t.get('symbol') in product_map:
                    expiry_dt = product_map[t['symbol']]
                    if expiry_dt > now:
                        t['expiry_dt'] = expiry_dt
                        options_data.append(t)

            if not options_data: return None
            
            unique_dates = sorted(list(set(o['expiry_dt'] for o in options_data)))
            if not unique_dates: return None
            nearest_exp_dt = unique_dates[0]
            nearest_exp_str = nearest_exp_dt.strftime('%d-%m-%Y')

            # 3. Process Data
            calls, puts = {}, {}
            total_call_oi, total_put_oi = 0, 0
            atm_delta = 0
            
            atm_strike = self.round_to_strike(spot_price, underlying)

            for opt in options_data:
                if opt['expiry_dt'] != nearest_exp_dt: continue
                
                try:
                    strike = float(opt.get('strike_price', 0))
                    if strike == 0: continue

                    # --- IV FIX ---
                    greeks = opt.get('greeks', {})
                    # Sometimes IV is missing, default to 0
                    iv = float(greeks.get('implied_volatility', 0) or 0)
                    
                    # If IV is like 0.45, make it 45. If it's 45, keep 45.
                    # Usually crypto IV is > 10. If less than 1, it's decimal.
                    if 0 < iv < 5: 
                        iv = iv * 100

                    delta = float(greeks.get('delta', 0) or 0)
                    oi = float(opt.get('oi', 0) or 0)
                    
                    # Vol Fix
                    usd_vol = float(opt.get('turnover', 0) or 0)
                    if usd_vol == 0:
                        usd_vol = float(opt.get('volume',0)) * float(opt.get('mark_price',0))

                    if 'C' in opt['symbol']: total_call_oi += oi
                    if 'P' in opt['symbol']: total_put_oi += oi

                    if strike == atm_strike and 'C' in opt['symbol']:
                        atm_delta = delta

                    data = {
                        'iv': iv,
                        'oi': oi,
                        'vol': usd_vol
                    }
                    
                    c_type = opt.get('contract_type', '').lower()
                    if 'call' in c_type or opt['symbol'].endswith('C'):
                        calls[strike] = data
                    elif 'put' in c_type or opt['symbol'].endswith('P'):
                        puts[strike] = data
                except:
                    continue

            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            strikes = self.get_strike_range(atm_strike, underlying, count=15)

            return {
                'u': underlying, 'spot': spot_price, 'atm': atm_strike, 'exp': nearest_exp_str,
                'pcr': pcr, 'atm_delta': atm_delta,
                'calls': {k: calls[k] for k in strikes if k in calls},
                'puts': {k: puts[k] for k in strikes if k in puts}
            }

        except Exception as e:
            logger.error(f"Logic Error: {e}")
            return None

    def round_to_strike(self, price, underlying):
        step = 1000 if underlying == 'BTC' else 100
        return round(price / step) * step

    def get_strike_range(self, atm, underlying, count=15):
        step = 1000 if underlying == 'BTC' else 100
        return [atm + (i * step) for i in range(-count, count + 1)]

# ==================== FORMATTER ====================
class TelegramFormatter:
    @staticmethod
    def compact_num(num):
        if num >= 1000000: return f"{num/1000000:.1f}m"
        if num >= 1000: return f"{num/1000:.0f}k"
        return f"{num:.0f}"

    def generate_message(self, data):
        if not data: return None
        
        u, spot, atm = data['u'], data['spot'], data['atm']
        pcr = data['pcr']
        
        trend = "Neutral ⚖️"
        if pcr > 1.0: trend = "Bullish 🟢"
        elif pcr < 0.65: trend = "Bearish 🔴"

        msg = f"📊 <b>{u} DEEP CHAIN (31 Strikes)</b>\n"
        msg += f"🎯 Spot: <b>{spot:,.0f}</b> | Delta: <b>{data['atm_delta']:.2f}</b>\n"
        msg += f"⚖️ PCR: <b>{pcr:.2f}</b> ({trend})\n"
        msg += f"📅 Exp: <b>{data['exp']}</b>\n"
        msg += "─" * 32 + "\n"
        msg += "<b> CALLS (Vol/OI/IV) | PUTS (IV/OI/Vol)</b>\n"
        msg += "<code> Vol  OI IV | IV OI  Vol </code>\n"
        msg += "─" * 32 + "\n"

        strikes = sorted(set(list(data['calls'].keys()) + list(data['puts'].keys())))
        
        for k in strikes:
            c = data['calls'].get(k, {})
            p = data['puts'].get(k, {})
            
            cv = self.compact_num(c.get('vol', 0))
            co = self.compact_num(c.get('oi', 0))
            
            # Fix Display of IV: Show 0 if 0, don't show '-' unless None
            ci_val = c.get('iv', 0)
            pi_val = p.get('iv', 0)
            
            ci = f"{int(ci_val)}" if ci_val > 0 else "-"
            pi = f"{int(pi_val)}" if pi_val > 0 else "-"
            
            po = self.compact_num(p.get('oi', 0))
            pv = self.compact_num(p.get('vol', 0))

            marker = "🔹" if k == atm else " "
            st_lbl = f"{k/1000:.0f}k" if u == 'BTC' else f"{k:.0f}"

            # Adjust spacing slightly for better fit
            row = f"{cv:>4} {co:>3} {ci:>2}|{pi:<2} {po:<3} {pv:<4}"
            
            msg += f"<code>{row}</code>{marker}<b>{st_lbl}</b>\n"

        msg += "─" * 32 + "\n"
        msg += "<i>Vol in USD ($) | IV in %</i>"
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
    logger.info("🚀 Deep Analysis Bot Started...")
    
    while True:
        try:
            data = client.get_analysis_data('BTC')
            if data:
                await bot.send_message(TELEGRAM_CHAT_ID, fmt.generate_message(data), parse_mode='HTML')
                logger.info("✅ BTC Sent")
            
            await asyncio.sleep(5)

            data = client.get_analysis_data('ETH')
            if data:
                await bot.send_message(TELEGRAM_CHAT_ID, fmt.generate_message(data), parse_mode='HTML')
                logger.info("✅ ETH Sent")

        except Exception as e:
            logger.error(f"⚠️ Error: {e}")
        
        await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
