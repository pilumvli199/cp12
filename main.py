#!/usr/bin/env python3
"""
DELTA EXCHANGE PRO DASHBOARD BOT (IMAGE GEN)
============================================
1. Generates High-Quality Image (Chart + Table)
2. Last 200 Candles (MPLFinance)
3. Accurate Option Chain Table (Pandas)
"""

import os
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import asyncio
from io import BytesIO
from datetime import datetime, timedelta
from telegram import Bot
import logging

# ==================== CONFIGURATION ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
BASE_URL = "https://api.india.delta.exchange"

# Configure Plot Style
plt.style.use('dark_background')

class DeltaDashboard:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    # ---------------- DATA FETCHING ----------------
    def get_products(self):
        """Fetch expiry mappings"""
        try:
            res = self.session.get(f"{BASE_URL}/v2/products", params={'contract_types': 'call_options,put_options', 'states': 'live'})
            if res.status_code == 200:
                data = res.json().get('result', [])
                mapping = {}
                for p in data:
                    if p.get('settlement_time'):
                        try:
                            dt = datetime.strptime(p['settlement_time'], '%Y-%m-%dT%H:%M:%SZ')
                            mapping[p['symbol']] = dt
                        except: pass
                return mapping
            return {}
        except: return {}

    def get_candles(self, symbol='BTCUSD', resolution='15m', count=200):
        """Fetch last 200 candles"""
        try:
            end_time = int(time.time())
            # Approx time for 200 candles (15m * 200 = 3000m = 50 hours)
            start_time = end_time - (200 * 15 * 60) 
            
            url = f"{BASE_URL}/v2/history/candles"
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'start': start_time,
                'end': end_time
            }
            res = self.session.get(url, params=params)
            if res.status_code == 200:
                candles = res.json().get('result', [])
                if not candles: return None
                
                # Create DataFrame
                df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df = df.sort_index()
                return df.tail(count)
            return None
        except Exception as e:
            logger.error(f"Candle Error: {e}")
            return None

    def get_chain_data(self, underlying='BTC'):
        """Fetch and organize option chain data"""
        try:
            # 1. Fetch Tickers & Products
            tickers_res = self.session.get(f"{BASE_URL}/v2/tickers")
            all_tickers = tickers_res.json().get('result', [])
            product_map = self.get_products()
            
            if not all_tickers or not product_map: return None

            # 2. Spot Price
            spot_sym = f"{underlying}USD"
            spot_data = next((t for t in all_tickers if t['symbol'] == spot_sym), None)
            spot_price = float(spot_data['mark_price']) if spot_data else 0

            # 3. Find Nearest Expiry
            valid_dates = []
            now = datetime.now()
            for sym, dt in product_map.items():
                if underlying in sym and dt > now:
                    valid_dates.append(dt)
            
            if not valid_dates: return None
            nearest_exp = sorted(list(set(valid_dates)))[0]

            # 4. Build DataFrame Rows
            rows = []
            for t in all_tickers:
                sym = t['symbol']
                if sym not in product_map: continue
                if product_map[sym] != nearest_exp: continue
                if underlying not in sym: continue

                strike = float(t.get('strike_price', 0))
                if strike == 0: continue

                # Calculate USD Volume
                vol_qty = float(t.get('volume', 0) or 0)
                mark_price = float(t.get('mark_price', 0) or 0)
                vol_usd = float(t.get('turnover', 0) or (vol_qty * mark_price))

                iv = float(t.get('greeks', {}).get('implied_volatility', 0) or 0)
                if 0 < iv < 5: iv *= 100 # Fix decimals

                row = {
                    'strike': strike,
                    'type': 'C' if 'C' in sym else 'P',
                    'ltp': mark_price,
                    'oi': float(t.get('oi', 0) or 0),
                    'iv': iv,
                    'vol': vol_usd
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            if df.empty: return None

            # 5. Pivot and Merge Calls/Puts
            calls = df[df['type'] == 'C'].set_index('strike')
            puts = df[df['type'] == 'P'].set_index('strike')

            # Combine
            chain = pd.concat([calls, puts], axis=1, keys=['Call', 'Put'])
            chain = chain.sort_index()

            # 6. Filter Range (ATM +/- 8)
            atm_idx = (chain.index.to_series() - spot_price).abs().idxmin()
            loc = chain.index.get_loc(atm_idx)
            start = max(0, loc - 8)
            end = min(len(chain), loc + 9)
            
            final_df = chain.iloc[start:end].copy()
            return final_df, spot_price, nearest_exp.strftime('%d-%b')

        except Exception as e:
            logger.error(f"Chain Error: {e}")
            return None, 0, ""

    # ---------------- IMAGE GENERATION ----------------
    def generate_dashboard(self, underlying='BTC'):
        """Create Image with Chart and Table"""
        logger.info(f"🎨 Generating Dashboard for {underlying}...")
        
        # Get Data
        candles = self.get_candles(f"{underlying}USD")
        chain_df, spot, exp = self.get_chain_data(underlying)
        
        if candles is None or chain_df is None: return None

        # Setup Figure
        fig = plt.figure(figsize=(12, 14)) # Height increased for table
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2]) # Chart top, Table bottom

        # --- PLOT 1: CANDLESTICK CHART ---
        ax1 = fig.add_subplot(gs[0])
        mc = mpf.make_marketcolors(up='#00ff00', down='#ff0000', edge='inherit', wick='inherit', volume='in')
        s = mpf.make_mpf_style(marketcolors=mc, style='nightclouds')
        
        mpf.plot(candles, type='candle', ax=ax1, style=s, volume=False, 
                 axtitle=f"{underlying}USD - 15m Chart | Spot: ${spot:,.0f}")

        # --- PLOT 2: OPTION CHAIN TABLE ---
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        ax2.set_title(f"OPTION CHAIN (Exp: {exp})", color='yellow', fontsize=14, pad=10)

        # Prepare Table Data
        table_data = []
        # Columns: C-Vol, C-OI, C-IV, C-LTP | STRIKE | P-LTP, P-IV, P-OI, P-Vol
        col_labels = ['Vol($)', 'OI', 'IV%', 'LTP', 'STRIKE', 'LTP', 'IV%', 'OI', 'Vol($)']

        for strike, row in chain_df.iterrows():
            # Format Numbers
            def fmt(val, is_curr=False):
                if pd.isna(val): return "-"
                if is_curr: return f"{val:,.0f}"
                if val > 1000000: return f"{val/1000000:.1f}M"
                if val > 1000: return f"{val/1000:.0f}K"
                return f"{int(val)}"

            c_row = row['Call']
            p_row = row['Put']

            r_data = [
                fmt(c_row.get('vol', 0)), fmt(c_row.get('oi', 0)), fmt(c_row.get('iv', 0)), fmt(c_row.get('ltp', 0), True),
                f"{int(strike)}",
                fmt(p_row.get('ltp', 0), True), fmt(p_row.get('iv', 0)), fmt(p_row.get('oi', 0)), fmt(p_row.get('vol', 0))
            ]
            table_data.append(r_data)

        # Draw Table
        table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        
        # Styling Table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8) # Make rows taller

        # Coloring
        for (row, col), cell in table.get_celld().items():
            if row == 0: # Header
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#333333')
            else:
                cell.set_edgecolor('#444444')
                cell.set_facecolor('black')
                cell.set_text_props(color='white')
                
                # Highlight ATM Strike Row (Middle row roughly)
                # A simpler way: Check strike text against spot
                try:
                    strike_val = float(table_data[row-1][4])
                    if abs(strike_val - spot) < (spot * 0.005): # Within 0.5%
                        cell.set_facecolor('#2A2A4A') # Dark Blue highlight
                        cell.set_text_props(color='#00FFFF')
                except: pass

                # Color Calls (Left side) Greenish text
                if col < 4: cell.set_text_props(color='#90EE90')
                # Color Puts (Right side) Reddish text
                elif col > 4: cell.set_text_props(color='#FFB6C1')
                # Strike (Center) Yellow
                elif col == 4: cell.set_text_props(weight='bold', color='yellow')

        # Save to buffer
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

# ==================== TELEGRAM BOT ====================
async def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ Token missing!")
        return

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dashboard = DeltaDashboard()
    
    logger.info("🚀 Image Bot Started...")

    while True:
        try:
            # BTC
            img_btc = dashboard.generate_dashboard('BTC')
            if img_btc:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img_btc, caption="🔥 #BTC Analysis")
                logger.info("✅ BTC Sent")

            await asyncio.sleep(10)

            # ETH
            img_eth = dashboard.generate_dashboard('ETH')
            if img_eth:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img_eth, caption="🔥 #ETH Analysis")
                logger.info("✅ ETH Sent")

            logger.info("💤 Waiting 2 mins...")
            await asyncio.sleep(120) # Image generation is heavy, wait 2 mins

        except Exception as e:
            logger.error(f"Loop Error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
