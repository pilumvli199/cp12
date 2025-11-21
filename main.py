#!/usr/bin/env python3
"""
DELTA EXCHANGE DASHBOARD (BUG FIXED)
====================================
1. Fixes 'Reindexing' error (Duplicate Strikes)
2. Fixes 'Style' error (MPLFinance)
3. Generates Image with Chart + Table
"""

import os
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import asyncio
from io import BytesIO
from datetime import datetime
from telegram import Bot
import logging

# ==================== CONFIGURATION ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
BASE_URL = "https://api.india.delta.exchange"

# Dark Background for Plots
plt.style.use('dark_background')

class DeltaDashboard:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})

    # ---------------- DATA FETCHING ----------------
    def get_products(self):
        try:
            res = self.session.get(f"{BASE_URL}/v2/products", params={'contract_types': 'call_options,put_options', 'states': 'live'}, timeout=10)
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
        try:
            end_time = int(time.time())
            start_time = end_time - (200 * 15 * 60)
            
            url = f"{BASE_URL}/v2/history/candles"
            params = {'symbol': symbol, 'resolution': resolution, 'start': start_time, 'end': end_time}
            res = self.session.get(url, params=params, timeout=10)
            
            if res.status_code == 200:
                candles = res.json().get('result', [])
                if not candles: return None
                
                df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df = df.sort_index()
                
                # Convert cols to float
                cols = ['open', 'high', 'low', 'close', 'volume']
                df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
                
                return df.tail(count)
            return None
        except Exception as e:
            logger.error(f"Candle Error: {e}")
            return None

    def get_chain_data(self, underlying='BTC'):
        try:
            # 1. Fetch Data
            tickers_res = self.session.get(f"{BASE_URL}/v2/tickers", timeout=10)
            all_tickers = tickers_res.json().get('result', [])
            product_map = self.get_products()
            
            if not all_tickers or not product_map: return None

            # 2. Spot Price
            spot_sym = f"{underlying}USD"
            spot_data = next((t for t in all_tickers if t['symbol'] == spot_sym), None)
            spot_price = float(spot_data['mark_price']) if spot_data else 0

            # 3. Nearest Expiry
            valid_dates = []
            now = datetime.now()
            for sym, dt in product_map.items():
                if underlying in sym and dt > now:
                    valid_dates.append(dt)
            
            if not valid_dates: return None
            nearest_exp = sorted(list(set(valid_dates)))[0]

            # 4. Build Rows
            rows = []
            for t in all_tickers:
                sym = t['symbol']
                if sym not in product_map: continue
                if product_map[sym] != nearest_exp: continue
                if underlying not in sym: continue

                strike = float(t.get('strike_price', 0))
                if strike == 0: continue

                # Calculate USD Volume (Turnover)
                vol_qty = float(t.get('volume', 0) or 0)
                mark_price = float(t.get('mark_price', 0) or 0)
                vol_usd = float(t.get('turnover', 0) or (vol_qty * mark_price))

                iv = float(t.get('greeks', {}).get('implied_volatility', 0) or 0)
                if 0 < iv < 5: iv *= 100

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

            # --- FIX 1: Remove Duplicates (The Reindexing Error Fix) ---
            # Keep the one with higher OI if duplicates exist
            df = df.sort_values(by='oi', ascending=False)
            df = df.drop_duplicates(subset=['strike', 'type'])
            df = df.sort_values(by='strike')

            # 5. Separate Calls & Puts
            calls = df[df['type'] == 'C'].set_index('strike')
            puts = df[df['type'] == 'P'].set_index('strike')

            # Combine
            chain = pd.concat([calls, puts], axis=1, keys=['Call', 'Put'])
            chain = chain.sort_index()

            # 6. Filter Range (ATM +/- 8)
            # Find closest strike to spot
            idx = (chain.index.to_series() - spot_price).abs().idxmin()
            try:
                loc = chain.index.get_loc(idx)
                start = max(0, loc - 8)
                end = min(len(chain), loc + 9)
                final_df = chain.iloc[start:end].copy()
            except:
                final_df = chain.iloc[:16].copy() # Fallback

            return final_df, spot_price, nearest_exp.strftime('%d-%b')

        except Exception as e:
            logger.error(f"Chain Error: {e}")
            return None, 0, ""

    # ---------------- IMAGE GENERATION ----------------
    def generate_dashboard(self, underlying='BTC'):
        logger.info(f"🎨 Generating Dashboard for {underlying}...")
        
        candles = self.get_candles(f"{underlying}USD")
        chain_df, spot, exp = self.get_chain_data(underlying)
        
        if candles is None or chain_df is None: 
            logger.warning(f"⚠️ Data missing for {underlying}")
            return None

        # Layout: Top for Chart, Bottom for Table
        fig = plt.figure(figsize=(12, 16))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.4])

        # --- PLOT 1: CHART ---
        ax1 = fig.add_subplot(gs[0])
        
        # --- FIX 2: Simple Style (Fixes 'Unrecognized kwarg' Error) ---
        # Using 'yahoo' style which is built-in and robust
        mpf.plot(candles, type='candle', ax=ax1, style='yahoo', volume=False,
                 axtitle=f"{underlying} - 15m | Spot: ${spot:,.0f}")

        # --- PLOT 2: TABLE ---
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        ax2.set_title(f"OPTION CHAIN (Exp: {exp})", color='yellow', fontsize=16, pad=10)

        table_data = []
        col_labels = ['Vol($)', 'OI', 'IV%', 'LTP', 'STRIKE', 'LTP', 'IV%', 'OI', 'Vol($)']

        for strike, row in chain_df.iterrows():
            def fmt(val, is_curr=False):
                if pd.isna(val): return "-"
                if is_curr: return f"{val:,.0f}"
                if val > 1000000: return f"{val/1000000:.1f}M"
                if val > 1000: return f"{val/1000:.0f}K"
                return f"{int(val)}"

            c = row['Call']
            p = row['Put']

            r = [
                fmt(c.get('vol', 0)), fmt(c.get('oi', 0)), fmt(c.get('iv', 0)), fmt(c.get('ltp', 0), True),
                f"{int(strike)}",
                fmt(p.get('ltp', 0), True), fmt(p.get('iv', 0)), fmt(p.get('oi', 0)), fmt(p.get('vol', 0))
            ]
            table_data.append(r)

        # Draw Table
        table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.0) # Taller rows

        # Styling
        for (row, col), cell in table.get_celld().items():
            if row == 0: # Header
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#333333')
            else:
                cell.set_edgecolor('#555555')
                cell.set_facecolor('black')
                cell.set_text_props(color='white')

                # Highlight ATM
                try:
                    stk = float(table_data[row-1][4])
                    if abs(stk - spot) < (spot * 0.005):
                        cell.set_facecolor('#2A2A4A')
                        cell.set_text_props(color='#00FFFF', weight='bold')
                except: pass

                # Green Calls / Red Puts
                if col < 4: cell.set_text_props(color='#00FF00') # Green
                elif col > 4: cell.set_text_props(color='#FF5555') # Red
                elif col == 4: cell.set_text_props(color='yellow', weight='bold') # Strike

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

# ==================== BOT RUNNER ====================
async def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ Token missing!")
        return

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dash = DeltaDashboard()
    logger.info("🚀 Dashboard Bot Started...")

    while True:
        try:
            # BTC
            img = dash.generate_dashboard('BTC')
            if img:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img, caption="📊 #BTC Option Chain")
                logger.info("✅ BTC Sent")
            
            await asyncio.sleep(10)

            # ETH
            img = dash.generate_dashboard('ETH')
            if img:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img, caption="📊 #ETH Option Chain")
                logger.info("✅ ETH Sent")

            logger.info("💤 Waiting 2 mins...")
            await asyncio.sleep(120)

        except Exception as e:
            logger.error(f"Main Loop Error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
