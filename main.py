#!/usr/bin/env python3
"""
DELTA EXCHANGE DASHBOARD (ROOT CAUSE FIXED)
===========================================
1. Fixed 'BTC' string bug (Puts were identified as Calls)
2. Fixed IV fetching logic
3. Accurate Data Alignment
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
                
                cols = ['open', 'high', 'low', 'close', 'volume']
                df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
                return df.tail(count)
            return None
        except: return None

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

            # 4. Separate Calls and Puts Lists (FIXED LOGIC)
            calls_list = []
            puts_list = []

            for t in all_tickers:
                sym = t['symbol']
                if sym not in product_map: continue
                if product_map[sym] != nearest_exp: continue
                if underlying not in sym: continue

                strike = float(t.get('strike_price', 0))
                if strike == 0: continue

                # Extract Data
                vol_qty = float(t.get('volume', 0) or 0)
                mark_price = float(t.get('mark_price', 0) or 0)
                vol_usd = float(t.get('turnover', 0) or (vol_qty * mark_price))
                
                # IV Fetching
                greeks = t.get('greeks')
                iv = 0
                if greeks and isinstance(greeks, dict):
                    iv = float(greeks.get('implied_volatility', 0) or 0)

                data = {
                    'strike': strike,
                    'ltp': mark_price,
                    'oi': float(t.get('oi', 0) or 0),
                    'iv': iv,
                    'vol': vol_usd
                }

                # --- THE FIX: Use contract_type instead of string matching ---
                c_type = t.get('contract_type', '').lower()
                
                if c_type == 'call_options':
                    calls_list.append(data)
                elif c_type == 'put_options':
                    puts_list.append(data)
                else:
                    # Fallback if contract_type is missing
                    if '_C' in sym or sym.endswith('C'):
                        calls_list.append(data)
                    elif '_P' in sym or sym.endswith('P'):
                        puts_list.append(data)

            # 5. Create DataFrames & Merge
            df_c = pd.DataFrame(calls_list)
            df_p = pd.DataFrame(puts_list)

            if df_c.empty and df_p.empty: return None, 0, ""

            # Remove duplicates (keep highest OI)
            if not df_c.empty:
                df_c = df_c.sort_values('oi', ascending=False).drop_duplicates('strike')
            if not df_p.empty:
                df_p = df_p.sort_values('oi', ascending=False).drop_duplicates('strike')

            # MERGE (Outer Join to keep both sides even if one is missing)
            if not df_c.empty and not df_p.empty:
                chain = pd.merge(df_c, df_p, on='strike', how='outer', suffixes=('_c', '_p'))
            elif not df_c.empty:
                chain = df_c.rename(columns=lambda x: x + '_c' if x != 'strike' else x)
                for col in ['ltp_p', 'oi_p', 'iv_p', 'vol_p']: chain[col] = 0
            else:
                chain = df_p.rename(columns=lambda x: x + '_p' if x != 'strike' else x)
                for col in ['ltp_c', 'oi_c', 'iv_c', 'vol_c']: chain[col] = 0

            chain = chain.fillna(0)
            chain = chain.sort_values('strike')

            # 6. Filter Range (ATM +/- 8)
            idx = (chain['strike'] - spot_price).abs().idxmin()
            try:
                loc = chain.index.get_loc(idx)
                # Handle potential duplicate index issue
                if isinstance(loc, slice): loc = loc.start
                if hasattr(loc, '__iter__'): loc = loc[0]
                
                loc = int(loc)
                start = max(0, loc - 8)
                end = min(len(chain), loc + 9)
                
                final_df = chain.iloc[start:end].copy()
                return final_df, spot_price, nearest_exp.strftime('%d-%b')
            except:
                return chain.head(16), spot_price, nearest_exp.strftime('%d-%b')

        except Exception as e:
            logger.error(f"Chain Error: {e}")
            return None, 0, ""

    # ---------------- IMAGE GENERATION ----------------
    def generate_dashboard(self, underlying='BTC'):
        logger.info(f"🎨 Generating Dashboard for {underlying}...")
        
        candles = self.get_candles(f"{underlying}USD")
        chain_df, spot, exp = self.get_chain_data(underlying)
        
        if candles is None or chain_df is None: return None

        fig = plt.figure(figsize=(12, 16))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.4])

        # Chart
        ax1 = fig.add_subplot(gs[0])
        mpf.plot(candles, type='candle', ax=ax1, style='yahoo', volume=False,
                 axtitle=f"{underlying} - 15m | Spot: ${spot:,.0f}")

        # Table
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        ax2.set_title(f"OPTION CHAIN (Exp: {exp})", color='yellow', fontsize=16, pad=10)

        table_data = []
        col_labels = ['Vol($)', 'OI', 'IV%', 'LTP', 'STRIKE', 'LTP', 'IV%', 'OI', 'Vol($)']

        for _, row in chain_df.iterrows():
            def fmt(val, is_curr=False):
                if val == 0: return "-"
                if is_curr: return f"{val:,.0f}"
                if val > 1000000: return f"{val/1000000:.1f}M"
                if val > 1000: return f"{val/1000:.0f}K"
                return f"{int(val)}"

            # IV Handling
            iv_c = row['iv_c']
            if 0 < iv_c < 5: iv_c *= 100 # Convert 0.5 to 50%
            
            iv_p = row['iv_p']
            if 0 < iv_p < 5: iv_p *= 100

            r = [
                fmt(row['vol_c']), fmt(row['oi_c']), fmt(iv_c), fmt(row['ltp_c'], True),
                f"{int(row['strike'])}",
                fmt(row['ltp_p'], True), fmt(iv_p), fmt(row['oi_p']), fmt(row['vol_p'])
            ]
            table_data.append(r)

        table = ax2.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.0)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#333333')
            else:
                cell.set_edgecolor('#555555')
                cell.set_facecolor('black')
                cell.set_text_props(color='white')

                try:
                    stk = float(table_data[row-1][4])
                    if abs(stk - spot) < (spot * 0.005):
                        cell.set_facecolor('#2A2A4A')
                        cell.set_text_props(color='#00FFFF', weight='bold')
                except: pass

                if col < 4: cell.set_text_props(color='#00FF00')
                elif col > 4: cell.set_text_props(color='#FF5555')
                elif col == 4: cell.set_text_props(color='yellow', weight='bold')

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
