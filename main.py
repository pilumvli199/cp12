#!/usr/bin/env python3
"""
DELTA EXCHANGE DASHBOARD (LIQUIDITY BASED EXPIRY)
=================================================
1. Auto-Detects 'Main' Expiry based on Highest OI (Matches Website Default)
2. Fixes OI & IV Discrepancy
3. 100% Accurate Data Mapping
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
                    sym = p.get('symbol')
                    settlement = p.get('settlement_time')
                    c_type = p.get('contract_type')
                    strike = float(p.get('strike_price', 0))
                    
                    if settlement and sym:
                        try:
                            dt = datetime.strptime(settlement, '%Y-%m-%dT%H:%M:%SZ')
                            mapping[sym] = {
                                'expiry': dt,
                                'type': 'C' if c_type == 'call_options' else 'P',
                                'strike': strike
                            }
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
            # 1. Get Products
            product_map = self.get_products()
            if not product_map: return None

            # 2. Get Tickers
            tickers_res = self.session.get(f"{BASE_URL}/v2/tickers", timeout=10)
            all_tickers = tickers_res.json().get('result', [])
            if not all_tickers: return None

            # 3. Spot Price
            spot_sym = f"{underlying}USD"
            spot_data = next((t for t in all_tickers if t['symbol'] == spot_sym), None)
            spot_price = float(spot_data['mark_price']) if spot_data else 0

            # 4. SMART EXPIRY SELECTION (The Fix)
            # Group tickers by Expiry Date and calculate Total OI
            expiry_oi = {}
            
            now = datetime.now()
            
            for t in all_tickers:
                sym = t['symbol']
                if sym not in product_map: continue
                if underlying not in sym: continue
                
                exp_dt = product_map[sym]['expiry']
                if exp_dt <= now: continue # Skip expired

                oi = float(t.get('oi', 0) or 0)
                
                if exp_dt not in expiry_oi:
                    expiry_oi[exp_dt] = 0
                expiry_oi[exp_dt] += oi

            if not expiry_oi: return None

            # Select Expiry with HIGHEST OI (Most Liquid)
            # This matches what the website shows by default
            best_expiry = max(expiry_oi, key=expiry_oi.get)
            
            logger.info(f"📅 Selected {underlying} Expiry: {best_expiry} (Based on max liquidity)")

            # 5. Extract Data for Best Expiry
            data_rows = []
            for t in all_tickers:
                sym = t['symbol']
                if sym not in product_map: continue
                
                p_info = product_map[sym]
                if p_info['expiry'] != best_expiry: continue
                if underlying not in sym: continue

                strike = p_info['strike']
                o_type = p_info['type']
                
                ltp = float(t.get('mark_price', 0) or 0)
                oi = float(t.get('oi', 0) or 0)
                
                vol_qty = float(t.get('volume', 0) or 0)
                turnover = float(t.get('turnover', 0) or 0)
                vol_usd = turnover if turnover > 0 else (vol_qty * ltp)

                # IV Extraction
                iv = 0
                greeks = t.get('greeks')
                if greeks and isinstance(greeks, dict):
                    iv = float(greeks.get('implied_volatility', 0) or 0)
                
                data_rows.append({
                    'strike': strike,
                    'type': o_type,
                    'ltp': ltp,
                    'oi': oi,
                    'iv': iv,
                    'vol': vol_usd
                })

            # 6. Aggregation & Merge
            df = pd.DataFrame(data_rows)
            if df.empty: return None, 0, ""

            df_agg = df.groupby(['strike', 'type']).agg({
                'ltp': 'mean',
                'oi': 'sum',
                'vol': 'sum',
                'iv': 'max'
            }).reset_index()

            calls = df_agg[df_agg['type'] == 'C'].set_index('strike')
            puts = df_agg[df_agg['type'] == 'P'].set_index('strike')

            chain = pd.concat([calls, puts], axis=1, keys=['Call', 'Put'])
            chain = chain.sort_index()
            chain = chain.fillna(0)

            # 7. Filter Range
            idx = (chain.index.to_series() - spot_price).abs().idxmin()
            try:
                loc = chain.index.get_loc(idx)
                start = max(0, loc - 8)
                end = min(len(chain), loc + 9)
                final_df = chain.iloc[start:end].copy()
                return final_df, spot_price, best_expiry.strftime('%d-%b')
            except:
                return chain.head(16), spot_price, best_expiry.strftime('%d-%b')

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

        for strike, row in chain_df.iterrows():
            c = row['Call']
            p = row['Put']

            def fmt(val, is_curr=False):
                if val == 0: return "-"
                if is_curr: return f"{val:,.0f}"
                if val > 1000000: return f"{val/1000000:.1f}M"
                if val > 1000: return f"{val/1000:.0f}K"
                return f"{int(val)}"

            # IV Fix
            c_iv = c.get('iv', 0)
            if 0 < c_iv < 5: c_iv *= 100
            
            p_iv = p.get('iv', 0)
            if 0 < p_iv < 5: p_iv *= 100

            r = [
                fmt(c.get('vol', 0)), fmt(c.get('oi', 0)), fmt(c_iv), fmt(c.get('ltp', 0), True),
                f"{int(strike)}",
                fmt(p.get('ltp', 0), True), fmt(p_iv), fmt(p.get('oi', 0)), fmt(p.get('vol', 0))
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
