#!/usr/bin/env python3
"""
DELTA EXCHANGE DASHBOARD (FIXED)
================================
- Correct API endpoint for option chain with expiry_date filter
- Proper OI & IV data extraction
- Matches website data exactly
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

plt.style.use('dark_background')

class DeltaDashboard:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    # ---------------- HELPER: Get Available Expiries ----------------
    def get_available_expiries(self, underlying='BTC'):
        """Get all available expiry dates for an underlying"""
        try:
            url = f"{BASE_URL}/v2/products"
            params = {
                'contract_types': 'call_options,put_options',
                'states': 'live'
            }
            res = self.session.get(url, params=params, timeout=10)
            if res.status_code != 200:
                return []
            
            products = res.json().get('result', [])
            expiries = set()
            now = datetime.utcnow()
            
            for p in products:
                sym = p.get('symbol', '')
                if underlying not in sym:
                    continue
                
                settlement = p.get('settlement_time')
                if settlement:
                    try:
                        dt = datetime.strptime(settlement, '%Y-%m-%dT%H:%M:%SZ')
                        if dt > now:
                            expiries.add(dt)
                    except:
                        pass
            
            return sorted(list(expiries))
        except Exception as e:
            logger.error(f"Error getting expiries: {e}")
            return []

    # ---------------- DATA FETCHING ----------------
    def get_candles(self, symbol='BTCUSD', resolution='15m', count=200):
        try:
            end_time = int(time.time())
            start_time = end_time - (count * 15 * 60)
            
            url = f"{BASE_URL}/v2/history/candles"
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'start': start_time,
                'end': end_time
            }
            res = self.session.get(url, params=params, timeout=10)
            
            if res.status_code == 200:
                candles = res.json().get('result', [])
                if not candles:
                    return None
                
                df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df = df.sort_index()
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                return df.tail(count)
            return None
        except Exception as e:
            logger.error(f"Candles Error: {e}")
            return None

    def get_chain_data(self, underlying='BTC'):
        """
        Fetch option chain using the CORRECT API endpoint:
        /v2/tickers?contract_types=call_options,put_options&underlying_asset_symbols=BTC&expiry_date=DD-MM-YYYY
        """
        try:
            # 1. Get spot price first
            spot_url = f"{BASE_URL}/v2/tickers/{underlying}USD"
            spot_res = self.session.get(spot_url, timeout=10)
            spot_price = 0
            if spot_res.status_code == 200:
                spot_data = spot_res.json().get('result', {})
                spot_price = float(spot_data.get('mark_price', 0) or spot_data.get('close', 0) or 0)
            
            if spot_price == 0:
                logger.error("Could not get spot price")
                return None, 0, ""

            # 2. Get available expiries and select nearest
            expiries = self.get_available_expiries(underlying)
            if not expiries:
                logger.error("No expiries found")
                return None, 0, ""
            
            # Select nearest expiry (first one)
            target_expiry = expiries[0]
            expiry_str = target_expiry.strftime('%d-%m-%Y')  # Format: DD-MM-YYYY
            
            logger.info(f"📅 Using expiry: {expiry_str} for {underlying}")

            # 3. Fetch option chain with CORRECT API
            url = f"{BASE_URL}/v2/tickers"
            params = {
                'contract_types': 'call_options,put_options',
                'underlying_asset_symbols': underlying,
                'expiry_date': expiry_str
            }
            
            res = self.session.get(url, params=params, timeout=15)
            if res.status_code != 200:
                logger.error(f"API Error: {res.status_code} - {res.text}")
                return None, 0, ""
            
            tickers = res.json().get('result', [])
            if not tickers:
                logger.error("No tickers returned")
                return None, 0, ""
            
            logger.info(f"📊 Got {len(tickers)} option contracts")

            # 4. Parse the data
            data_rows = []
            for t in tickers:
                sym = t.get('symbol', '')
                contract_type = t.get('contract_type', '')
                
                # Determine Call or Put
                if contract_type == 'call_options' or sym.startswith('C-'):
                    o_type = 'C'
                elif contract_type == 'put_options' or sym.startswith('P-'):
                    o_type = 'P'
                else:
                    continue
                
                # Strike price - from ticker or parse from symbol
                strike = float(t.get('strike_price', 0) or 0)
                if strike == 0:
                    # Parse from symbol: C-BTC-85000-221125
                    parts = sym.split('-')
                    if len(parts) >= 3:
                        try:
                            strike = float(parts[2])
                        except:
                            continue
                
                if strike == 0:
                    continue
                
                # Mark Price (LTP)
                ltp = float(t.get('mark_price', 0) or 0)
                
                # Open Interest - use oi_value_usd for USD value, or oi for contracts
                oi = float(t.get('oi', 0) or 0)
                oi_value = float(t.get('oi_value_usd', 0) or t.get('oi_value', 0) or 0)
                
                # Volume
                vol = float(t.get('turnover_usd', 0) or t.get('turnover', 0) or 0)
                
                # IV - from mark_vol or greeks
                iv = 0
                mark_vol = t.get('mark_vol')
                if mark_vol:
                    iv = float(mark_vol)
                
                # Also check greeks
                greeks = t.get('greeks')
                if greeks and isinstance(greeks, dict):
                    greek_iv = greeks.get('implied_volatility')
                    if greek_iv:
                        iv = float(greek_iv)
                
                # Also check quotes for bid/ask IV
                quotes = t.get('quotes', {})
                if quotes:
                    ask_iv = float(quotes.get('ask_iv', 0) or 0)
                    bid_iv = float(quotes.get('bid_iv', 0) or 0)
                    if ask_iv > 0 and bid_iv > 0:
                        iv = (ask_iv + bid_iv) / 2
                    elif ask_iv > 0:
                        iv = ask_iv
                    elif bid_iv > 0:
                        iv = bid_iv
                
                # IV is returned as decimal (0.95 = 95%), convert to percentage
                if 0 < iv < 5:
                    iv = iv * 100
                
                data_rows.append({
                    'strike': strike,
                    'type': o_type,
                    'ltp': ltp,
                    'oi': oi,
                    'oi_value': oi_value,
                    'iv': iv,
                    'vol': vol
                })
            
            if not data_rows:
                logger.error("No valid data rows")
                return None, 0, ""
            
            # 5. Create DataFrame and pivot
            df = pd.DataFrame(data_rows)
            logger.info(f"📈 Parsed {len(df)} options")
            
            # Aggregate by strike and type
            df_agg = df.groupby(['strike', 'type']).agg({
                'ltp': 'first',
                'oi': 'sum',
                'oi_value': 'sum',
                'vol': 'sum',
                'iv': 'max'
            }).reset_index()
            
            # Pivot calls and puts
            calls = df_agg[df_agg['type'] == 'C'].set_index('strike')
            puts = df_agg[df_agg['type'] == 'P'].set_index('strike')
            
            # Rename columns
            calls = calls.rename(columns={
                'ltp': 'c_ltp', 'oi': 'c_oi', 'oi_value': 'c_oi_val',
                'vol': 'c_vol', 'iv': 'c_iv'
            })
            puts = puts.rename(columns={
                'ltp': 'p_ltp', 'oi': 'p_oi', 'oi_value': 'p_oi_val',
                'vol': 'p_vol', 'iv': 'p_iv'
            })
            
            # Merge
            chain = pd.concat([
                calls[['c_vol', 'c_oi', 'c_iv', 'c_ltp']],
                puts[['p_ltp', 'p_iv', 'p_oi', 'p_vol']]
            ], axis=1)
            
            chain = chain.sort_index()
            chain = chain.fillna(0)
            
            # 6. Filter to strikes around spot price
            all_strikes = chain.index.tolist()
            if not all_strikes:
                return chain, spot_price, target_expiry.strftime('%d-%b')
            
            # Find ATM strike
            atm_strike = min(all_strikes, key=lambda x: abs(x - spot_price))
            atm_idx = all_strikes.index(atm_strike)
            
            # Get 8 strikes above and below ATM
            start_idx = max(0, atm_idx - 8)
            end_idx = min(len(all_strikes), atm_idx + 9)
            
            filtered_strikes = all_strikes[start_idx:end_idx]
            chain = chain.loc[filtered_strikes]
            
            return chain, spot_price, target_expiry.strftime('%d-%b')

        except Exception as e:
            logger.error(f"Chain Error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0, ""

    # ---------------- IMAGE GENERATION ----------------
    def generate_dashboard(self, underlying='BTC'):
        logger.info(f"🎨 Generating Dashboard for {underlying}...")
        
        candles = self.get_candles(f"{underlying}USD")
        chain_result = self.get_chain_data(underlying)
        
        if chain_result is None:
            logger.error("Chain data is None")
            return None
            
        chain_df, spot, exp = chain_result
        
        if candles is None or chain_df is None or chain_df.empty:
            logger.error(f"Data missing - Candles: {candles is not None}, Chain: {chain_df is not None}")
            return None

        fig = plt.figure(figsize=(14, 18))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5])

        # Chart
        ax1 = fig.add_subplot(gs[0])
        mpf.plot(candles, type='candle', ax=ax1, style='yahoo', volume=False,
                 axtitle=f"{underlying} - 15m | Spot: ${spot:,.2f}")

        # Table
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        ax2.set_title(f"OPTION CHAIN (Exp: {exp})", color='yellow', fontsize=16, pad=10)

        table_data = []
        col_labels = ['Vol($)', 'OI', 'IV%', 'LTP', 'STRIKE', 'LTP', 'IV%', 'OI', 'Vol($)']

        def fmt(val, is_curr=False, is_iv=False):
            if val == 0 or pd.isna(val):
                return "-"
            if is_iv:
                return f"{val:.1f}"
            if is_curr:
                return f"{val:,.2f}" if val < 1000 else f"{val:,.0f}"
            if val >= 1000000:
                return f"{val/1000000:.2f}M"
            if val >= 1000:
                return f"{val/1000:.1f}K"
            return f"{val:.0f}"

        for strike in chain_df.index:
            row = chain_df.loc[strike]
            
            # Call data
            c_vol = row.get('c_vol', 0)
            c_oi = row.get('c_oi', 0)
            c_iv = row.get('c_iv', 0)
            c_ltp = row.get('c_ltp', 0)
            
            # Put data
            p_ltp = row.get('p_ltp', 0)
            p_iv = row.get('p_iv', 0)
            p_oi = row.get('p_oi', 0)
            p_vol = row.get('p_vol', 0)
            
            r = [
                fmt(c_vol),
                fmt(c_oi),
                fmt(c_iv, is_iv=True),
                fmt(c_ltp, is_curr=True),
                f"{int(strike):,}",
                fmt(p_ltp, is_curr=True),
                fmt(p_iv, is_iv=True),
                fmt(p_oi),
                fmt(p_vol)
            ]
            table_data.append(r)

        if not table_data:
            logger.error("No table data")
            return None

        table = ax2.table(cellText=table_data, colLabels=col_labels, 
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.0)

        # Style the table
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#333333')
            else:
                cell.set_edgecolor('#555555')
                cell.set_facecolor('black')
                cell.set_text_props(color='white')

                try:
                    stk = float(table_data[row-1][4].replace(',', ''))
                    if abs(stk - spot) < (spot * 0.01):
                        cell.set_facecolor('#2A2A4A')
                        cell.set_text_props(color='#00FFFF', weight='bold')
                except:
                    pass

                # Color coding: Calls (green), Puts (red), Strike (yellow)
                if col < 4:
                    cell.set_text_props(color='#00FF00')
                elif col > 4:
                    cell.set_text_props(color='#FF5555')
                elif col == 4:
                    cell.set_text_props(color='yellow', weight='bold')

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                   facecolor='black', edgecolor='none')
        buf.seek(0)
        plt.close(fig)
        return buf

# ==================== BOT RUNNER ====================
async def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN missing!")
        return

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dash = DeltaDashboard()
    logger.info("🚀 Dashboard Bot Started...")

    while True:
        try:
            # BTC Dashboard
            img = dash.generate_dashboard('BTC')
            if img:
                await bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID, 
                    photo=img, 
                    caption="📊 #BTC Option Chain\n💡 Data from Delta Exchange"
                )
                logger.info("✅ BTC Sent")
            else:
                logger.warning("⚠️ BTC dashboard generation failed")
            
            await asyncio.sleep(10)

            # ETH Dashboard
            img = dash.generate_dashboard('ETH')
            if img:
                await bot.send_photo(
                    chat_id=TELEGRAM_CHAT_ID, 
                    photo=img, 
                    caption="📊 #ETH Option Chain\n💡 Data from Delta Exchange"
                )
                logger.info("✅ ETH Sent")
            else:
                logger.warning("⚠️ ETH dashboard generation failed")

            logger.info("💤 Waiting 2 mins...")
            await asyncio.sleep(120)

        except Exception as e:
            logger.error(f"Main Loop Error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(60)

if __name__ == "__main__":
    # Test mode - run once without Telegram
    if os.getenv('TEST_MODE'):
        dash = DeltaDashboard()
        
        print("Testing BTC chain...")
        chain, spot, exp = dash.get_chain_data('BTC')
        if chain is not None:
            print(f"Spot: ${spot:,.2f}, Expiry: {exp}")
            print(chain.head(10))
        
        print("\nTesting ETH chain...")
        chain, spot, exp = dash.get_chain_data('ETH')
        if chain is not None:
            print(f"Spot: ${spot:,.2f}, Expiry: {exp}")
            print(chain.head(10))
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            pass
