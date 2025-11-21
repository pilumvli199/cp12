#!/usr/bin/env python3
"""
DELTA EXCHANGE DASHBOARD (FIXED V2)
===================================
- Uses oi_value_usd instead of oi (contract count)
- Proper IV extraction from mark_vol and quotes
- Handles all edge cases
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
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'python-delta-dashboard'
        })

    def get_available_expiries(self, underlying='BTC'):
        """Get all available expiry dates for an underlying"""
        try:
            url = f"{BASE_URL}/v2/products"
            params = {
                'contract_types': 'call_options,put_options',
                'states': 'live'
            }
            res = self.session.get(url, params=params, timeout=15)
            if res.status_code != 200:
                logger.error(f"Products API error: {res.status_code}")
                return []
            
            products = res.json().get('result', [])
            expiry_map = {}  # expiry_date -> total OI
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
                            date_key = dt.strftime('%d-%m-%Y')
                            if date_key not in expiry_map:
                                expiry_map[date_key] = dt
                    except:
                        pass
            
            # Sort by date
            sorted_expiries = sorted(expiry_map.items(), key=lambda x: x[1])
            return sorted_expiries
            
        except Exception as e:
            logger.error(f"Error getting expiries: {e}")
            return []

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
            res = self.session.get(url, params=params, timeout=15)
            
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

    def safe_float(self, val, default=0.0):
        """Safely convert to float"""
        if val is None:
            return default
        try:
            return float(val)
        except:
            return default

    def get_chain_data(self, underlying='BTC'):
        """
        Fetch option chain with CORRECT data mapping:
        - OI: Use oi_value_usd (USD value, not contract count)
        - IV: Use mark_vol or quotes.ask_iv/bid_iv
        - LTP: Use mark_price
        """
        try:
            # 1. Get spot price
            spot_url = f"{BASE_URL}/v2/tickers/{underlying}USD"
            spot_res = self.session.get(spot_url, timeout=10)
            spot_price = 0
            if spot_res.status_code == 200:
                spot_data = spot_res.json().get('result', {})
                spot_price = self.safe_float(spot_data.get('mark_price') or spot_data.get('close'))
            
            if spot_price == 0:
                logger.error("Could not get spot price")
                return None, 0, ""

            logger.info(f"💰 {underlying} Spot: ${spot_price:,.2f}")

            # 2. Get available expiries
            expiries = self.get_available_expiries(underlying)
            if not expiries:
                logger.error("No expiries found")
                return None, 0, ""
            
            # Use nearest expiry
            expiry_str, expiry_dt = expiries[0]
            logger.info(f"📅 Using expiry: {expiry_str}")

            # 3. Fetch option chain with expiry filter
            url = f"{BASE_URL}/v2/tickers"
            params = {
                'contract_types': 'call_options,put_options',
                'underlying_asset_symbols': underlying,
                'expiry_date': expiry_str
            }
            
            logger.info(f"🔍 Fetching: {url}?{params}")
            res = self.session.get(url, params=params, timeout=20)
            
            if res.status_code != 200:
                logger.error(f"Tickers API error: {res.status_code} - {res.text[:200]}")
                return None, 0, ""
            
            tickers = res.json().get('result', [])
            logger.info(f"📊 Received {len(tickers)} tickers")
            
            if not tickers:
                logger.error("No tickers returned")
                return None, 0, ""

            # 4. Parse each ticker
            calls_data = {}
            puts_data = {}
            
            for t in tickers:
                sym = t.get('symbol', '')
                contract_type = t.get('contract_type', '')
                
                # Determine if Call or Put
                is_call = contract_type == 'call_options' or sym.startswith('C-')
                is_put = contract_type == 'put_options' or sym.startswith('P-')
                
                if not is_call and not is_put:
                    continue
                
                # Get strike price
                strike = self.safe_float(t.get('strike_price'))
                if strike == 0:
                    # Parse from symbol: C-BTC-85000-221125 or P-ETH-2700-221125
                    parts = sym.split('-')
                    if len(parts) >= 3:
                        strike = self.safe_float(parts[2])
                
                if strike == 0:
                    continue
                
                # === CRITICAL FIX: Use correct fields ===
                
                # LTP = mark_price (not last traded, but current fair value)
                ltp = self.safe_float(t.get('mark_price'))
                
                # OI = oi_value_usd (USD value shown on website, NOT contract count)
                # Website shows: $3.45M, $1.66M etc - this is oi_value_usd
                oi_usd = self.safe_float(t.get('oi_value_usd') or t.get('oi_value'))
                
                # Volume = turnover_usd
                vol_usd = self.safe_float(t.get('turnover_usd') or t.get('turnover'))
                
                # IV = mark_vol (this is the IV shown on website)
                # mark_vol is already in percentage format (95.5 = 95.5%)
                iv = self.safe_float(t.get('mark_vol'))
                
                # Also check quotes for bid/ask IV
                quotes = t.get('quotes') or {}
                if iv == 0 and quotes:
                    ask_iv = self.safe_float(quotes.get('ask_iv'))
                    bid_iv = self.safe_float(quotes.get('bid_iv'))
                    if ask_iv > 0 and bid_iv > 0:
                        iv = (ask_iv + bid_iv) / 2 * 100  # Convert decimal to %
                    elif ask_iv > 0:
                        iv = ask_iv * 100
                    elif bid_iv > 0:
                        iv = bid_iv * 100
                
                # Store data
                data = {
                    'ltp': ltp,
                    'oi': oi_usd,  # This is USD value now!
                    'iv': iv,
                    'vol': vol_usd
                }
                
                if is_call:
                    calls_data[strike] = data
                else:
                    puts_data[strike] = data
            
            logger.info(f"📈 Parsed: {len(calls_data)} calls, {len(puts_data)} puts")
            
            if not calls_data and not puts_data:
                logger.error("No valid options data")
                return None, 0, ""
            
            # 5. Combine into DataFrame
            all_strikes = sorted(set(list(calls_data.keys()) + list(puts_data.keys())))
            
            rows = []
            for strike in all_strikes:
                c = calls_data.get(strike, {})
                p = puts_data.get(strike, {})
                
                rows.append({
                    'strike': strike,
                    'c_vol': c.get('vol', 0),
                    'c_oi': c.get('oi', 0),
                    'c_iv': c.get('iv', 0),
                    'c_ltp': c.get('ltp', 0),
                    'p_ltp': p.get('ltp', 0),
                    'p_iv': p.get('iv', 0),
                    'p_oi': p.get('oi', 0),
                    'p_vol': p.get('vol', 0)
                })
            
            chain = pd.DataFrame(rows)
            chain = chain.set_index('strike')
            
            # 6. Filter strikes around ATM
            atm_strike = min(all_strikes, key=lambda x: abs(x - spot_price))
            atm_idx = all_strikes.index(atm_strike)
            
            start_idx = max(0, atm_idx - 8)
            end_idx = min(len(all_strikes), atm_idx + 9)
            filtered_strikes = all_strikes[start_idx:end_idx]
            
            chain = chain.loc[filtered_strikes]
            
            return chain, spot_price, expiry_dt.strftime('%d-%b')

        except Exception as e:
            logger.error(f"Chain Error: {e}")
            import traceback
            traceback.print_exc()
            return None, 0, ""

    def format_value(self, val, is_price=False, is_iv=False):
        """Format values like website: OI in K/M, IV as %, price with comma"""
        if val == 0 or pd.isna(val):
            return "-"
        
        if is_iv:
            return f"{val:.1f}"
        
        if is_price:
            if val >= 1000:
                return f"{val:,.0f}"
            elif val >= 1:
                return f"{val:.2f}"
            else:
                return f"{val:.4f}"
        
        # OI/Volume formatting (in USD)
        if val >= 1_000_000:
            return f"{val/1_000_000:.2f}M"
        elif val >= 1_000:
            return f"{val/1_000:.1f}K"
        else:
            return f"{val:.0f}"

    def generate_dashboard(self, underlying='BTC'):
        logger.info(f"🎨 Generating Dashboard for {underlying}...")
        
        candles = self.get_candles(f"{underlying}USD")
        chain_result = self.get_chain_data(underlying)
        
        if chain_result is None or chain_result[0] is None:
            logger.error("Failed to get chain data")
            return None
            
        chain_df, spot, exp = chain_result
        
        if candles is None or chain_df is None or chain_df.empty:
            logger.error("Missing data for dashboard")
            return None

        fig = plt.figure(figsize=(14, 18))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5])

        # Price Chart
        ax1 = fig.add_subplot(gs[0])
        mpf.plot(candles, type='candle', ax=ax1, style='yahoo', volume=False,
                 axtitle=f"{underlying} - 15m | Spot: ${spot:,.2f}")

        # Option Chain Table
        ax2 = fig.add_subplot(gs[1])
        ax2.axis('off')
        ax2.set_title(f"OPTION CHAIN (Exp: {exp})", color='yellow', fontsize=16, pad=10)

        table_data = []
        col_labels = ['Vol($)', 'OI($)', 'IV%', 'LTP', 'STRIKE', 'LTP', 'IV%', 'OI($)', 'Vol($)']

        for strike in chain_df.index:
            row = chain_df.loc[strike]
            
            r = [
                self.format_value(row['c_vol']),           # Call Volume
                self.format_value(row['c_oi']),            # Call OI (USD)
                self.format_value(row['c_iv'], is_iv=True), # Call IV
                self.format_value(row['c_ltp'], is_price=True), # Call LTP
                f"{int(strike):,}",                         # Strike
                self.format_value(row['p_ltp'], is_price=True), # Put LTP
                self.format_value(row['p_iv'], is_iv=True),  # Put IV
                self.format_value(row['p_oi']),             # Put OI (USD)
                self.format_value(row['p_vol'])             # Put Volume
            ]
            table_data.append(r)

        if not table_data:
            logger.error("No table data to display")
            return None

        table = ax2.table(cellText=table_data, colLabels=col_labels, 
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.0)

        # Style table
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#333333')
            else:
                cell.set_edgecolor('#555555')
                cell.set_facecolor('black')
                cell.set_text_props(color='white')

                # Highlight ATM strike
                try:
                    stk_str = table_data[row-1][4].replace(',', '')
                    stk = float(stk_str)
                    if abs(stk - spot) < (spot * 0.01):
                        cell.set_facecolor('#2A2A4A')
                        cell.set_text_props(color='#00FFFF', weight='bold')
                except:
                    pass

                # Color: Calls green, Puts red, Strike yellow
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


# ==================== TEST & RUN ====================
async def main():
    if not TELEGRAM_BOT_TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN missing!")
        return

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    dash = DeltaDashboard()
    logger.info("🚀 Dashboard Bot Started...")

    while True:
        try:
            # BTC
            img = dash.generate_dashboard('BTC')
            if img:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img,
                    caption="📊 #BTC Option Chain\n💡 Data: Delta Exchange")
                logger.info("✅ BTC Sent")
            
            await asyncio.sleep(10)

            # ETH
            img = dash.generate_dashboard('ETH')
            if img:
                await bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=img,
                    caption="📊 #ETH Option Chain\n💡 Data: Delta Exchange")
                logger.info("✅ ETH Sent")

            logger.info("💤 Waiting 2 mins...")
            await asyncio.sleep(120)

        except Exception as e:
            logger.error(f"Main Loop Error: {e}")
            await asyncio.sleep(60)


if __name__ == "__main__":
    # Test mode
    if os.getenv('TEST_MODE'):
        dash = DeltaDashboard()
        
        print("=" * 50)
        print("Testing BTC Option Chain")
        print("=" * 50)
        chain, spot, exp = dash.get_chain_data('BTC')
        if chain is not None:
            print(f"Spot: ${spot:,.2f}, Expiry: {exp}")
            print("\nSample data:")
            print(chain.head(5))
            print("\nGenerating image...")
            img = dash.generate_dashboard('BTC')
            if img:
                with open('btc_test.png', 'wb') as f:
                    f.write(img.read())
                print("✅ Saved: btc_test.png")
        
        print("\n" + "=" * 50)
        print("Testing ETH Option Chain")
        print("=" * 50)
        chain, spot, exp = dash.get_chain_data('ETH')
        if chain is not None:
            print(f"Spot: ${spot:,.2f}, Expiry: {exp}")
            print("\nSample data:")
            print(chain.head(5))
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            pass
