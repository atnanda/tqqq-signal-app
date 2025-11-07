import pandas as pd
import yfinance as yf
import pandas_ta as pta
from datetime import datetime, timedelta
import warnings
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pytz
from decimal import Decimal, getcontext
import math

# Set precision for Decimal operations to high value (e.g., 50 places)
getcontext().prec = 50

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration (Constants) ---
TICKER = "QQQ"
EMA_PERIOD = 5
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
SMA_PERIOD = 200
# 🚨 Convert INITIAL_INVESTMENT to Decimal
INITIAL_INVESTMENT = Decimal("10000.00")
LEVERAGED_TICKER = "TQQQ"
INVERSE_TICKER = "SQQQ"

# TQQQ Inception Date (February 9, 2010)
TQQQ_INCEPTION_DATE = datetime(2010, 2, 9).date()

# Initialize Session State for date/price overrides
if 'override_price' not in st.session_state:
    st.session_state['override_price'] = None

# --- Timezone-Aware Date Function ---

def get_most_recent_trading_day():
    """Determines the most recent CLOSED trading day."""
    california_tz = pytz.timezone('America/Los_Angeles')
    now = datetime.now(california_tz)
    
    # Check if the market is currently open (9:30 AM to 1:00 PM PST)
    # Note: Stock market closes at 1:00 PM PST (4:00 PM EST)
    is_trading_day = now.weekday() < 5
    market_close_time = now.replace(hour=13, minute=0, second=0, microsecond=0) # 1:00 PM PST

    if is_trading_day and now > market_close_time:
        # Market closed today, use today's date
        return now.date()
    elif is_trading_day and now <= market_close_time:
        # Market is open or closed early, use yesterday's date
        return (now - timedelta(days=1)).date()
    else:
        # Weekend, use last Friday
        days_to_subtract = now.weekday() - 4 # Saturday is 5-4=1, Sunday is 6-4=2
        return (now - timedelta(days=days_to_subtract)).date()

# --- Data Fetching and Preparation ---

@st.cache_data(ttl=timedelta(hours=6))
def fetch_data(start_date, end_date):
    """Fetches and merges data for QQQ, TQQQ, and SQQQ."""
    
    # 1. Fetch QQQ data
    data = yf.download(TICKER, start=start_date, end=end_date, progress=False)
    
    # 🚨 FIX: Check for None or empty data immediately
    if data is None or data.empty:
        st.error(f"Failed to fetch base data for {TICKER} between {start_date} and {end_date}.")
        return pd.DataFrame()
        
    data.index = data.index.tz_localize(None) # Remove timezone info
    
    # 2. Calculate Indicators on QQQ (Base asset)
    data['EMA_5'] = pta.ema(data['Close'], length=EMA_PERIOD).apply(lambda x: Decimal(str(x)))
    data['SMA_200'] = pta.sma(data['Close'], length=SMA_PERIOD).apply(lambda x: Decimal(str(x)))
    data['ATR'] = pta.atr(data['High'], data['Low'], data['Close'], length=ATR_PERIOD).apply(lambda x: Decimal(str(x)))
    
    # Remove initial NaN values
    data.dropna(inplace=True)
    
    # 3. Fetch Leveraged/Inverse data (only need price data)
    leveraged_data = yf.download(LEVERAGED_TICKER, start=start_date, end=end_date, progress=False)
    inverse_data = yf.download(INVERSE_TICKER, start=start_date, end=end_date, progress=False)

    # 🚨 FIX: Check for None or empty leveraged/inverse data
    if leveraged_data is None or leveraged_data.empty:
         st.error(f"Failed to fetch leveraged data for {LEVERAGED_TICKER}.")
         return pd.DataFrame()
    if inverse_data is None or inverse_data.empty:
         st.error(f"Failed to fetch inverse data for {INVERSE_TICKER}.")
         return pd.DataFrame()
         
    # 4. Merge data into a single DataFrame
    df = data.copy()
    
    # Create dictionary columns for leveraged/inverse data
    # Use .loc to ensure we are only using the common index dates
    
    common_index = df.index.intersection(leveraged_data.index).intersection(inverse_data.index)
    df = df.loc[common_index]
    leveraged_data = leveraged_data.loc[common_index]
    inverse_data = inverse_data.loc[common_index]
    
    # Now that indices are aligned, apply should work fine
    df[f'{LEVERAGED_TICKER}_Data'] = leveraged_data.apply(
        lambda row: {
            'Open': row['Open'], 'Close': row['Close']
        }, axis=1
    )
    df[f'{INVERSE_TICKER}_Data'] = inverse_data.apply(
        lambda row: {
            'Open': row['Open'], 'Close': row['Close']
        }, axis=1
    )
    df[f'{TICKER}_Data'] = data.loc[common_index].apply(
        lambda row: {
            'Open': row['Open'], 'Close': row['Close']
        }, axis=1
    )

    # Convert prices in the main DataFrame to Decimal for safe calculation
    df['Close'] = df['Close'].apply(lambda x: Decimal(str(x)))
    df['High'] = df['High'].apply(lambda x: Decimal(str(x)))
    df['Low'] = df['Low'].apply(lambda x: Decimal(str(x)))
    df['Volume'] = df['Volume'].apply(lambda x: Decimal(str(x)))
    
    # Final check after merging/cleaning
    if df.empty:
         st.error("Data is empty after indicator calculation and merging.")
         return pd.DataFrame()

    return df

# --- Signal Generation Logic ---

def generate_signal(price, ema_5, sma_200, atr, current_holding_ticker):
    """
    Generates the current day's signal based on the strategy rules.
    This function is primarily for the Streamlit dashboard display (current signal).
    """
    
    # Ensure all inputs are Decimals
    price, ema_5, sma_200, atr = Decimal(str(price)), Decimal(str(ema_5)), Decimal(str(sma_200)), Decimal(str(atr))
    ATR_MULTIPLIER_D = Decimal(str(ATR_MULTIPLIER))
    
    # Determine DMA Regime based on Price vs 200-SMA
    dma_bull_price = (price >= sma_200)
    
    final_signal = "HOLD CASH / UNCERTAIN"
    trade_ticker = 'CASH'
    conviction_status = "Waiting for entry conditions"

    # --- PRIORITY 1: The VASL Exit (2.0x ATR) ---
    vasl_trigger_level = ema_5 - (ATR_MULTIPLIER_D * atr)

    if current_holding_ticker == LEVERAGED_TICKER and price < vasl_trigger_level:
        final_signal = "**SELL TQQQ / CASH (VASL Exit Triggered)**"
        trade_ticker = 'CASH'
        conviction_status = "VASL Triggered - Exiting TQQQ position"
        return final_signal, trade_ticker, conviction_status
    
    # --- PRIORITY 2: Regime-Based Entry/Exit (If not already triggered by VASL) ---
    if dma_bull_price:
        # --- START DMA BULL REGIME (Price >= 200 SMA) ---
        
        # **NEW RULE: Mini-VASL Exit (DMA Bull, but price drops near 5-EMA)**
        mini_vasl_exit_level = ema_5 - (Decimal("0.5") * atr)
        
        if current_holding_ticker == LEVERAGED_TICKER and price < mini_vasl_exit_level:
             final_signal = "**SELL TQQQ / CASH (Mini-VASL Exit Triggered)**"
             trade_ticker = 'CASH'
             conviction_status = "Mini-VASL Triggered - Exiting TQQQ position"
        else:
            # Continue with original DMA Bull logic if Mini-VASL is NOT triggered or not holding TQQQ
            dma_bull_ema = (ema_5 >= sma_200) # EMA CONFIRMATION FILTER
            
            if dma_bull_ema:
                # TQQQ ENTRY/HOLD
                final_signal = f"**BUY / HOLD {LEVERAGED_TICKER} (DMA Bull & EMA Confirmed)**"
                trade_ticker = LEVERAGED_TICKER
                conviction_status = f"Bullish: {TICKER} is over 200-SMA and 5-EMA is over 200-SMA."
            else:
                # EMA LAGGING: If already in TQQQ, HOLD. Otherwise, CASH (Lagging Exit Filter)
                if current_holding_ticker == LEVERAGED_TICKER:
                    final_signal = f"**HOLD {LEVERAGED_TICKER} (EMA Lagging)**"
                    trade_ticker = LEVERAGED_TICKER
                    conviction_status = f"Bullish (Lagging EMA): {TICKER} is over 200-SMA, but 5-EMA is lagging. Holding TQQQ."
                else:
                    final_signal = f"**HOLD CASH (EMA Lagging - Waiting)**"
                    trade_ticker = 'CASH'
                    conviction_status = f"Neutral (Waiting): {TICKER} is over 200-SMA, but 5-EMA is lagging. Remaining in CASH."

    else:
        # --- START DMA BEAR REGIME (Price < 200 SMA) ---
        
        # PRIORITY 3: Inverse EMA Exit
        if price > ema_5:
            final_signal = f"**SELL {INVERSE_TICKER} / CASH (Inverse EMA Exit)**"
            trade_ticker = 'CASH' 
            conviction_status = f"Bearish (Exit Inverse): {TICKER} is under 200-SMA, but price is above 5-EMA (Inverse Exit)."
        else:
            # SQQQ ENTRY/HOLD
            final_signal = f"**BUY / HOLD {INVERSE_TICKER} (DMA Bear & EMA Confirmed)**"
            trade_ticker = INVERSE_TICKER 
            conviction_status = f"Bearish: {TICKER} is under 200-SMA and price is under 5-EMA."
            
    return final_signal, trade_ticker, conviction_status


# --- Backtest Engine Class ---

class BacktestEngine(object):
    """
    Handles the backtesting logic, including calculating historical signals and running
    the simulation based on those signals.
    """

    def __init__(self, df, initial_investment, leveraged_ticker, inverse_ticker, atr_multiplier):
        # Ensure Decimal type for financial calculations
        self.df = df
        self.INITIAL_INVESTMENT = Decimal(str(initial_investment))
        self.LEVERAGED_TICKER = leveraged_ticker
        self.INVERSE_TICKER = inverse_ticker
        self.ATR_MULTIPLIER = Decimal(str(atr_multiplier))
        
        # Initialize trade history list
        self.trade_history = []


    def generate_historical_signals(self, initial_ticker):
        """
        Calculates the appropriate 'Trade_Ticker' for every day in the historical DataFrame.
        This is the core signal generation logic.
        """
        # Ensure Decimal constants are used for calculation
        ATR_MULTIPLIER_D = self.ATR_MULTIPLIER
        
        # Initialize historical columns
        self.df['Trade_Ticker'] = 'CASH'
        current_ticker_historical = initial_ticker

        for index, row in self.df.iterrows():
            # Current values for signal generation
            price = Decimal(str(row['Close']))
            ema_5 = Decimal(str(row['EMA_5']))
            sma_200 = Decimal(str(row['SMA_200']))
            atr = Decimal(str(row['ATR']))

            # --- DEBUG: Calculate and store Mini-VASL level for inspection ---
            mini_vasl_exit_level = ema_5 - (Decimal("0.5") * atr)
            self.df.loc[index, 'Mini_VASL_Exit_Level'] = mini_vasl_exit_level
            self.df.loc[index, 'Mini_VASL_Triggered_Indicator'] = \
                current_ticker_historical == self.LEVERAGED_TICKER and price < mini_vasl_exit_level
            # --- END DEBUG CALCULATION ---
            
            # Determine DMA Regime based on Price vs 200-SMA
            dma_bull_price = (price >= sma_200)

            trade_ticker = current_ticker_historical # Default to holding previous position

            # --- PRIORITY 1: The VASL Exit (2.0x ATR) ---
            vasl_trigger_level = ema_5 - (ATR_MULTIPLIER_D * atr)
            
            if current_ticker_historical == self.LEVERAGED_TICKER and price < vasl_trigger_level:
                trade_ticker = 'CASH'
            
            # --- PRIORITY 2: Regime-Based Entry/Exit (If not already triggered by VASL) ---
            elif dma_bull_price:
                # --- START DMA BULL REGIME ---
                
                # 🚨 PRIORITY 2A (FIXED MINI-VASL): Mini-VASL Exit (0.5x ATR, Only if holding TQQQ)
                if current_ticker_historical == self.LEVERAGED_TICKER and price < mini_vasl_exit_level:
                    trade_ticker = 'CASH'
                else:
                    # Continue with original DMA Bull logic if Mini-VASL is NOT triggered or not holding TQQQ
                    
                    # DMA Bull & EMA Confirmation Filter
                    dma_bull_ema = (ema_5 >= sma_200) 
                    
                    if dma_bull_ema:
                        # TQQQ ENTRY/HOLD
                        trade_ticker = self.LEVERAGED_TICKER
                    else:
                        # EMA LAGGING: If already in TQQQ, HOLD. Otherwise, CASH (Lagging Exit Filter)
                        trade_ticker = self.LEVERAGED_TICKER if current_ticker_historical == self.LEVERAGED_TICKER else 'CASH'
            
            else: # dma_bull_price is False (Price < 200 SMA)
                # DMA: BEAR - Check PRIORITY 3
                
                # 3. PRIORITY 3: Inverse EMA Exit
                if price > ema_5:
                    trade_ticker = 'CASH' 
                else:
                    # SQQQ ENTRY/HOLD
                    trade_ticker = self.INVERSE_TICKER 

            
            # --- END OF DAY: Record Signal and Update Holding ---
            self.df.loc[index, 'Trade_Ticker'] = trade_ticker
            current_ticker_historical = trade_ticker

        # --- DEBUG EXPORT: Save last 10 days of raw signals ---
        debug_df = self.df.tail(10).copy()
        
        # Convert index to a
