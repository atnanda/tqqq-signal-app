import pandas as pd
import yfinance as yf
import talib as ta
from datetime import datetime, timedelta
import numpy as np
import warnings
import streamlit as st

# Suppress the specific FutureWarning about auto_adjust
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Configuration (Constants) ---
TICKER = "QQQ"
EMA_PERIOD = 5
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
SMA_PERIOD = 200

# --- Helper Functions (Same as before) ---

def get_market_end_date(target_date):
    """Calculates the end date for yfinance (day *after* the target trading day)."""
    return datetime.combine(target_date, datetime.min.time()) + timedelta(days=1)

@st.cache_data(ttl=60*60*4) # Cache the data fetch for 4 hours to avoid rate limits
def fetch_and_calculate_data(target_date):
    """Fetches data and calculates all indicators for the target date."""
    
    market_end_date = get_market_end_date(target_date)
    # Fetch enough data for 200 SMA + buffer
    start_date_daily = target_date - timedelta(days=400) 

    # --- 1. Get Daily Data ---
    daily_data = yf.download(TICKER, 
                             start=start_date_daily, 
                             end=market_end_date, 
                             interval="1d", 
                             progress=False)
    
    # CRITICAL CLEANING STEP FOR HISTORICAL DATA
    if daily_data.empty:
        return "ERROR: No data fetched from Yahoo Finance.", None
        
    # 1. Drop the last row (which may be a partial day or empty)
    if daily_data.index[-1].date() >= target_date:
        daily_data = daily_data.iloc[:-1].copy() 

    # 2. Drop all rows with NaNs (required before TA-Lib calculation)
    daily_data.dropna(inplace=True)
    
    # --- 2. Validation ---
    if daily_data.empty or daily_data.shape[0] < SMA_PERIOD:
        return f"ERROR: Insufficient clean data ({daily_data.shape[0]} points) to calculate 200-Day SMA.", None

    # --- 3. Intraday Data Proxy (The Close Price for the Target Date) ---
    intraday_data_proxy = daily_data.tail(1).copy() 

    # --- 4. Calculate Indicators ---
    try:
        # CRITICAL: Force to float, convert to NumPy, and then FLATTEN to ensure 1D array (N,)
        close_prices = daily_data['Close'].astype(float).to_numpy().flatten()
        high_prices = daily_data['High'].astype(float).to_numpy().flatten()
        low_prices = daily_data['Low'].astype(float).to_numpy().flatten()
        
        # Calculate Indicators
        daily_data['SMA_200'] = ta.SMA(close_prices, timeperiod=SMA_PERIOD)
        daily_data['EMA_5'] = ta.EMA(close_prices, timeperiod=EMA_PERIOD)
        daily_data['ATR'] = ta.ATR(high_prices, low_prices, close_prices, timeperiod=ATR_PERIOD)

        # Extract scalar values
        current_price = intraday_data_proxy['Close'].iloc[-1].item()
        current_sma_200 = daily_data['SMA_200'].iloc[-1].item()
        latest_ema_5 = daily_data['EMA_5'].iloc[-1].item()
        latest_atr = daily_data['ATR'].iloc[-1].item()

        indicators = {
            'current_price': current_price,
            'sma_200': current_sma_200,
            'ema_5': latest_ema_5,
            'atr': latest_atr
        }
        return "SUCCESS", indicators
        
    except Exception as e:
        return f"FATAL ERROR calculating indicators: {e}", None

def generate_signal(indicators):
    """Applies the trading strategy logic."""
    price = indicators['current_price']
    sma_200 = indicators['sma_200']
    ema_5 = indicators['ema_5']
    atr = indicators['atr']

    # Volatility Stop-Loss (VASL)
    vasl_trigger_level = ema_5 - (ATR_MULTIPLIER * atr)
    
    if price < vasl_trigger_level:
        final_signal = "**SELL TQQQ / CASH (Exit)**"
        conviction_status = "VASL Triggered"
    else:
        # Conviction Filter (DMA)
        dma_bull = (price >= sma_200)
        
        if dma_bull:
            conviction_status = "DMA - Bull"
            final_signal = "**BUY TQQQ**"
        else:
            conviction_status = "DMA - Bear"
            final_signal = "**BUY SQQQ**"

    return final_signal, conviction_status, vasl_trigger_level

# --- Streamlit Application Layout ---

st.title("TQQQ/SQQQ Daily Signal Generator")
st.markdown("---")

# 1. Date Input Sidebar
today = datetime.today().date()
yesterday = today - timedelta(days=1)
# Default to the most recent weekday's close (usually yesterday)
if yesterday.weekday() > 4:
    # Find the last Friday
    days_back = yesterday.weekday() - 4 
    default_date = yesterday - timedelta(days=days_back)
else:
    default_date = yesterday

st.sidebar.header("Select Date")
# The user selects the date for the signal
target_date = st.sidebar.date_input(
    "Signal Date (Close of Day)",
    value=default_date,
    max_value=today - timedelta(days=1)
)

# Check for weekends
if target_date.weekday() > 4:
    st.error(f"**{target_date.strftime('%Y-%m-%d')} is a weekend.** Please select a trading day.")
    st.stop()

# 2. Execute Strategy
st.header(f"Signal for QQQ Close on: {target_date.strftime('%Y-%m-%d')}")
st.info("Fetching data for QQQ...")

status, indicators = fetch_and_calculate_data(target_date)

if status.startswith("ERROR") or indicators is None:
    st.error(status)
else:
    final_signal, conviction_status, vasl_level = generate_signal(indicators)

    # 3. Display Results
    
    # Signal Box
    if "TQQQ" in final_signal:
        st.success(f"## {final_signal}")
    elif "SQQQ" in final_signal:
        st.warning(f"## {final_signal}")
    else: # SELL / CASH
        st.error(f"## {final_signal}")

    st.markdown("---")
    
    # Detail Metrics
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Current Price (QQQ Close)", f"${indicators['current_price']:.2f}")
    col2.metric("200-Day SMA", f"${indicators['sma_200']:.2f}")
    col3.metric("DMA Conviction", conviction_status)

    col_details, col_vasl = st.columns(2)
    
    with col_details:
        st.subheader("Volatility Metrics")
        st.markdown(f"**5-Day EMA:** ${indicators['ema_5']:.2f}")
        st.markdown(f"**14-Day ATR:** ${indicators['atr']:.2f}")

    with col_vasl:
        st.subheader("Stop-Loss (VASL)")
        if "Triggered" in conviction_status:
            st.error(f"**Trigger Level:** ${vasl_level:.2f}")
            st.error(f"Price is below the VASL level.")
        else:
            st.success(f"**Trigger Level:** ${vasl_level:.2f}")
            st.markdown("Price is **ABOVE** the VASL level.")

st.markdown("---")
st.markdown(f"Strategy Ticker: **{TICKER}** | ATR Multiplier: **{ATR_MULTIPLIER}**")