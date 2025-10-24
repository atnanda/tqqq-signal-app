import pandas as pd
import yfinance as yf
import pandas_ta as pta 
from datetime import datetime, timedelta
import warnings
import streamlit as st
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration (Constants) ---
TICKER = "QQQ"
EMA_PERIOD = 5
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
SMA_PERIOD = 200

# Initialize Session State for date/price overrides
if 'override_price' not in st.session_state:
    st.session_state['override_price'] = None

# --- Core Helper Function for Data Fetching and Cleaning (Fixes all yfinance column issues) ---

@st.cache_data(ttl=60*60*4) 
def fetch_historical_data(target_date):
    """
    Fetches historical data, using auto_adjust=False and forces the column
    names to the required lowercase structure to bypass persistent platform errors.
    """
    market_end_date = target_date + timedelta(days=1)
    start_date_daily = target_date - timedelta(days=400)
    
    st.info(f"Fetching historical data for {TICKER} up to {target_date.strftime('%Y-%m-%d')}...")
    
    try:
        # CRITICAL: auto_adjust=False to keep all OHLCV/Adj Close data
        daily_data = yf.download(
            TICKER, 
            start=start_date_daily, 
            end=market_end_date, 
            interval="1d", 
            progress=False,
            auto_adjust=False, 
            timeout=15 
        )
        
        if daily_data.empty:
            st.error("yfinance returned an empty dataset. Check ticker and date range.")
            return pd.DataFrame()
            
        # --- ROBUST COLUMN FIX: FORCE COLUMN NAMES ---
        
        # 1. Handle MultiIndex (if it exists) by dropping the top level
        if isinstance(daily_data.columns, pd.MultiIndex):
            daily_data.columns = daily_data.columns.droplevel(0)

        # 2. Rename columns using the **EXACT ORDER** yfinance returns them
        # Default yfinance order (6 columns): Open, High, Low, Close, Adj Close, Volume
        if daily_data.shape[1] >= 6:
            daily_data.columns = [
                'open',
                'high',
                'low',
                'non_adj_close', # Placeholder for non-adjusted Close
                'close',         # This MUST be the Adjusted Close for calculations
                'volume'
            ]
        elif daily_data.shape[1] == 5:
            # Fallback for 5 columns (likely only if auto_adjust=True was somehow used)
            daily_data.columns = ['open', 'high', 'low', 'close', 'volume']
            
        # --- END ROBUST COLUMN FIX ---

        # 3. Check for missing critical columns
        required_cols = ['high', 'low', 'close', 'open', 'volume']
        missing_cols = [c for c in ['high', 'low', 'close'] if c not in daily_data.columns]
        
        if missing_cols:
             st.error(f"Data cleaning failed: Missing OHLC columns {missing_cols}. DataFrame shape: {daily_data.shape}")
             return pd.DataFrame()

        # 4. Final data preparation
        daily_data = daily_data[daily_data.index.date <= target_date]
        # Only select required columns and ensure they are float after dropping NaNs
        data_for_indicators = daily_data[required_cols].dropna().astype(float)
        
        # Check for sufficient data (need enough rows for ATR's initial calculations)
        if data_for_indicators.shape[0] < SMA_PERIOD:
            st.error(f"FATAL ERROR: Insufficient clean data ({data_for_indicators.shape[0]} rows) to calculate {SMA_PERIOD}-Day SMA.")
            return pd.DataFrame() 
            
        return data_for_indicators

    except Exception as e:
        st.error(f"FATAL ERROR during data download: {e}")
        return pd.DataFrame()


# --- Calculation and Signal Functions (Fixes ATR access error) ---

def calculate_indicators(data_daily, current_price):
    """Calculates all indicators using pandas-ta."""
    
    # 1. 200-Day SMA
    data_daily.ta.sma(length=SMA_PERIOD, append=True)
    current_sma_200 = data_daily[f'SMA_{SMA_PERIOD}'].iloc[-1].item()
    
    # 2. 5-Day EMA
    data_daily.ta.ema(length=EMA_PERIOD, append=True)
    latest_ema_5 = data_daily[f'EMA_{EMA_PERIOD}'].iloc[-1].item()

    # 3. 14-Day ATR
    atr_col_name = f'ATR_{ATR_PERIOD}'
    # The calculation needs high, low, close columns to be present and clean
    data_daily.ta.atr(length=ATR_PERIOD, append=True)
    
    # CRITICAL: Check for the ATR column explicitly after calculation
    if atr_col_name not in data_daily.columns:
        # This confirms pandas-ta failed to produce the column
        raise KeyError(f"Failed to calculate or access indicator: '{atr_col_name}'. Check data integrity for ATR.")

    latest_atr = data_daily[atr_col_name].iloc[-1].item()

    return {
        'current_price': current_price,
        'sma_200': current_sma_200,
        'ema_5': latest_ema_5,
        'atr': latest_atr
    }


def generate_signal(indicators):
    """Applies the defined trading strategy logic: VASL and DMA."""
    price = indicators['current_price']
    sma_200 = indicators['sma_200']
    ema_5 = indicators['ema_5']
    atr = indicators['atr']

    # --- 1. Volatility Stop-Loss (VASL) ---
    vasl_trigger_level = ema_5 - (ATR_MULTIPLIER * atr)
    
    if price < vasl_trigger_level:
        final_signal = "**SELL TQQQ / CASH (Exit)**"
        conviction_status = "VASL Triggered"
    else:
        # --- 2. Conviction Filter (DMA) ---
        dma_bull = (price >= sma_200)
        
        if dma_bull:
            conviction_status = "DMA - Bull (LONG TQQQ Default)"
            final_signal = "**BUY TQQQ**"
        else:
            conviction_status = "DMA - Bear (CASH/SQQQ Default)"
            final_signal = "**BUY SQQQ**"

    return final_signal, conviction_status, vasl_trigger_level


# --- Streamlit Application Layout ---

def get_most_recent_trading_day():
    today = datetime.today().date()
    target_date = today
    while target_date.weekday() > 4: # 5=Saturday, 6=Sunday
        target_date -= timedelta(days=1)
    return target_date

def display_app():
    
    st.set_page_config(page_title="TQQQ/SQQQ Signal", layout="wide")
    st.title("📈 TQQQ/SQQQ Daily Signal Generator")
    st.markdown("Strategy based on **200-Day SMA** (Conviction) and **5-Day EMA/14-Day ATR** (Volatility Stop-Loss).")
    st.markdown("---")

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("1. Target Date & Price")
        
        default_date = get_most_recent_trading_day()
        target_date = st.date_input(
            "Select Signal Date",
            value=default_date,
            max_value=default_date
        )
        
        if target_date.weekday() > 4:
            st.error(f"**{target_date.strftime('%Y-%m-%d')} is a weekend.** Please select a trading day.")
            st.stop()
        
        st.session_state['override_price'] = st.number_input(
            "Optional: Override Signal Price ($)",
            value=None,
            min_value=0.01,
            format="%.2f",
            help="Enter a price here to manually test the strategy at a specific level."
        )

        st.header("2. Strategy Parameters")
        st.metric("Ticker", TICKER)
        st.metric("SMA Period (DMA)", f"{SMA_PERIOD} days")
        st.metric("EMA Period (VASL)", f"{EMA_PERIOD} days")
        st.metric("ATR Period (VASL)", f"{ATR_PERIOD} days")
        st.metric("ATR Multiplier (VASL)", ATR_MULTIPLIER)

    # --- Main Logic ---

    # 1. Data Fetch
    data_for_indicators = fetch_historical_data(target_date)

    if data_for_indicators.empty:
        st.error("FATAL ERROR: Signal calculation aborted due to insufficient or missing data.")
        st.stop()

    # 2. Determine Final Signal Price
    final_signal_price = st.session_state['override_price']
    price_source_label = "Forced Override"
    
    if final_signal_price is None:
        # Get the market price (Last Close/Adj Close) for the target date
        if not data_for_indicators.empty and data_for_indicators.index[-1].date() == target_date:
            final_signal_price = data_for_indicators['close'].iloc[-1].item()
            price_source_label = f"Close (Adjusted) of {target_date.strftime('%Y-%m-%d')}"
        else:
             st.error(f"FATAL ERROR: Could not find the Close price for {target_date.strftime('%Y-%m-%d')} in fetched data.")
             st.stop()

    # 3. Calculate and Generate Signal
    try:
        indicators = calculate_indicators(data_for_indicators, final_signal_price)
        final_signal, conviction_status, vasl_level = generate_signal(indicators)
    except Exception as e:
        # Catch the specific ATR error here for a clean display
        st.error(f"FATAL ERROR during indicator calculation or signal generation: {e}")
        st.stop()

    # --- 4. Display Results ---

    st.header(f"Signal based on {TICKER} Price at: {target_date.strftime('%Y-%m-%d')}")
    st.markdown(f"**Execution Price Source:** *{price_source_label}*")
    
    st.markdown("---")

    # Signal Box Display
    if "BUY TQQQ" in final_signal:
        st.success(f"## {final_signal}")
    elif "BUY SQQQ" in final_signal:
        st.warning(f"## {final_signal}")
    else: # SELL / CASH
        st.error(f"## {final_signal}")

    st.markdown("---")

    # Detail Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Signal Price (QQQ)", f"${indicators['current_price']:.2f}")
    col2.metric("200-Day SMA", f"${indicators['sma_200']:.2f}")
    col3.metric("DMA Conviction", conviction_status.split('(')[0].strip()) 

    st.subheader("Volatility Stop-Loss (VASL) Details")
    col_vasl_level, col_vasl_status = st.columns(2)

    col_vasl_level.metric("5-Day EMA", f"${indicators['ema_5']:.2f}")
    col_vasl_level.metric("14-Day ATR", f"${indicators['atr']:.2f}")

    if "Triggered" in conviction_status:
        col_vasl_status.error(f"**VASL Trigger Level:** ${vasl_level:.2f}")
        col_vasl_status.error(f"Price (${indicators['current_price']:.2f}) is **BELOW** the stop-loss level.")
    else:
        col_vasl_status.success(f"**VASL Trigger Level:** ${vasl_level:.2f}")
        col_vasl_status.markdown(f"Price (${indicators['current_price']:.2f}) is **ABOVE** the VASL level.")

if __name__ == "__main__":
    display_app()


