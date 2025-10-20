import pandas as pd
import yfinance as yf
import pandas_ta as pta 
from datetime import datetime, timedelta
import warnings
import streamlit as st

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration (Constants) ---
TICKER = "QQQ"
EMA_PERIOD = 5
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
SMA_PERIOD = 200

# Initialize Session State
if 'use_realtime' not in st.session_state:
    st.session_state['use_realtime'] = False

# --- Core Helper Function for Column Cleaning ---

def clean_yfinance_columns(df):
    """
    Ensures column names are clean and the DataFrame is ready for pandas-ta.
    Assumes df is NOT empty.
    
    NOTE: With auto_adjust=True in yfinance.download(), yfinance handles some 
    MultiIndex issues, but we keep this check for safety against non-standard output.
    """
    # 1. Map existing column names to clean lowercase names (Handling MultiIndex if needed)
    col_map = {}
    for col in df.columns:
        # Extract the base name, handling MultiIndex (tuples)
        simple_name = col[0] if isinstance(col, tuple) else col
        if simple_name:
            col_map[col] = simple_name.lower()
            
    df.rename(columns=col_map, inplace=True)
    
    # 2. Select only the standardized columns and coerce types
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'adj close']
    
    df_clean = pd.DataFrame(index=df.index)
    for col in required_cols:
        if col in df.columns:
            df_clean[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Prioritize 'close' or 'adj close' if available
    if 'adj close' in df_clean.columns:
        df_clean.rename(columns={'adj close': 'close'}, inplace=True)
    
    # CRITICAL FIX for the KeyError: Check if the required columns exist 
    missing_cols = [c for c in ['high', 'low', 'close'] if c not in df_clean.columns]
    if missing_cols:
        # This message will be caught and displayed by the calling function
        st.error(f"Data cleaning failed: Missing columns {missing_cols}.")
        return pd.DataFrame()
        
    df_clean.dropna(subset=['high', 'low', 'close'], inplace=True)
    
    return df_clean

# --- Data Fetching Functions ---

@st.cache_data(ttl=60*60*4) 
def get_realtime_price_live(ticker):
    """Fetches the current, non-historical price using yfinance Ticker info."""
    try:
        ticker_obj = yf.Ticker(ticker)
        # Try two different keys for robustness
        price = ticker_obj.info.get('currentPrice') or ticker_obj.info.get('regularMarketPrice')
        if price is None:
             raise ValueError("Could not find a valid real-time price in yfinance info.")
        return price
    except Exception as e:
        st.error(f"Error fetching real-time price for {ticker}: {e}")
        return None

@st.cache_data(ttl=60*60*4) 
def fetch_historical_data(end_date):
    """
    Fetches historical data up to the end_date (exclusive). 
    Includes defensive API parameters.
    """
    
    start_date = end_date - timedelta(days=400) # Ensure enough data for 200 SMA
    
    # --- YF DOWNLOAD ---
    try:
        daily_data = yf.download(
            TICKER, 
            start=start_date, 
            end=end_date, 
            interval="1d", 
            progress=False,
            # CRITICAL DEFENSE: auto_adjust simplifies column names to 'Close'
            auto_adjust=True, 
            # Prevents indefinite hanging in case of a blocked connection
            timeout=10 
        )
    except Exception as e:
        st.error(f"yfinance download failed (network/API): {e}")
        return pd.DataFrame()

    # CRITICAL CHECK: If yfinance failed, it often returns an empty DataFrame.
    if daily_data.empty:
        st.error("yfinance returned an empty dataset. Check ticker and date range.")
        return pd.DataFrame()
        
    # Apply the column cleaning logic
    daily_data = clean_yfinance_columns(daily_data)
    
    # Final check after cleaning
    if daily_data.empty or daily_data.shape[0] < SMA_PERIOD:
        # This will trigger the FATAL ERROR display in the main app block
        return pd.DataFrame() 
        
    return daily_data

# --- Calculation and Signal Functions ---

def calculate_indicators(data_daily, final_signal_price):
    """Calculates all indicators using pandas-ta."""
    
    # 1. 200-Day SMA (Column: 'SMA_200')
    data_daily.ta.sma(length=SMA_PERIOD, append=True)
    current_sma_200 = data_daily[f'SMA_{SMA_PERIOD}'].iloc[-1].item()
    
    # 2. 5-Day EMA (Column: 'EMA_5')
    data_daily.ta.ema(length=EMA_PERIOD, append=True)
    latest_ema_5 = data_daily[f'EMA_{EMA_PERIOD}'].iloc[-1].item()

    # 3. 14-Day ATR (Column: 'ATR_14')
    # This must work now as 'high', 'low', 'close' are confirmed and clean
    data_daily.ta.atr(length=ATR_PERIOD, append=True)
    latest_atr = data_daily[f'ATR_{ATR_PERIOD}'].iloc[-1].item()

    return {
        'current_price': final_signal_price,
        'sma_200': current_sma_200,
        'ema_5': latest_ema_5,
        'atr': latest_atr
    }

def generate_signal(indicators):
    """Applies the trading strategy logic (VASL & DMA)."""
    price = indicators['current_price']
    sma_200 = indicators['sma_200']
    ema_5 = indicators['ema_5']
    atr = indicators['atr']

    # 1. Volatility Stop-Loss (VASL)
    vasl_trigger_level = ema_5 - (ATR_MULTIPLIER * atr)
    
    if price < vasl_trigger_level:
        final_signal = "**SELL TQQQ / CASH (Exit)**"
        conviction_status = "VASL Triggered"
    else:
        # 2. Conviction Filter (DMA)
        dma_bull = (price >= sma_200)
        
        if dma_bull:
            conviction_status = "DMA - Bull"
            final_signal = "**BUY TQQQ**"
        else:
            conviction_status = "DMA - Bear"
            final_signal = "**BUY SQQQ**"

    return final_signal, conviction_status, vasl_trigger_level

# --- Streamlit Application Layout ---

def toggle_realtime():
    """Callback function to toggle the real-time state and clear cache."""
    st.cache_data.clear()

st.set_page_config(page_title="TQQQ/SQQQ Signal", layout="wide")
st.title("📈 TQQQ/SQQQ Daily Signal Generator")
st.markdown("A strategy based on **200-Day SMA** (Conviction) and **EMA/ATR** (Volatility Stop-Loss).")
st.markdown("---")

# 1. Date/Price Input and Real-time Button in Sidebar
today = datetime.today().date()

# --- WEEKEND FIX LOGIC ---
default_date_raw = today - timedelta(days=1)

# Adjust default date to the most recent Friday if needed
if default_date_raw.weekday() == 5: # Saturday
    default_date = today - timedelta(days=2) 
elif default_date_raw.weekday() == 6: # Sunday
    default_date = today - timedelta(days=3) 
elif default_date_raw.weekday() == 0: # Monday: Default is Sunday, go back to Friday
    default_date = today - timedelta(days=3)
else:
    default_date = default_date_raw
# --- END WEEKEND FIX LOGIC ---

st.sidebar.header("Data Source Selection")

# Real-time Toggle
realtime_toggle = st.sidebar.toggle(
    "Use **Real-Time** Market Price",
    value=st.session_state['use_realtime'],
    on_change=toggle_realtime,
    key='realtime_toggle'
)
st.session_state['use_realtime'] = realtime_toggle 

# Conditional Date Input
if st.session_state['use_realtime']:
    target_date = today
    signal_date_text = f"**Current Market** ({datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')})"
    
    final_signal_price = get_realtime_price_live(TICKER)
    price_source_label = "Live Market Price"
    indicator_end_date = today + timedelta(days=1) # Fetch data up to, but not including, tomorrow (i.e., include today's close if available)
    if final_signal_price is None:
        st.warning("Could not fetch live price. Using yesterday's close for the signal price.")
        
else:
    target_date = st.sidebar.date_input(
        "Historical **Signal Date** (Close Price)",
        value=default_date,
        max_value=today
    )
    
    if target_date.weekday() > 4:
        st.error(f"**{target_date.strftime('%Y-%m-%d')} is a weekend.** Please select a trading day.")
        st.stop()
        
    indicator_end_date = target_date + timedelta(days=1) # Fetch up to END date (exclusive) to include signal day's close
    signal_date_text = f"Close of **{target_date.strftime('%Y-%m-%d')}**"
    price_source_label = f"Historical Close ({target_date.strftime('%Y-%m-%d')})"
    final_signal_price = None

st.sidebar.markdown(f"*(Indicators calculated using data up to: **{indicator_end_date - timedelta(days=1)}**)*")

# 2. Execute Strategy
st.header(f"Signal based on {TICKER} Price at: {signal_date_text}")
st.info(f"Fetching historical data for indicators up to {indicator_end_date - timedelta(days=1)}...")

# Fetch data for indicators (includes the signal date's close)
daily_data = fetch_historical_data(indicator_end_date) 

# --- CRITICAL CHECKS ---
if daily_data.empty:
    st.error("FATAL ERROR: Insufficient data or data download failed. Cannot proceed with calculations.")
    st.stop()
    
# 3. Determine Final Signal Price (if not already set by real-time fetch)
if final_signal_price is None:
    # Use the close price of the last day in the fetched historical data (the signal date's close)
    try:
        final_signal_price = daily_data['close'].iloc[-1].item()
    except Exception as e:
        st.error(f"FATAL ERROR: Could not determine the close price for the signal date. {e}")
        st.stop()
    
# 4. Calculate Indicators and Generate Signal
try:
    indicators = calculate_indicators(daily_data, final_signal_price)
    final_signal, conviction_status, vasl_level = generate_signal(indicators)
except Exception as e:
    st.error(f"FATAL ERROR during indicator calculation or signal generation: {e}")
    st.stop()

# 5. Display Results
st.markdown("---")

# Signal Box Display
if "BUY TQQQ" in final_signal:
    st.success(f"## {final_signal}")
elif "BUY SQQQ" in final_signal:
    st.warning(f"## {final_signal}")
else: # SELL / CASH
    st.error(f"## {final_signal}")

st.markdown(f"**Price Source:** *{price_source_label}*")
st.markdown("---")

# Detail Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Signal Price (QQQ)", f"${indicators['current_price']:.2f}")
col2.metric("200-Day SMA", f"${indicators['sma_200']:.2f}")
col3.metric("DMA Conviction", conviction_status)

st.subheader("Strategy Details")
col_details, col_vasl = st.columns(2)

with col_details:
    st.markdown(f"**5-Day EMA:** ${indicators['ema_5']:.2f}")
    st.markdown(f"**14-Day ATR:** ${indicators['atr']:.2f}")

with col_vasl:
    if "Triggered" in conviction_status:
        st.error(f"**VASL Trigger Level:** ${vasl_level:.2f}")
        st.error(f"Price (${indicators['current_price']:.2f}) is below the stop-loss level.")
    else:
        st.success(f"**VASL Trigger Level:** ${vasl_level:.2f}")
        st.markdown(f"Price (${indicators['current_price']:.2f}) is **ABOVE** the VASL level.")

st.markdown("---")
st.markdown(f"Strategy Ticker: **{TICKER}** | ATR Multiplier: **{ATR_MULTIPLIER}**")


