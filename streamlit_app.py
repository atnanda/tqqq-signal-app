import pandas as pd
import yfinance as yf
# Using pandas-ta for easy cloud deployment (pure Python)
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
    Ensures column names are simple, clean, and explicitly set to the 
    standard lowercase names required by pandas-ta ('open', 'high', 'low', 'close').
    """
    # Create a new list for clean column names
    new_cols = []
    
    for col in df.columns:
        simple_name = None
        
        # 1. Handle MultiIndex/Tuple: Get the first element
        if isinstance(col, tuple):
            simple_name = col[0]
        # 2. Handle Simple Index/String
        elif isinstance(col, str):
            simple_name = col
        
        # 3. Process the name: Lowercase it
        if simple_name:
            new_cols.append(simple_name.lower())
        else:
            new_cols.append('')
            
    df.columns = new_cols
    
    # Remove any columns that ended up with empty names
    df = df.loc[:, df.columns != '']
    
    # Coerce required columns to numeric types for pandas-ta
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

# --- Data Fetching Functions ---

@st.cache_data(ttl=60*60*4) 
def get_realtime_price_live(ticker):
    """Fetches the current, non-historical price using yfinance Ticker info."""
    try:
        ticker_obj = yf.Ticker(ticker)
        price = ticker_obj.info.get('currentPrice') or ticker_obj.info.get('regularMarketPrice')
        if price is None:
             raise ValueError("Could not find a valid real-time price in yfinance info.")
        return price
    except Exception as e:
        st.error(f"Error fetching real-time price for {ticker}: {e}")
        return None

@st.cache_data(ttl=60*60*4) 
def fetch_historical_data(target_date):
    """
    Fetches historical data up to the trading day *prior* to the target date.
    """
    
    indicator_end_date = target_date 
    start_date_daily = indicator_end_date - timedelta(days=400) # Ensure enough data for 200 SMA
    
    daily_data = yf.download(TICKER, 
                             start=start_date_daily, 
                             end=indicator_end_date, 
                             interval="1d", 
                             progress=False)
    
    # Apply the column cleaning logic
    daily_data = clean_yfinance_columns(daily_data)
    
    daily_data.dropna(inplace=True)
    
    if daily_data.empty or daily_data.shape[0] < SMA_PERIOD:
        return None
        
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
# Calculate the default date (yesterday)
default_date_raw = today - timedelta(days=1)

# Check if yesterday was a Saturday (5) or Sunday (6)
if default_date_raw.weekday() == 5: # Saturday
    default_date = today - timedelta(days=2) # Set to Friday
elif default_date_raw.weekday() == 6: # Sunday
    default_date = today - timedelta(days=3) # Set to Friday
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
    st.sidebar.markdown(f"*(Indicators calculated using data up to: **{today - timedelta(days=1)}**)*")
    
    final_signal_price = get_realtime_price_live(TICKER)
    price_source_label = "Live Market Price"
    if final_signal_price is None:
        st.error("Cannot proceed without a live price.")
        st.stop()
        
else:
    target_date = st.sidebar.date_input(
        "Historical **Signal Date** (Close Price)",
        value=default_date,
        max_value=today
    )
    
    # Prevent user from manually selecting a weekend/holiday (though the default is fixed)
    if target_date.weekday() > 4:
        st.error(f"**{target_date.strftime('%Y-%m-%d')} is a weekend.** Please select a trading day.")
        st.stop()
        
    indicator_data_date = target_date - timedelta(days=1)
    st.sidebar.markdown(f"*(Indicators calculated using data up to: **{indicator_data_date.strftime('%Y-%m-%d')}**)*")
    signal_date_text = f"Close of **{target_date.strftime('%Y-%m-%d')}**"
    price_source_label = f"Historical Close ({target_date.strftime('%Y-%m-%d')})"
    
    final_signal_price = None

# 2. Execute Strategy
st.header(f"Signal based on {TICKER} Price at: {signal_date_text}")
st.info(f"Fetching historical data for indicators up to the day before {target_date.strftime('%Y-%m-%d')}...")

daily_data = fetch_historical_data(target_date)

if daily_data is None:
    st.error("FATAL ERROR: Insufficient data to calculate long-term indicators (200-Day SMA).")
    st.stop()
    
# Check for required columns again before calculation
if 'close' not in daily_data.columns or 'high' not in daily_data.columns or 'low' not in daily_data.columns:
    st.error("FATAL ERROR: Required columns ('close', 'high', 'low') are missing after cleaning.")
    st.stop()
    
# 3. Determine Final Signal Price (if not already set by real-time fetch)
if final_signal_price is None:
    try:
        data_with_signal_price = yf.download(TICKER, 
                                             start=target_date, 
                                             end=target_date + timedelta(days=1), 
                                             interval="1d", 
                                             progress=False)
        
        # Clean the columns of this one-row DataFrame too!
        data_with_signal_price = clean_yfinance_columns(data_with_signal_price)
        
        final_signal_price = data_with_signal_price['close'].iloc[-1].item()
    except Exception as e:
        st.error(f"FATAL ERROR: Could not find a valid close price for {target_date.strftime('%Y-%m-%d')}.")
        st


