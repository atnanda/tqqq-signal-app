import pandas as pd
import yfinance as yf
import pandas_ta as pta 
from datetime import datetime, timedelta
import warnings
import streamlit as st
import requests # NEW: Required for the robust fallback
import time

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

# --- Core Helper Function for Cookie/Crumb Retrieval (Bypassing common yfinance errors) ---

@st.cache_data(ttl=60*60*2) # Cache for 2 hours
def get_yfinance_cookie(ticker="SPY"):
    """Fetches the necessary cookie and crumb for manual data download."""
    try:
        # A simple, robust call to get an access cookie (which contains the crumb)
        url = f'https://finance.yahoo.com/quote/{ticker}'
        response = requests.get(url, timeout=10)
        
        # Check if cookie exists
        cookie = response.cookies
        if not cookie:
             return None, None
             
        # Find the crumb/csrf token in the response text using lxml
        import lxml.html as html
        tree = html.fromstring(response.text)
        
        crumb_element = tree.xpath("//script[contains(., 'CrumbStore')]/text()")
        if not crumb_element:
            return cookie, None
        
        # Extract crumb from the script element
        crumb_line = [line for line in crumb_element[0].split('\n') if 'Crumb' in line]
        if not crumb_line:
             return cookie, None
             
        crumb = crumb_line[0].split(':')[1].strip().strip('"')
        return cookie, crumb

    except Exception as e:
        st.error(f"Failed to fetch Yahoo Finance cookie/crumb: {e}")
        return None, None

# --- Core Helper Function for Column Cleaning ---

def clean_yfinance_columns(df):
    """
    Ensures column names are clean and the DataFrame is ready for pandas-ta.
    """
    # 1. Standardize column names (assuming yfinance/requests returned standard columns)
    df.columns = [col.lower().replace('adj close', 'close') for col in df.columns]
    
    # 2. Select only the standardized columns and coerce types
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    df_clean = pd.DataFrame(index=df.index)
    for col in required_cols:
        if col in df.columns:
            df_clean[col] = pd.to_numeric(df[col], errors='coerce')
    
    # CRITICAL CHECK for the KeyError
    missing_cols = [c for c in ['high', 'low', 'close'] if c not in df_clean.columns]
    if missing_cols:
        st.error(f"Data cleaning failed: Missing columns {missing_cols}. Returning empty DataFrame.")
        return pd.DataFrame()
        
    df_clean.dropna(subset=['high', 'low', 'close'], inplace=True)
    
    return df_clean

# --- Data Fetching Functions ---

@st.cache_data(ttl=60*60*4) 
def fetch_historical_data(end_date):
    """
    Fetches historical data using a robust method (requests call with cookie/crumb) 
    if yfinance.download fails.
    """
    
    start_date = end_date - timedelta(days=400) 
    
    # First, try the highly defensive yfinance.download call
    try:
        daily_data = yf.download(
            TICKER, 
            start=start_date, 
            end=end_date, 
            interval="1d", 
            progress=False,
            auto_adjust=True, # Use auto_adjust to simplify column output
            timeout=10 
        )
        if not daily_data.empty and 'Close' in daily_data.columns:
            st.info("Data successfully fetched using yfinance.download.")
            return clean_yfinance_columns(daily_data)
        
    except Exception:
        # Pass to the requests fallback if yfinance fails
        st.warning("yfinance.download failed. Attempting manual requests fallback...")

    # --- FALLBACK: Manual Requests Call ---
    try:
        start_ts = int(time.mktime(start_date.timetuple()))
        end_ts = int(time.mktime(end_date.timetuple()))
        
        cookie, crumb = get_yfinance_cookie()
        if not cookie or not crumb:
            raise Exception("Could not retrieve necessary Yahoo Finance authentication.")
            
        api_url = (f"https://query1.finance.yahoo.com/v7/finance/download/{TICKER}"
                   f"?period1={start_ts}&period2={end_ts}&interval=1d"
                   f"&events=history&crumb={crumb}")

        response = requests.get(api_url, cookies=cookie, timeout=10)
        response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)

        from io import StringIO
        daily_data = pd.read_csv(StringIO(response.text), index_col='Date', parse_dates=True)
        
        st.info("Data successfully fetched using requests fallback.")
        return clean_yfinance_columns(daily_data)
        
    except Exception as e:
        st.error(f"FATAL: Both yfinance and requests fallback failed: {e}")
        return pd.DataFrame()

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
    indicator_end_date = today + timedelta(days=1) 
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
        
    indicator_end_date = target_date + timedelta(days=1) 
    signal_date_text = f"Close of **{target_date.strftime('%Y-%m-%d')}**"
    price_source_label = f"Historical Close ({target_date.strftime('%Y-%m-%d')})"
    final_signal_price = None

st.sidebar.markdown(f"*(Indicators calculated using data up to: **{indicator_end_date - timedelta(days=1)}**)*")

# 2. Execute Strategy
st.header(f"Signal based on {TICKER} Price at: {signal_date_text}")
st.info(f"Attempting to fetch historical data for indicators up to {indicator_end_date - timedelta(days=1)}...")

# Fetch data for indicators (includes the signal date's close)
daily_data = fetch_historical_data(indicator_end_date) 

# --- CRITICAL CHECKS ---
if daily_data.empty:
    st.error("FATAL ERROR: Insufficient data or data download failed. Cannot proceed with calculations.")
    # Show how the fetch failed (this is already handled inside the fetch_historical_data function)
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

