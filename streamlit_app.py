import pandas as pd
import yfinance as yf
# NOTE: Using pandas-ta for easy cloud deployment (pure Python, no C dependencies)
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

# Initialize Session State for the real-time toggle
if 'use_realtime' not in st.session_state:
    st.session_state['use_realtime'] = False

# --- Helper Functions ---

# Cache data for 4 hours to reduce API calls and speed up interaction
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
    
    # We need data up to the day *before* the signal price date.
    indicator_end_date = target_date 
    start_date_daily = indicator_end_date - timedelta(days=400) # Ensure enough data for 200 SMA
    
    daily_data = yf.download(TICKER, 
                             start=start_date_daily, 
                             end=indicator_end_date, 
                             interval="1d", 
                             progress=False)
    
    # Drop NaNs
    daily_data.dropna(inplace=True)
    
    if daily_data.empty or daily_data.shape[0] < SMA_PERIOD:
        return None
        
    return daily_data

def calculate_indicators(data_daily, final_signal_price):
    """Calculates all indicators using pandas-ta."""
    
    # --- Integration with pandas-ta ---
    # The .ta accessor and append=True automatically add columns to data_daily.
    
    # 1. 200-Day SMA (Column: 'SMA_200')
    data_daily.ta.sma(length=SMA_PERIOD, append=True)
    current_sma_200 = data_daily[f'SMA_{SMA_PERIOD}'].iloc[-1].item()
    
    # 2. 5-Day EMA (Column: 'EMA_5')
    data_daily.ta.ema(length=EMA_PERIOD, append=True)
    latest_ema_5 = data_daily[f'EMA_{EMA_PERIOD}'].iloc[-1].item()

    # 3. 14-Day ATR (Column: 'ATR_14')
    data_daily.ta.atr(length=ATR_PERIOD, append=True)
    latest_atr = data_daily[f'ATR_{ATR_PERIOD}'].iloc[-1].item()

    # --- End Integration ---

    return {
        'current_price': final_signal_price, # The price used for the signal
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
default_date = today - timedelta(days=max(1, today.weekday() - 4) if today.weekday() > 4 else 1)

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
    target_date = today # Indicators will use data up to yesterday's close
    signal_date_text = f"**Current Market** ({datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z')})"
    st.sidebar.markdown(f"*(Indicators calculated using data up to: **{today - timedelta(days=1)}**)*")
    
    final_signal_price = get_realtime_price_live(TICKER)
    price_source_label = "Live Market Price"
    if final_signal_price is None:
        st.error("Cannot proceed without a live price.")
        st.stop()
        
else:
    # Historical Close Mode: Date Input
    target_date = st.sidebar.date_input(
        "Historical **Signal Date** (Close Price)",
        value=default_date,
        max_value=today
    )
    
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

# Fetch historical data (used for SMA, EMA, ATR calculation)
daily_data = fetch_historical_data(target_date)

if daily_data is None:
    st.error("FATAL ERROR: Insufficient data to calculate long-term indicators (200-Day SMA).")
    st.stop()
    
# 3. Determine Final Signal Price (if not already set by real-time fetch)
if final_signal_price is None:
    # Fetch the close price for the target date
    try:
        data_with_signal_price = yf.download(TICKER, 
                                             start=target_date, 
                                             end=target_date + timedelta(days=1), 
                                             interval="1d", 
                                             progress=False)
        final_signal_price = data_with_signal_price['Close'].iloc[-1].item()
    except Exception as e:
        st.error(f"FATAL ERROR: Could not find a valid close price for {target_date.strftime('%Y-%m-%d')}.")
        st.stop()
    
# 4. Calculate Indicators and Generate Signal
try:
    indicators = calculate_indicators(daily_data, final_signal_price)
    final_signal, conviction_status, vasl_level = generate_signal(indicators)
except Exception as e:
    st.error(f"FATAL ERROR during calculation or signal generation: {e}")
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
