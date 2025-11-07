"""
Advanced QQQ/TQQQ Trading Dashboard — patched for stability and timezone correctness
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import pytz
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from decimal import Decimal, ROUND_HALF_UP
import warnings
from typing import Optional

# --------------------------------------------------
# --- Configuration ---
# --------------------------------------------------

pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings('ignore')

# Constants
INITIAL_INVESTMENT = Decimal('10000.00')
TICKER = 'QQQ'
EMA_PERIOD = 5
SMA_PERIOD = 200
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0

# --------------------------------------------------
# --- Utility Functions ---
# --------------------------------------------------

@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def get_data(ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Download historical data using yfinance"""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        data.reset_index(inplace=True)
        data.columns = data.columns.str.lower()
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def get_most_recent_trading_day() -> date:
    """Determines the most recent CLOSED trading day using NY market hours (16:00 ET)."""
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(ny_tz)
    market_close_hour = 16
    market_close_minute = 0

    target_date = now_ny.date()
    cutoff_time = now_ny.replace(hour=market_close_hour, minute=market_close_minute,
                                 second=0, microsecond=0)

    if now_ny < cutoff_time:
        target_date -= timedelta(days=1)

    # Step back on weekends
    while target_date.weekday() > 4:
        target_date -= timedelta(days=1)

    return target_date

# --------------------------------------------------
# --- Technical Indicator Calculations ---
# --------------------------------------------------

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SMA, EMA, and ATR indicators"""
    df[f'SMA_{SMA_PERIOD}'] = ta.sma(df['close'], length=SMA_PERIOD)
    df[f'EMA_{EMA_PERIOD}'] = ta.ema(df['close'], length=EMA_PERIOD)
    df[f'ATR_{ATR_PERIOD}'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)
    return df

# --------------------------------------------------
# --- Signal Generation ---
# --------------------------------------------------

def generate_historical_signals(df: pd.DataFrame, target_date: date) -> dict:
    """Return current signal and prior-day indicator data."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.date

    if target_date not in df['date'].values:
        available_date = df['date'].iloc[-1]
        st.warning(f"No data for {target_date}, using last available date {available_date}.")
        target_date = available_date

    current_row = df[df['date'] == target_date]
    prev_row = df[df['date'] < target_date].tail(1)

    if current_row.empty or prev_row.empty:
        raise ValueError("Insufficient data to calculate signals")

    prev_row = prev_row.iloc[0]
    current_row = current_row.iloc[0]

    prev_price = Decimal(str(prev_row['close']))
    prev_ema = Decimal(str(prev_row[f'EMA_{EMA_PERIOD}']))
    prev_atr = Decimal(str(prev_row[f'ATR_{ATR_PERIOD}']))
    prev_sma = Decimal(str(prev_row[f'SMA_{SMA_PERIOD}']))

    indicators = {
        'prev_price': prev_price,
        'prev_ema': prev_ema,
        'prev_atr': prev_atr,
        'prev_sma': prev_sma,
        'prev_upper': prev_ema + Decimal(str(ATR_MULTIPLIER)) * prev_atr,
        'prev_lower': prev_ema - Decimal(str(ATR_MULTIPLIER)) * prev_atr,
    }

    return indicators

def generate_signal_for_date(df: pd.DataFrame, date_obj: date, price_override: Optional[float]) -> str:
    """Generate trade signal for a given date."""
    indicators = generate_historical_signals(df, date_obj)
    price = Decimal(str(price_override)) if price_override is not None else indicators['prev_price']

    if price > indicators['prev_sma']:
        dma_trend = 'Bullish'
    elif price < indicators['prev_sma']:
        dma_trend = 'Bearish'
    else:
        dma_trend = 'Neutral'

    if price > indicators['prev_upper']:
        vasl_signal = 'Buy'
    elif price < indicators['prev_lower']:
        vasl_signal = 'Sell'
    else:
        vasl_signal = 'Hold'

    if vasl_signal == 'Buy' and dma_trend == 'Bullish':
        signal = 'BUY'
    elif vasl_signal == 'Sell':
        signal = 'SELL'
    else:
        signal = 'HOLD'

    return signal

# --------------------------------------------------
# --- Backtesting Engine ---
# --------------------------------------------------

def backtest_strategy(df: pd.DataFrame, initial_balance: Decimal) -> pd.DataFrame:
    """Simulate trading strategy."""
    df = df.copy()
    balance = initial_balance
    position = Decimal('0.0')
    trade_log = []

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        close_price = Decimal(str(curr['close']))

        signal = generate_signal_for_date(df.iloc[:i + 1], curr['date'], None)

        if signal == 'BUY' and position == 0:
            qty = (balance / close_price).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
            balance -= qty * close_price
            position += qty
            trade_log.append((curr['date'], 'BUY', float(close_price), float(balance), float(position)))

        elif signal == 'SELL' and position > 0:
            balance += position * close_price
            position = Decimal('0')
            trade_log.append((curr['date'], 'SELL', float(close_price), float(balance), float(position)))

        df.loc[df.index[i], 'signal'] = signal
        df.loc[df.index[i], 'balance'] = float(balance)
        df.loc[df.index[i], 'position'] = float(position)

    if position > 0:
        balance += position * Decimal(str(df.iloc[-1]['close']))

    df['final_value'] = df['balance'] + df['position'] * df['close']
    df['return_pct'] = df['final_value'].pct_change() * 100
    df['cumulative_return'] = (df['final_value'] / df['final_value'].iloc[0] - 1) * 100

    return df

# --------------------------------------------------
# --- Visualization ---
# --------------------------------------------------

def plot_signals(df: pd.DataFrame, signal_date: date):
    """Generate plot with signal highlights"""
    df_plot = df.copy()
    df_plot['date'] = pd.to_datetime(df_plot['date'])

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_plot['date'],
        open=df_plot['open'],
        high=df_plot['high'],
        low=df_plot['low'],
        close=df_plot['close'],
        name='Price'
    ))

    fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot[f'EMA_{EMA_PERIOD}'], name=f'EMA {EMA_PERIOD}', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot[f'SMA_{SMA_PERIOD}'], name=f'SMA {SMA_PERIOD}', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot[f'EMA_{EMA_PERIOD}'] + df_plot[f'ATR_{ATR_PERIOD}'] * ATR_MULTIPLIER, name='Upper Band', line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=df_plot['date'], y=df_plot[f'EMA_{EMA_PERIOD}'] - df_plot[f'ATR_{ATR_PERIOD}'] * ATR_MULTIPLIER, name='Lower Band', line=dict(color='gray', dash='dot')))

    if signal_date in df_plot['date'].dt.date.values:
        signal_date_index = df_plot[df_plot['date'].dt.date == signal_date]['date'].iloc[0]
        # Vertical line with legend
        fig.add_trace(go.Scatter(
            x=[signal_date_index, signal_date_index],
            y=[df_plot['low'].min(), df_plot['high'].max()],
            mode='lines',
            name='Signal Date',
            line=dict(color='yellow', width=2, dash='dash'),
            showlegend=True
        ))

    fig.update_layout(
        title=f"{TICKER} Price and Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# --- Streamlit UI ---
# --------------------------------------------------

st.set_page_config(page_title="QQQ/TQQQ Strategy Dashboard", layout="wide")
st.title("📊 Advanced QQQ/TQQQ Strategy Dashboard")

# Load data
data_for_backtest = get_data(TICKER, start='2010-01-01')

if data_for_backtest.empty:
    st.error("Failed to load QQQ data. Please try again later.")
    st.stop()

data_for_backtest = calculate_indicators(data_for_backtest)
first_available_date = data_for_backtest['date'].iloc[0].date()
latest_available_date = data_for_backtest['date'].iloc[-1].date()
latest_closed_date = get_most_recent_trading_day()

# Handle weekends/holidays
if latest_available_date < latest_closed_date:
    st.info(f"Latest available market data: {latest_available_date}. Market closed or data not yet available for {latest_closed_date}.")
    latest_closed_date = latest_available_date

# --- Sidebar ---
with st.sidebar:
    st.header("1. Target Date & Price")

    min_tradeable_date = first_available_date + timedelta(days=SMA_PERIOD)
    target_date = st.date_input(
        "Select Signal Date",
        value=latest_closed_date,
        min_value=min_tradeable_date,
        max_value=latest_available_date
    )

    if target_date.weekday() > 4:
        st.warning(f"{target_date} is a weekend; the most recent prior trading day will be used if available.")

    override_enabled = st.checkbox("Override Signal Price (manual QQQ Close)", value=False)
    if override_enabled:
        default_override = float(data_for_backtest['close'].iloc[-1])
        st.session_state['override_price'] = st.number_input(
            "Signal Price (QQQ Close $)",
            value=round(default_override, 2),
            min_value=0.01,
            format="%.2f"
        )
    else:
        st.session_state['override_price'] = None

    if st.button("Clear Data Cache & Rerun"):
        st.cache_data.clear()
        st.rerun()

    st.header("2. Strategy Parameters")
    st.metric("Ticker", TICKER)
    st.metric("SMA Period", f"{SMA_PERIOD} days")
    st.metric("EMA Period", f"{EMA_PERIOD} days")
    st.metric("ATR Period", f"{ATR_PERIOD} days")
    st.metric("ATR Multiplier", ATR_MULTIPLIER)
    st.metric("Initial Investment", f"${float(INITIAL_INVESTMENT):,.2f}")
    st.caption(f"Backtest starts from {first_available_date}")

# --- Main Content ---
try:
    signal = generate_signal_for_date(data_for_backtest, target_date, st.session_state['override_price'])
    st.subheader(f"Signal for {target_date}: **{signal}**")

    backtest_results = backtest_strategy(data_for_backtest, INITIAL_INVESTMENT)
    st.metric("Final Portfolio Value", f"${backtest_results['final_value'].iloc[-1]:,.2f}")

    plot_signals(data_for_backtest, target_date)

except Exception as e:
    st.error(f"Error computing signals: {e}")
