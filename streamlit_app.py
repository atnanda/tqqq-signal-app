"""
Advanced QQQ/TQQQ Trading Dashboard — with Trade Log Table & Fixed yfinance Columns
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
from typing import Optional, List, Tuple

# --------------------------------------------------
# --- Configuration ---
# --------------------------------------------------

pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings('ignore')

INITIAL_INVESTMENT = Decimal('10000.00')
TICKER = 'QQQ'
EMA_PERIOD = 5
SMA_PERIOD = 200
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0

# --------------------------------------------------
# --- Data & Utility ---
# --------------------------------------------------

@st.cache_data(ttl=60 * 60 * 4, show_spinner=False)
def get_data(ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Download historical price data using yfinance, flattening MultiIndex columns if needed."""
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        data.reset_index(inplace=True)

        # --- FIX: Handle MultiIndex columns safely ---
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip().lower() if isinstance(col, tuple) else str(col).lower() for col in data.columns]
        else:
            data.columns = data.columns.str.lower()

        # Normalize column names for consistency
        rename_map = {
            'adj close': 'close',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        for old, new in rename_map.items():
            if old in data.columns and old != new:
                data.rename(columns={old: new}, inplace=True)

        # Validate essential columns
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        missing = required_cols - set(data.columns)
        if missing:
            st.warning(f"Missing columns: {missing}")

        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()


def get_most_recent_trading_day() -> date:
    """Return the most recent closed trading day (after 4PM ET)."""
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(ny_tz)
    market_close = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    target = now_ny.date()
    if now_ny < market_close:
        target -= timedelta(days=1)
    while target.weekday() > 4:
        target -= timedelta(days=1)
    return target


# --------------------------------------------------
# --- Indicators ---
# --------------------------------------------------

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df[f'SMA_{SMA_PERIOD}'] = ta.sma(df['close'], length=SMA_PERIOD)
    df[f'EMA_{EMA_PERIOD}'] = ta.ema(df['close'], length=EMA_PERIOD)
    df[f'ATR_{ATR_PERIOD}'] = ta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)
    return df


# --------------------------------------------------
# --- Signal Generation ---
# --------------------------------------------------

def generate_historical_signals(df: pd.DataFrame, target_date: date) -> dict:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.date

    if target_date not in df['date'].values:
        available_date = df['date'].iloc[-1]
        st.warning(f"No data for {target_date}, using {available_date}.")
        target_date = available_date

    current_row = df[df['date'] == target_date]
    prev_row = df[df['date'] < target_date].tail(1)

    if current_row.empty or prev_row.empty:
        raise ValueError("Not enough data to compute indicators.")

    prev_row = prev_row.iloc[0]
    prev_price = Decimal(str(prev_row['close']))
    prev_ema = Decimal(str(prev_row[f'EMA_{EMA_PERIOD}']))
    prev_atr = Decimal(str(prev_row[f'ATR_{ATR_PERIOD}']))
    prev_sma = Decimal(str(prev_row[f'SMA_{SMA_PERIOD}']))

    return {
        'prev_price': prev_price,
        'prev_ema': prev_ema,
        'prev_atr': prev_atr,
        'prev_sma': prev_sma,
        'prev_upper': prev_ema + Decimal(str(ATR_MULTIPLIER)) * prev_atr,
        'prev_lower': prev_ema - Decimal(str(ATR_MULTIPLIER)) * prev_atr,
    }


def generate_signal_for_date(df: pd.DataFrame, date_obj: date, price_override: Optional[float]) -> str:
    indicators = generate_historical_signals(df, date_obj)
    price = Decimal(str(price_override)) if price_override is not None else indicators['prev_price']

    dma_trend = 'Bullish' if price > indicators['prev_sma'] else 'Bearish'
    vasl_signal = (
        'Buy' if price > indicators['prev_upper']
        else 'Sell' if price < indicators['prev_lower']
        else 'Hold'
    )

    if vasl_signal == 'Buy' and dma_trend == 'Bullish':
        return 'BUY'
    elif vasl_signal == 'Sell':
        return 'SELL'
    else:
        return 'HOLD'


# --------------------------------------------------
# --- Backtesting Engine ---
# --------------------------------------------------

def backtest_strategy(df: pd.DataFrame, initial_balance: Decimal) -> Tuple[pd.DataFrame, List[Tuple]]:
    """Simulate the strategy and return DataFrame + trade log."""
    df = df.copy()
    balance = initial_balance
    position = Decimal('0.0')
    trade_log = []

    for i in range(1, len(df)):
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

    return df, trade_log


# --------------------------------------------------
# --- Visualization ---
# --------------------------------------------------

def plot_signals(df: pd.DataFrame, signal_date: date):
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
        date_index = df_plot[df_plot['date'].dt.date == signal_date]['date'].iloc[0]
        fig.add_trace(go.Scatter(
            x=[date_index, date_index],
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
# --- Streamlit App ---
# --------------------------------------------------

st.set_page_config(page_title="QQQ/TQQQ Strategy Dashboard", layout="wide")
st.title("📊 Advanced QQQ/TQQQ Strategy Dashboard")

data = get_data(TICKER, start='2010-01-01')
if data.empty:
    st.error("Failed to load data.")
    st.stop()

data = calculate_indicators(data)
first_date = data['date'].iloc[0].date()
last_date = data['date'].iloc[-1].date()
latest_closed = get_most_recent_trading_day()

if last_date < latest_closed:
    latest_closed = last_date
    st.info(f"Latest available data ends {last_date} (market closed or delayed).")

# Sidebar
with st.sidebar:
    st.header("1. Target Date & Price")

    min_trade_date = first_date + timedelta(days=SMA_PERIOD)
    target_date = st.date_input(
        "Select Signal Date",
        value=latest_closed,
        min_value=min_trade_date,
        max_value=last_date
    )

    if target_date.weekday() > 4:
        st.warning("Weekend selected; using most recent trading day if needed.")

    override_enabled = st.checkbox("Override Signal Price", value=False)
    if override_enabled:
        default_val = float(data['close'].iloc[-1])
        st.session_state['override_price'] = st.number_input(
            "QQQ Close ($)",
            value=round(default_val, 2),
            min_value=0.01,
            format="%.2f"
        )
    else:
        st.session_state['override_price'] = None

    if st.button("Clear Cache & Rerun"):
        st.cache_data.clear()
        st.rerun()

    st.header("2. Strategy Parameters")
    st.metric("Ticker", TICKER)
    st.metric("SMA", f"{SMA_PERIOD} days")
    st.metric("EMA", f"{EMA_PERIOD} days")
    st.metric("ATR", f"{ATR_PERIOD} days")
    st.metric("ATR Multiplier", ATR_MULTIPLIER)
    st.metric("Initial Investment", f"${float(INITIAL_INVESTMENT):,.2f}")
    st.caption(f"Backtest start: {first_date}")

# Main
try:
    signal = generate_signal_for_date(data, target_date, st.session_state['override_price'])
    st.subheader(f"Signal for {target_date}: **{signal}**")

    backtest_df, trade_log = backtest_strategy(data, INITIAL_INVESTMENT)
    st.metric("Final Portfolio Value", f"${backtest_df['final_value'].iloc[-1]:,.2f}")

    plot_signals(data, target_date)

    if trade_log:
        trade_log_df = pd.DataFrame(trade_log, columns=["Date", "Action", "Price", "Balance", "Position"])
        st.subheader("💹 Trade Log")
        st.dataframe(trade_log_df, use_container_width=True, hide_index=True)
    else:
        st.info("No trades were executed during this backtest window.")

except Exception as e:
    st.error(f"Error computing signals: {e}")
