import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as pta
from datetime import datetime, timedelta, date
import numpy as np
import warnings
from decimal import Decimal, getcontext
import pytz
import altair as alt

# --- Configuration & Setup ---
getcontext().prec = 50
warnings.filterwarnings("ignore")

TQQQ_INCEPTION_DATE = date(2010, 2, 9)
INITIAL_INVESTMENT = Decimal("10000.00")

# --- Helper Functions ---

def get_last_closed_trading_day():
    new_york_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(new_york_tz)
    target_date = now_et.date()
    if now_et.hour < 16:
        target_date -= timedelta(days=1)
    while target_date.weekday() > 4: 
        target_date -= timedelta(days=1)
    return target_date

@st.cache_data(ttl=3600)
def fetch_data(tickers, start_date, end_date):
    adj_start = start_date - timedelta(days=400)
    # Force auto_adjust=False to keep Adj Close
    data = yf.download(tickers, start=adj_start, end=end_date + timedelta(days=1), progress=False, auto_adjust=False)
    if data.empty: return pd.DataFrame()
    
    df = pd.DataFrame(index=data.index)
    for t in tickers:
        # Multi-index safe retrieval
        if ('Adj Close', t) in data.columns:
            df[f'{t}_close'] = data[('Adj Close', t)]
        elif ('Close', t) in data.columns:
            df[f'{t}_close'] = data[('Close', t)]
            
        if t == tickers[0]: # Underlying (QQQ)
            df['high'] = data[('High', t)]
            df['low'] = data[('Low', t)]
            df['close'] = df[f'{t}_close']
    return df.dropna()

def calculate_indicators(df, sma_p=200, sma_s=20, ema_p=5, atr_p=14):
    df = df.copy()
    df['SMA_200'] = pta.sma(df['close'], length=sma_p)
    df['SMA_20'] = pta.sma(df['close'], length=sma_s)
    df['EMA_5'] = pta.ema(df['close'], length=ema_p)
    df['SMA_20_slope'] = df['SMA_20'].diff(1)
    
    adx_df = pta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx_df['ADX_14']
    
    high_low = df['high'] - df['low']
    high_cp = np.abs(df['high'] - df['close'].shift(1))
    low_cp = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(span=atr_p, adjust=False).mean()
    
    return df

def run_strategy_backtest(df, start_date, params):
    df = df[df.index.date >= start_date].copy()
    portfolio_value = INITIAL_INVESTMENT
    current_ticker = 'CASH'
    shares = Decimal("0")
    cost_basis = Decimal("0")
    loss_carry = Decimal("0")
    yearly_gain = Decimal("0")
    
    history = []
    tax_logs = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        curr_date = row.name.date()
        
        prices = {
            'QQQ': Decimal(str(row['QQQ_close'])),
            'TQQQ': Decimal(str(row['TQQQ_close'])),
            'SQQQ': Decimal(str(row['SQQQ_close']))
        }
        
        # v22td Signal Logic
        target_ticker = 'CASH'
        is_bull = row['close'] >= row['SMA_200']
        vasl = row['EMA_5'] - (2.0 * row['ATR'])
        mini_vasl = row['EMA_5'] - (float(params['mini_vasl']) * row['ATR'])
        
        if is_bull:
            if row['ADX'] < params['adx_threshold']:
                target_ticker = 'CASH'
            elif row['close'] < vasl or row['close'] < mini_vasl or row['EMA_5'] < row['SMA_200']:
                target_ticker = 'CASH'
            else:
                target_ticker = 'TQQQ'
        else:
            if row['EMA_5'] > row['SMA_20'] and row['SMA_20_slope'] > 0:
                target_ticker = 'TQQQ'
            else:
                target_ticker = 'CASH'

        # Execution
        if target_ticker != current_ticker:
            if current_ticker != 'CASH':
                sell_val = shares * prices[current_ticker]
                realized = sell_val - cost_basis
                yearly_gain += realized
                portfolio_value = sell_val
                history.append({'Date': curr_date, 'Action': 'SELL', 'Ticker': current_ticker, 'Price': float(prices[current_ticker]), 'Value': float(portfolio_value)})
                shares = Decimal("0")
            
            if target_ticker != 'CASH':
                shares = portfolio_value / prices[target_ticker]
                cost_basis = portfolio_value
                history.append({'Date': curr_date, 'Action': 'BUY', 'Ticker': target_ticker, 'Price': float(prices[target_ticker]), 'Value': float(portfolio_value)})
            
            current_ticker = target_ticker

        if current_ticker != 'CASH':
            portfolio_value = shares * prices[current_ticker]
        df.at[row.name, 'Strategy_Value'] = float(portfolio_value)

        # Tax Settlement
        if (i == len(df)-1) or (df.index[i+1].year > row.name.year):
            taxable = max(Decimal("0"), yearly_gain - loss_carry)
            tax_due = taxable * Decimal(str(params['tax_rate']))
            if tax_due > 0:
                portfolio_value -= tax_due
                if current_ticker != 'CASH':
                    shares = portfolio_value / prices[current_ticker]
                history.append({'Date': curr_date, 'Action': 'TAX_PAID', 'Ticker': 'CASH', 'Price': 0.0, 'Value': float(portfolio_value)})
            tax_logs.append({'Year': row.name.year, 'Gain': float(yearly_gain), 'Tax': float(tax_due)})
            loss_carry = max(Decimal("0"), (loss_carry - yearly_gain) if yearly_gain > 0 else (loss_carry + abs(yearly_gain)))
            yearly_gain = Decimal("0")

    return df, pd.DataFrame(history), pd.DataFrame(tax_logs)

# --- UI ---

st.set_page_config(layout="wide", page_title="TQQQ v22td Alpha")

with st.sidebar:
    st.header("⚙️ Parameters")
    st_date = st.date_input("Start Date", date(2023, 1, 1))
    en_date = st.date_input("End Date", get_last_closed_trading_day())
    adx_t = st.slider("ADX Sideways Filter", 0.0, 25.0, 12.0)
    m_vasl = st.slider("Mini-VASL (ATR Mult)", 0.5, 2.5, 1.5)
    tax_r = st.number_input("Tax Rate", 0.0, 0.5, 0.25)

tickers = ["QQQ", "TQQQ", "SQQQ"]
raw_data = fetch_data(tickers, st_date, en_date)

if not raw_data.empty:
    processed_df = calculate_indicators(raw_data)
    results_df, trades, tax_history = run_strategy_backtest(processed_df, st_date, {'adx_threshold': adx_t, 'mini_vasl': m_vasl, 'tax_rate': tax_r})
    
    # Header Metrics
    last_row = processed_df.iloc[-1]
    curr_signal = trades.iloc[-1]['Ticker'] if not trades.empty else "CASH"
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signal", curr_signal)
    c2.metric("Portfolio", f"${results_df['Strategy_Value'].iloc[-1]:,.2f}")
    c3.metric("ADX", f"{last_row['ADX']:.2f}")
    c4.metric("Regime", "BULL" if last_row['close'] >= last_row['SMA_200'] else "BEAR")

    # Chart
    results_df['B&H_QQQ'] = (results_df['QQQ_close'] / results_df['QQQ_close'].iloc[0]) * 10000
    st.altair_chart(alt.Chart(results_df.reset_index().melt('Date', ['Strategy_Value', 'B&H_QQQ'])).mark_line().encode(
        x='Date:T', y='value:Q', color='variable:N'
    ).interactive(), use_container_width=True)

    # Tables
    tab1, tab2 = st.tabs(["📑 Trade History", "💰 Tax History"])
    tab1.dataframe(trades, use_container_width=True)
    tab2.dataframe(tax_history, use_container_width=True)
