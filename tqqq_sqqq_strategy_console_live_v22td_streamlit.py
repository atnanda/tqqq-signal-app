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

# --- Configuration (Constants) ---
getcontext().prec = 50
warnings.filterwarnings("ignore")

# Strategy Constants
EMA_PERIOD = 5
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
SMA_PERIOD = 200
SMA_SHORT_PERIOD = 20
INITIAL_INVESTMENT = Decimal("10000.00")
TQQQ_INCEPTION_DATE = date(2010, 2, 9)

# --- Helper Functions ---

def get_last_closed_trading_day():
    new_york_tz = pytz.timezone('America/New_York')
    now_et = datetime.now(new_york_tz)
    target_date = now_et.date()
    # Market close is 4pm ET
    if now_et.hour < 16:
        target_date -= timedelta(days=1)
    while target_date.weekday() > 4: 
        target_date -= timedelta(days=1)
    return target_date

@st.cache_data(ttl=3600)
def fetch_historical_data(tickers, start_date, end_date):
    """Fetches data and handles yfinance MultiIndex + Adj Close logic."""
    adj_start = start_date - timedelta(days=400)
    try:
        data = yf.download(tickers, start=adj_start, end=end_date + timedelta(days=1), 
                           progress=False, auto_adjust=False)
        if data.empty: return pd.DataFrame()
        
        df_combined = pd.DataFrame(index=data.index)
        for ticker in tickers:
            # Handle Adj Close vs Close
            col = ('Adj Close', ticker) if ('Adj Close', ticker) in data.columns else ('Close', ticker)
            df_combined[f'{ticker}_close'] = data[col]
            
            if ticker == tickers[0]: # Primary Ticker (QQQ)
                df_combined['high'] = data[('High', ticker)]
                df_combined['low'] = data[('Low', ticker)]
                df_combined['close'] = df_combined[f'{ticker}_close']
                
        return df_combined.dropna()
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

def calculate_indicators(df, mini_vasl_mult):
    df = df.copy()
    # Trend & Momentum
    df['SMA_200'] = pta.sma(df['close'], length=SMA_PERIOD)
    df['SMA_20'] = pta.sma(df['close'], length=SMA_SHORT_PERIOD)
    df['EMA_5'] = pta.ema(df['close'], length=EMA_PERIOD)
    df['SMA_20_slope'] = df['SMA_20'].diff(1)
    
    # Volatility & Strength
    adx_df = pta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx_df['ADX_14']
    
    # ATR Calculation for Stops
    hl = df['high'] - df['low']
    hpc = np.abs(df['high'] - df['close'].shift(1))
    lpc = np.abs(df['low'] - df['close'].shift(1))
    df['ATR'] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).ewm(span=ATR_PERIOD, adjust=False).mean()
    
    # Stops
    df['VASL'] = df['EMA_5'] - (2.0 * df['ATR'])
    df['Mini_VASL'] = df['EMA_5'] - (mini_vasl_mult * df['ATR'])
    
    return df

def run_v22td_engine(df, start_date, adx_threshold, tax_rate):
    """Core logic from the v22td console script."""
    sim_df = df[df.index.date >= start_date].copy()
    
    portfolio_value = INITIAL_INVESTMENT
    current_ticker = 'CASH'
    shares = Decimal("0")
    cost_basis = Decimal("0")
    loss_carry = Decimal("0")
    yearly_gain = Decimal("0")
    
    history = []
    tax_logs = []
    
    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        prices = {'QQQ': Decimal(str(row['QQQ_close'])), 'TQQQ': Decimal(str(row['TQQQ_close']))}
        
        # Signal Logic
        is_bull = row['close'] >= row['SMA_200']
        target = 'CASH'
        
        if is_bull:
            # Bull: Stay in TQQQ unless sideways or stops hit
            if row['ADX'] >= adx_threshold and row['close'] >= row['VASL'] and row['close'] >= row['Mini_VASL']:
                target = 'TQQQ'
        else:
            # Bear: Only TQQQ if momentum relief rally is confirmed
            if row['EMA_5'] > row['SMA_20'] and row['SMA_20_slope'] > 0:
                target = 'TQQQ'
        
        # Trade Execution
        if target != current_ticker:
            if current_ticker != 'CASH':
                sell_val = shares * prices[current_ticker]
                yearly_gain += (sell_val - cost_basis)
                portfolio_value = sell_val
                history.append({'Date': row.name.date(), 'Action': f'SELL {current_ticker}', 'Value': float(portfolio_value)})
            
            if target != 'CASH':
                shares = portfolio_value / prices[target]
                cost_basis = portfolio_value
                history.append({'Date': row.name.date(), 'Action': f'BUY {target}', 'Value': float(portfolio_value)})
            current_ticker = target

        # Daily Value Update
        if current_ticker != 'CASH':
            portfolio_value = shares * prices[current_ticker]
        sim_df.at[row.name, 'Strategy_Value'] = float(portfolio_value)

        # Annual Tax Settlement
        if (i == len(sim_df)-1) or (sim_df.index[i+1].year > row.name.year):
            taxable = max(Decimal("0"), yearly_gain - loss_carry)
            tax_due = taxable * Decimal(str(tax_rate))
            if tax_due > 0:
                portfolio_value -= tax_due
                if current_ticker != 'CASH': shares = portfolio_value / prices[current_ticker]
                history.append({'Date': row.name.date(), 'Action': 'ANNUAL TAX', 'Value': float(portfolio_value)})
            
            tax_logs.append({'Year': row.name.year, 'Realized Gain': float(yearly_gain), 'Tax Paid': float(tax_due)})
            loss_carry = max(Decimal("0"), loss_carry - yearly_gain if yearly_gain > 0 else loss_carry + abs(yearly_gain))
            yearly_gain = Decimal("0")

    return sim_df, pd.DataFrame(history), pd.DataFrame(tax_logs)

# --- UI Implementation ---

st.set_page_config(layout="wide", page_title="TQQQ v22td Alpha")

# Sidebar - Matching tqqq_cash_enh visuals
with st.sidebar:
    st.header("Strategy Parameters")
    
    st.subheader("Tickers")
    TICKER = st.text_input("Underlying", "QQQ")
    LEV_TICKER = st.text_input("Leveraged", "TQQQ")
    
    st.subheader("Period")
    st_date = st.date_input("Start Date", date(2024, 1, 1))
    en_date = st.date_input("End Date", get_last_closed_trading_day())
    
    st.subheader("Logic & Taxes")
    adx_t = st.slider("ADX Sideways Filter", 0.0, 25.0, 12.0)
    m_vasl = st.slider("Mini-VASL ATR Mult", 0.5, 2.5, 1.5)
    tax_r = st.number_input("Annual Tax Rate", 0.0, 0.5, 0.25)
    
    st.markdown("---")
    use_log = st.checkbox("Logarithmic Scale", value=True)
    if st.button("Run Analysis", type="primary"): st.rerun()

# Execution
raw_df = fetch_historical_data([TICKER, LEV_TICKER], st_date, en_date)

if not raw_df.empty:
    df_with_inds = calculate_indicators(raw_df, m_vasl)
    results, trades, taxes = run_v22td_engine(df_with_inds, st_date, adx_t, tax_r)
    
    # 1. Metric Header
    last = df_with_inds.iloc[-1]
    last_signal = trades.iloc[-1]['Action'].split(' ')[-1] if not trades.empty else "CASH"
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Signal", last_signal)
    c2.metric(f"{TICKER} Price", f"${last['close']:.2f}")
    c3.metric("ADX (Trend Strength)", f"{last['ADX']:.1f}")
    c4.metric("Market Regime", "BULL" if last['close'] >= last['SMA_200'] else "BEAR")

    # 2. Performance Comparison Chart
    st.subheader("Growth of $10,000 (After-Tax Strategy vs Benchmarks)")
    
    # Normalize benchmarks
    results['B&H QQQ'] = (results['QQQ_close'] / results['QQQ_close'].iloc[0]) * 10000
    results['B&H TQQQ'] = (results['TQQQ_close'] / results['TQQQ_close'].iloc[0]) * 10000
    
    chart_data = results[['Strategy_Value', 'B&H QQQ', 'B&H TQQQ']].reset_index()
    chart_data.columns = ['Date', 'Strategy', 'B&H QQQ', 'B&H TQQQ']
    chart_melted = chart_data.melt('Date', var_name='Metric', value_name='Value')
    
    y_scale = alt.Scale(type='log' if use_log else 'linear')
    
    main_chart = alt.Chart(chart_melted).mark_line().encode(
        x=alt.X('Date:T', title=""),
        y=alt.Y('Value:Q', title="Portfolio Value ($)", scale=y_scale, axis=alt.Axis(format='$,.0f')),
        color=alt.Color('Metric:N', scale=alt.Scale(range=['#00CC96', '#636EFA', '#EF553B'])),
        tooltip=['Date', 'Metric', alt.Tooltip('Value:Q', format='$,.2f')]
    ).properties(height=500).interactive()
    
    st.altair_chart(main_chart, use_container_width=True)

    # 3. Tables - Styled like tqqq_cash_enh
    st.subheader("Backtest Summary")
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.write("**Annual Tax Breakdown**")
        st.dataframe(taxes.style.format({"Realized Gain": "${:,.2f}", "Tax Paid": "${:,.2f}"}), use_container_width=True)
    
    with col_r:
        st.write("**Recent Trade History**")
        st.dataframe(trades.tail(10), use_container_width=True)
