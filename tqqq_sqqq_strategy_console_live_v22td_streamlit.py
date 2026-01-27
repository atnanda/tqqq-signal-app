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
    market_close_hour = 16 
    target_date = now_et.date()
    cutoff_time = now_et.replace(hour=market_close_hour, minute=0, second=0, microsecond=0)
    if now_et < cutoff_time:
        target_date -= timedelta(days=1)
    while target_date.weekday() > 4: 
        target_date -= timedelta(days=1)
    return target_date

def get_default_ytd_start_date(today_date):
    return date(today_date.year, 1, 1)

@st.cache_data(ttl=24*3600)
def fetch_historical_data(end_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER):
    start_date = TQQQ_INCEPTION_DATE
    tickers = [TICKER, LEVERAGED_TICKER, INVERSE_TICKER]
    market_end_date = end_date + timedelta(days=1) 
    try:
        all_data = yf.download(tickers, start=start_date, end=market_end_date, interval="1d", progress=False, auto_adjust=False)
        if all_data.empty: return pd.DataFrame()
        df_combined = pd.DataFrame(index=all_data.index)
        for ticker in tickers:
            close_col = ('Adj Close', ticker) if ('Adj Close', ticker) in all_data.columns else ('Close', ticker)
            if close_col in all_data.columns:
                df_combined[f'{ticker}_close'] = all_data[close_col]
            if ticker == TICKER:
                for metric in ['High', 'Low']: 
                    if (metric, ticker) in all_data.columns:
                        df_combined[metric.lower()] = all_data[metric][ticker] 
        df_combined['close'] = df_combined[f'{TICKER}_close']
        required_cols = ['close', 'high', 'low', f'{LEVERAGED_TICKER}_close', f'{INVERSE_TICKER}_close']
        df_combined.dropna(subset=required_cols, inplace=True)
        df_combined = df_combined[df_combined.index.date <= end_date]
        return df_combined 
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

def fetch_live_price(ticker):
    try:
        t = yf.Ticker(ticker)
        return t.fast_info['lastPrice']
    except: return None

def calculate_true_range_and_atr(df, atr_period):
    high_minus_low = df['high'] - df['low']
    high_minus_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_minus_prev_close = np.abs(df['low'] - df['close'].shift(1))
    true_range = pd.DataFrame({'hl': high_minus_low, 'hpc': high_minus_prev_close, 'lpc': low_minus_prev_close}).max(axis=1)
    return true_range.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean()

@st.cache_data(ttl=6*3600)
def calculate_indicators(data_daily, target_date, current_price):
    df = data_daily[data_daily.index.date <= target_date].copy()
    df.ta.sma(length=SMA_PERIOD, append=True)
    df.ta.sma(length=SMA_SHORT_PERIOD, append=True)
    df.ta.ema(length=EMA_PERIOD, append=True)
    adx_df = pta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx_df['ADX_14']
    df['SMA_20_slope'] = df[f'SMA_{SMA_SHORT_PERIOD}'].diff(1)
    df['ATR'] = calculate_true_range_and_atr(df, ATR_PERIOD)
    
    return {
        'current_price': current_price,
        'sma_200': df[f'SMA_{SMA_PERIOD}'].iloc[-1],
        'sma_20': df[f'SMA_{SMA_SHORT_PERIOD}'].iloc[-1],
        'sma_20_slope': df['SMA_20_slope'].iloc[-1],
        'ema_5': df[f'EMA_{EMA_PERIOD}'].iloc[-1],
        'atr': df['ATR'].ffill().iloc[-1],
        'adx': df['ADX'].iloc[-1]
    }

@st.cache_data(ttl=24*3600)
def generate_historical_signals(df, LEVERAGED_TICKER, mini_vasl_mult, adx_threshold):
    df = df.copy()
    df.ta.sma(length=SMA_PERIOD, append=True)
    df.ta.sma(length=SMA_SHORT_PERIOD, append=True)
    df.ta.ema(length=EMA_PERIOD, append=True)
    adx_df = pta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx_df['ADX_14']
    df['SMA_20_slope'] = df[f'SMA_{SMA_SHORT_PERIOD}'].diff(1)
    df['ATR'] = calculate_true_range_and_atr(df, ATR_PERIOD)
    
    df['Trade_Ticker'] = 'CASH'
    for index, row in df.iterrows():
        if pd.isna(row[f'SMA_{SMA_PERIOD}']): continue
        
        price, ema5, sma200, sma20, slope, atr, adx = row['close'], row[f'EMA_{EMA_PERIOD}'], row[f'SMA_{SMA_PERIOD}'], row[f'SMA_{SMA_SHORT_PERIOD}'], row['SMA_20_slope'], row['ATR'], row['ADX']
        
        vasl = ema5 - (2.0 * atr)
        mini_vasl = ema5 - (mini_vasl_mult * atr)
        
        if price >= sma200:
            if adx < adx_threshold or price < vasl or price < mini_vasl or ema5 < sma200:
                df.at[index, 'Trade_Ticker'] = 'CASH'
            else:
                df.at[index, 'Trade_Ticker'] = LEVERAGED_TICKER
        else:
            df.at[index, 'Trade_Ticker'] = LEVERAGED_TICKER if (ema5 > sma20 and slope > 0) else 'CASH'
            
    return df.dropna(subset=[f'SMA_{SMA_PERIOD}'])

@st.cache_data(ttl=6*3600)
def run_tax_sim(signals_df, start_date, TICKER, LEVERAGED_TICKER, tax_rate, ltcg_rate):
    sim_df = signals_df[signals_df.index.date >= start_date].copy()
    portfolio_value, shares, current_ticker, total_tax = INITIAL_INVESTMENT, Decimal("0"), 'CASH', Decimal("0")
    cost_basis, loss_carry = Decimal("0"), Decimal("0")
    yearly_gain = Decimal("0")
    trade_hist, tax_logs = [], []
    
    tax_rate_dec = Decimal(str(tax_rate))

    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        day_prices = {t: Decimal(str(row[f'{t}_close'])) for t in [TICKER, LEVERAGED_TICKER]}
        trade_ticker = row['Trade_Ticker']
        
        if trade_ticker != current_ticker:
            if current_ticker != 'CASH':
                sell_val = shares * day_prices[current_ticker]
                yearly_gain += (sell_val - cost_basis)
                portfolio_value = sell_val
                trade_hist.append({'Date': row.name.date(), 'Action': f"SELL {current_ticker}", 'Asset': current_ticker, 'Price': float(day_prices[current_ticker]), 'Portfolio Value': float(portfolio_value)})
                shares = Decimal("0")
            
            if trade_ticker != 'CASH':
                buy_price = day_prices[trade_ticker]
                shares = portfolio_value / buy_price
                cost_basis = portfolio_value
                trade_hist.append({'Date': row.name.date(), 'Action': f"BUY {trade_ticker}", 'Asset': trade_ticker, 'Price': float(buy_price), 'Portfolio Value': float(portfolio_value)})
            current_ticker = trade_ticker

        if current_ticker != 'CASH': portfolio_value = shares * day_prices[current_ticker]
        
        # End of Year Tax Logic
        is_last = (i == len(sim_df)-1) or (sim_df.index[i+1].year > row.name.year)
        if is_last:
            tax_due = max(Decimal("0"), (yearly_gain - loss_carry) * tax_rate_dec)
            if tax_due > 0:
                portfolio_value -= tax_due
                total_tax += tax_due
                if current_ticker != 'CASH': shares = portfolio_value / day_prices[current_ticker]
                trade_hist.append({'Date': row.name.date(), 'Action': "ANNUAL TAX", 'Asset': 'CASH', 'Price': 0.0, 'Portfolio Value': float(portfolio_value)})
            
            tax_logs.append({'Year': row.name.year, 'Realized Gain': float(yearly_gain), 'Tax Paid': float(tax_due)})
            loss_carry = max(Decimal("0"), loss_carry - yearly_gain) if yearly_gain > 0 else loss_carry + abs(yearly_gain)
            yearly_gain = Decimal("0")
            
        sim_df.at[row.name, 'Portfolio_Value'] = float(portfolio_value)

    # B&H Logic (Simple LTCG at end)
    bh_q_gross = float(INITIAL_INVESTMENT) * (sim_df[f'{TICKER}_close'].iloc[-1] / sim_df[f'{TICKER}_close'].iloc[0])
    bh_t_gross = float(INITIAL_INVESTMENT) * (sim_df[f'{LEVERAGED_TICKER}_close'].iloc[-1] / sim_df[f'{LEVERAGED_TICKER}_close'].iloc[0])
    
    return float(portfolio_value), bh_q_gross, bh_t_gross, pd.DataFrame(trade_hist), pd.DataFrame(tax_logs), sim_df, float(total_tax)

# --- UI Components ---

def main_app():
    st.set_page_config(layout="wide", page_title="TQQQ v22td After-Tax Backtester")
    st.title("⭐ TQQQ Volatility Strategy (v22td - After Tax)")
    
    last_day = get_last_closed_trading_day()
    
    with st.sidebar:
        st.header("1. Assets")
        TICKER = st.text_input("Underlying", "QQQ")
        LEVERAGED_TICKER = st.text_input("Leveraged (3x)", "TQQQ")
        INVERSE_TICKER = st.text_input("Inverse (3x)", "SQQQ")
        
        st.header("2. Backtest Period")
        start_date = st.date_input("Start Date", get_default_ytd_start_date(last_day))
        end_date = st.date_input("End Date", last_day)
        
        st.header("3. Strategy Logic")
        adx_thresh = st.slider("ADX Sideways Threshold", 0.0, 25.0, 12.0)
        m_vasl = st.number_input("Mini-VASL Multiplier", 0.0, 3.0, 1.5, 0.1)
        
        st.header("4. Tax Setup")
        st_tax = st.number_input("Annual Tax Rate (STCG)", 0.0, 0.5, 0.25)
        ltcg_rate = st.number_input("Liquidated Tax Rate (LTCG)", 0.0, 0.5, 0.15)
        
        if st.button("Run Analysis", type="primary"): st.rerun()

    # --- EXECUTION ---
    full_data = fetch_historical_data(end_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER)
    if full_data.empty: return

    live_p = fetch_live_price(TICKER)
    signal_price = live_p if live_p else full_data['close'].iloc[-1]
    
    inds = calculate_indicators(full_data, end_date, signal_price)
    
    # Live Signal Logic
    is_bull = (inds['current_price'] >= inds['sma_200'])
    if is_bull:
        if inds['adx'] < adx_thresh: signal, tick = "CASH (Sideways ADX)", "CASH"
        elif inds['current_price'] < inds['ema_5'] - (2.0 * inds['atr']): signal, tick = "SELL (Hard Stop)", "CASH"
        elif inds['current_price'] < inds['ema_5'] - (m_vasl * inds['atr']): signal, tick = "SELL (Soft Stop)", "CASH"
        else: signal, tick = f"BUY/HOLD {LEVERAGED_TICKER}", LEVERAGED_TICKER
    else:
        tick = LEVERAGED_TICKER if (inds['ema_5'] > inds['sma_20'] and inds['sma_20_slope'] > 0) else "CASH"
        signal = f"BEAR REGIME: {tick}"

    # --- DISPLAY ---
    c1, c2, c3 = st.columns([2,1,1])
    c1.metric("Current Signal", tick, signal)
    c2.metric(f"{TICKER} Price", f"${inds['current_price']:.2f}")
    c3.metric("ADX (Trend Strength)", f"{inds['adx']:.1f}")

    # Run Backtest
    sig_df = generate_historical_signals(full_data, LEVERAGED_TICKER, m_vasl, adx_thresh)
    final_v, bh_q, bh_t, history, taxes, sim_df, total_paid = run_tax_sim(sig_df, start_date, TICKER, LEVERAGED_TICKER, st_tax, ltcg_rate)

    # Metrics Summary
    st.subheader("Performance Summary (After-Tax)")
    
    # Net of LTCG for B&H
    bh_q_net = bh_q - max(0, (bh_q - 10000) * ltcg_rate)
    bh_t_net = bh_t - max(0, (bh_t - 10000) * ltcg_rate)

    res = pd.DataFrame({
        "Metric": ["Strategy (Annual Tax)", f"B&H {TICKER} (Net)", f"B&H {LEVERAGED_TICKER} (Net)"],
        "Final Value": [final_v, bh_q_net, bh_t_net],
        "Total Return": [(final_v/10000-1)*100, (bh_q_net/10000-1)*100, (bh_t_net/10000-1)*100],
        "Taxes Paid": [total_paid, (bh_q - bh_q_net), (bh_t - bh_t_net)]
    })
    st.table(res.style.format({"Final Value": "${:,.2f}", "Total Return": "{:.2f}%", "Taxes Paid": "${:,.2f}"}))

    # Chart - LIGHTENED VERSION
    st.subheader("Strategy Equity Curve")
    chart_df = sim_df.reset_index()
    chart = alt.Chart(chart_df).mark_line(
        strokeWidth=1.5,       # Thinner line
        opacity=0.7,           # Slight transparency to "lighten" the visual
        color='#1f77b4'        # Classic blue
    ).encode(
        x='Date:T',
        y=alt.Y('Portfolio_Value:Q', title="Portfolio Value ($)"),
        tooltip=['Date', 'Portfolio_Value']
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    # Details
    col_a, col_b = st.columns(2)
    with col_a:
        with st.expander("Annual Tax Breakdown"):
            st.dataframe(taxes.style.format({"Realized Gain": "${:,.2f}", "Tax Paid": "${:,.2f}"}), use_container_width=True)
    with col_b:
        with st.expander("Trade History"):
            st.dataframe(history, use_container_width=True)

if __name__ == "__main__":
    main_app()
