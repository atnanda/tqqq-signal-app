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

TNX_TICKER = "^TNX" 
IRX_TICKER = "^IRX" 

INITIAL_INVESTMENT = Decimal("10000.00")
TQQQ_INCEPTION_DATE = date(2010, 2, 9)
DEFAULT_ADX_THRESHOLD = 12.0
DEFAULT_STEEPENING_THRESHOLD = 0.16
DEFAULT_MACRO_BUFFER = 0.03
DEFAULT_DI_SPREAD_PCT = -0.8
SLIPPAGE_PCT = 0.002  

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

@st.cache_data(ttl=3600)
def fetch_historical_data(end_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER):
    tickers = [TICKER, LEVERAGED_TICKER, INVERSE_TICKER, TNX_TICKER, IRX_TICKER]
    try:
        all_data = yf.download(tickers, start=TQQQ_INCEPTION_DATE, end=end_date + timedelta(days=1), 
                              progress=False, auto_adjust=False)
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
        df_combined['Yield_Spread'] = df_combined[f'{TNX_TICKER}_close'] - df_combined[f'{IRX_TICKER}_close']
        df_combined['Spread_Slope'] = df_combined['Yield_Spread'].diff(10)
        
        df_combined.dropna(subset=['close', 'high', 'low', f'{TNX_TICKER}_close'], inplace=True)
        return df_combined[df_combined.index.date <= end_date]
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def generate_historical_signals(df, LEVERAGED_TICKER, mini_vasl_mult, 
                                adx_threshold, di_spread_pct, steepening_threshold, macro_buffer):
    df = df.copy()
    df.ta.sma(length=SMA_PERIOD, append=True)
    df.ta.sma(length=SMA_SHORT_PERIOD, append=True)
    df.ta.ema(length=EMA_PERIOD, append=True)
    adx_df = pta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'], df['DMP'], df['DMN'] = adx_df['ADX_14'], adx_df['DMP_14'], adx_df['DMN_14']
    df['SMA_20_slope'] = df[f'SMA_{SMA_SHORT_PERIOD}'].diff(1)
    
    # ATR & VASL Logic
    high_minus_low = df['high'] - df['low']
    high_minus_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_minus_prev_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.DataFrame({'hl': high_minus_low, 'hpc': high_minus_prev_close, 'lpc': low_minus_prev_close}).max(axis=1)
    df['ATR'] = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
    
    # Dynamic Exit Levels for Charting
    df['VASL'] = df[f'EMA_{EMA_PERIOD}'] - (ATR_MULTIPLIER * df['ATR'])
    df['Mini_VASL'] = df[f'EMA_{EMA_PERIOD}'] - (mini_vasl_mult * df['ATR'])
    
    df['Trade_Ticker'] = 'CASH'
    macro_lockout = False
    recovery_level = steepening_threshold - macro_buffer
    
    for i in range(len(df)):
        row = df.iloc[i]
        if pd.isna(row[f'SMA_{SMA_PERIOD}']): continue
        
        if macro_lockout and row['Spread_Slope'] <= recovery_level:
            macro_lockout = False
        
        if row['close'] >= row[f'SMA_{SMA_PERIOD}']: # BULL
            if (row['Yield_Spread'] > 0) and (row['Spread_Slope'] > steepening_threshold):
                macro_lockout = True
                ticker = 'CASH'
            elif macro_lockout:
                ticker = 'CASH'
            elif row['ADX'] < adx_threshold: ticker = 'CASH'
            elif row['DMP'] <= (row['DMN'] * (1 + di_spread_pct)): ticker = 'CASH'
            elif row['close'] < row['VASL']: ticker = 'CASH'
            elif row['close'] < row['Mini_VASL']: ticker = 'CASH'
            else: ticker = LEVERAGED_TICKER
        else: # BEAR
            if row[f'EMA_{EMA_PERIOD}'] > row[f'SMA_{SMA_SHORT_PERIOD}'] and row['SMA_20_slope'] > 0:
                ticker = LEVERAGED_TICKER
            else:
                ticker = 'CASH'
        
        df.iloc[i, df.columns.get_loc('Trade_Ticker')] = ticker
        
    return df.dropna(subset=[f'SMA_{SMA_PERIOD}'])

# --- RECONSTRUCTED ORIGINAL CHARTING ---

def plot_original_style_chart(sig_df, history_df, TICKER, start_date):
    plot_data = sig_df[sig_df.index.date >= start_date].copy()
    
    # Indicator Columns
    price_cols = ['close', f'SMA_{SMA_PERIOD}', f'SMA_{SMA_SHORT_PERIOD}', f'EMA_{EMA_PERIOD}', 'VASL', 'Mini_VASL']
    
    plot_long = plot_data.reset_index().rename(columns={'index': 'Date'})[['Date'] + price_cols].melt(
        'Date', var_name='Metric', value_name='Price')
    
    selection = alt.selection_point(fields=['Metric'], bind='legend', name='MetricSelect')
    
    base = alt.Chart(plot_long).encode(x=alt.X('Date:T', title='Date')).properties(height=600)
    
    # 1. Main Price Line
    price_line = base.mark_line(color='gray', opacity=0.3, size=1).encode(
        y=alt.Y('Price:Q', scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip('Price:Q', format='$.2f')]
    ).transform_filter(alt.datum.Metric == 'close')
    
    # 2. Layered Indicators
    indicators = base.mark_line().encode(
        y='Price:Q',
        color=alt.Color('Metric:N', scale=alt.Scale(
            domain=price_cols[1:],
            range=['#FFA500', '#1E90FF', '#9370DB', '#FF4500', '#FFD700'] # Orange, Blue, Purple, Red-Orange, Gold
        )),
        opacity=alt.condition(selection, alt.value(1.0), alt.value(0.1))
    ).add_params(selection).transform_filter(alt.datum.Metric != 'close')
    
    # 3. Trade Profit/Loss Segments
    trades = history_df[history_df['Action'].str.startswith(('BUY', 'SELL'))].copy()
    df_segments = []
    buy_trade = None
    
    for _, row in trades.iterrows():
        t_date = pd.to_datetime(row['Date'])
        if t_date not in sig_df.index: continue
        
        if row['Action'].startswith('BUY'):
            buy_trade = {'d': t_date, 'p': sig_df.loc[t_date, 'close'], 'v': row['Portfolio Value']}
        elif row['Action'].startswith('SELL') and buy_trade:
            is_win = row['Portfolio Value'] >= buy_trade['v']
            df_segments.append({'tid': len(df_segments), 'Date': buy_trade['d'], 'Price': buy_trade['p'], 'Win': is_win})
            df_segments.append({'tid': len(df_segments), 'Date': t_date, 'Price': sig_df.loc[t_date, 'close'], 'Win': is_win})
            buy_trade = None
            
    segments = alt.Chart(pd.DataFrame(df_segments)).mark_line(size=4).encode(
        x='Date:T', y='Price:Q', detail='tid:N',
        color=alt.condition(alt.datum.Win == True, alt.value('#00CC00'), alt.value('#FF0000'))
    )
    
    return (price_line + indicators + segments).interactive()

# --- Main App ---

def main():
    st.set_page_config(layout="wide", page_title="TQQQ v28td_m Full Restore")
    st.title("⭐ TQQQ/SQQQ Strategy v28td_m (Original UI Reconstructed)")
    
    last_day = get_last_closed_trading_day()
    
    with st.sidebar:
        st.header("Strategy Settings")
        TICKER = st.text_input("Underlying", "QQQ")
        LEVERAGED_TICKER = st.text_input("Leveraged", "TQQQ")
        start_date = st.date_input("Start Date", date(last_day.year, 1, 1))
        
        adx_thresh = st.number_input("ADX Threshold", 0.0, 30.0, DEFAULT_ADX_THRESHOLD)
        di_spread = st.number_input("DI Spread %", -1.0, 1.0, DEFAULT_DI_SPREAD_PCT)
        steep_thresh = st.number_input("Macro Threshold", 0.0, 1.0, DEFAULT_STEEPENING_THRESHOLD)
        m_buffer = st.number_input("Macro Buffer", 0.0, 0.1, DEFAULT_MACRO_BUFFER)
        
        slippage = st.number_input("Slippage (%)", 0.0, 1.0, 0.2) / 100
        
        if st.button("Run Simulation", type="primary"):
            st.cache_data.clear()
            st.rerun()

    data = fetch_historical_data(last_day, TICKER, LEVERAGED_TICKER, "SQQQ")
    if data.empty: return
    
    sig_df = generate_historical_signals(data, LEVERAGED_TICKER, 1.5, adx_thresh, di_spread, steep_thresh, m_buffer)
    
    # Backtest Execution
    sim_df = sig_df[sig_df.index.date >= start_date].copy()
    portfolio_value, shares, current_ticker = INITIAL_INVESTMENT, Decimal("0"), 'CASH'
    history = []
    
    q_start = Decimal(str(sim_df[f'{TICKER}_close'].iloc[0]))
    t_start = Decimal(str(sim_df[f'{LEVERAGED_TICKER}_close'].iloc[0]))
    
    for idx, row in sim_df.iterrows():
        prices = {t: Decimal(str(row[f'{t}_close'])) for t in [TICKER, LEVERAGED_TICKER]}
        target = row['Trade_Ticker']
        
        if target != current_ticker:
            if current_ticker != 'CASH':
                portfolio_value = shares * prices[current_ticker]
                history.append({'Date': idx.date(), 'Action': f"SELL {current_ticker}", 'Portfolio Value': float(portfolio_value)})
            if target != 'CASH':
                shares = (portfolio_value * (1 - Decimal(str(slippage)))) / prices[target]
                history.append({'Date': idx.date(), 'Action': f"BUY {target}", 'Portfolio Value': float(portfolio_value)})
            current_ticker = target
        
        if current_ticker != 'CASH': portfolio_value = shares * prices[current_ticker]
        sim_df.at[idx, 'Portfolio_Value'] = float(portfolio_value)
        sim_df.at[idx, 'BH_QQQ'] = float((prices[TICKER] / q_start) * INITIAL_INVESTMENT)
        sim_df.at[idx, 'BH_TQQQ'] = float((prices[LEVERAGED_TICKER] / t_start) * INITIAL_INVESTMENT)

    hist_df = pd.DataFrame(history)

    # UI Header
    st.subheader("🔴 Live Trading Signal")
    c1, c2, c3, c4 = st.columns(4)
    curr_sig = hist_df.iloc[-1]['Asset'] if not hist_df.empty else "CASH"
    c1.markdown(f"## {'🟢' if curr_sig != 'CASH' else '🟡'} **{curr_sig}**")
    c2.metric("Yield Spread", f"{sig_df['Yield_Spread'].iloc[-1]:.4f}")
    c3.metric("ADX (Trend)", f"{sig_df['ADX'].iloc[-1]:.1f}")
    c4.metric("Strategy P/L", f"{((float(portfolio_value)/10000)-1)*100:+.1f}%")

    # Benchmarks
    st.markdown("---")
    b1, b2, b3 = st.columns(3)
    b1.metric("Strategy Final", f"${float(portfolio_value):,.2f}")
    b2.metric(f"B&H {TICKER}", f"${sim_df['BH_QQQ'].iloc[-1]:,.2f}")
    b3.metric(f"B&H {LEVERAGED_TICKER}", f"${sim_df['BH_TQQQ'].iloc[-1]:,.2f}")

    st.markdown("## 📉 Original Style Layered Chart")
    st.altair_chart(plot_original_style_chart(sig_df, hist_df, TICKER, start_date), use_container_width=True)

    with st.expander("View Full Trade Log"):
        st.dataframe(hist_df, use_container_width=True)

if __name__ == "__main__":
    main()
