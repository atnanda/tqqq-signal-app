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
    
    # Calculate ATR & VASL for plotting
    high_minus_low = df['high'] - df['low']
    high_minus_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_minus_prev_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.DataFrame({'hl': high_minus_low, 'hpc': high_minus_prev_close, 'lpc': low_minus_prev_close}).max(axis=1)
    df['ATR'] = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
    
    df['Trade_Ticker'] = 'CASH'
    df['Trade_Reason'] = 'N/A'
    
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
                ticker, reason = 'CASH', "Macro Steepening"
            elif macro_lockout:
                ticker, reason = 'CASH', f"Macro Lockout (<{recovery_level:.2f})"
            elif row['ADX'] < adx_threshold: ticker, reason = 'CASH', "ADX Filter"
            elif row['DMP'] <= (row['DMN'] * (1 + di_spread_pct)): ticker, reason = 'CASH', "DMI Filter"
            elif row['close'] < (row[f'EMA_{EMA_PERIOD}'] - (ATR_MULTIPLIER * row['ATR'])): ticker, reason = 'CASH', "VASL Stop"
            elif row['close'] < (row[f'EMA_{EMA_PERIOD}'] - (mini_vasl_mult * row['ATR'])): ticker, reason = 'CASH', "Mini-VASL Stop"
            else: ticker, reason = LEVERAGED_TICKER, "Bull Trend"
        else: # BEAR
            if row[f'EMA_{EMA_PERIOD}'] > row[f'SMA_{SMA_SHORT_PERIOD}'] and row['SMA_20_slope'] > 0:
                ticker, reason = LEVERAGED_TICKER, "Bear Recovery"
            else:
                ticker, reason = 'CASH', "Bear Regime"
        
        df.iloc[i, df.columns.get_loc('Trade_Ticker')] = ticker
        df.iloc[i, df.columns.get_loc('Trade_Reason')] = reason
        
    return df.dropna(subset=[f'SMA_{SMA_PERIOD}'])

# --- Restored Original Charting Logic ---

def plot_pro_signals(sig_df, history_df, TICKER, start_date):
    plot_data = sig_df[sig_df.index.date >= start_date].copy()
    
    # Metrics to plot
    price_cols = ['close', f'SMA_{SMA_PERIOD}', f'SMA_{SMA_SHORT_PERIOD}', f'EMA_{EMA_PERIOD}']
    plot_long = plot_data.reset_index().rename(columns={'index': 'Date'})[['Date'] + price_cols].melt(
        'Date', var_name='Metric', value_name='Price')
    
    # Interactive Selection
    selection = alt.selection_point(fields=['Metric'], bind='legend', name='MetricSelect')
    
    base = alt.Chart(plot_long).encode(x=alt.X('Date:T', title='Date')).properties(height=550)
    
    # Gray underlying price line
    price_line = base.mark_line(color='gray', opacity=0.3, size=1).encode(
        y=alt.Y('Price:Q', title=f'{TICKER} Price ($)'),
        tooltip=[alt.Tooltip('Price:Q', format='$.2f')]
    ).transform_filter(alt.datum.Metric == 'close')
    
    # Colored indicator lines
    indicators = base.mark_line().encode(
        y='Price:Q',
        color=alt.Color('Metric:N', scale=alt.Scale(
            domain=price_cols[1:],
            range=['orange', 'blue', 'purple']
        )),
        opacity=alt.condition(selection, alt.value(1.0), alt.value(0.1))
    ).add_params(selection).transform_filter(alt.datum.Metric != 'close')
    
    # Trade Segments (Green/Red lines)
    trades = history_df[history_df['Action'].str.startswith(('BUY', 'SELL'))].copy()
    df_segments = []
    buy_trade = None
    
    for _, row in trades.iterrows():
        trade_date = pd.to_datetime(row['Date'])
        if trade_date not in sig_df.index: continue
        
        if row['Action'].startswith('BUY'):
            buy_trade = {'date': trade_date, 'price': sig_df.loc[trade_date, 'close'], 'val': row['Portfolio Value']}
        elif row['Action'].startswith('SELL') and buy_trade:
            is_prof = row['Portfolio Value'] >= buy_trade['val']
            df_segments.append({'tid': len(df_segments), 'Date': buy_trade['date'], 'Price': buy_trade['price'], 'Cat': is_prof})
            df_segments.append({'tid': len(df_segments), 'Date': trade_date, 'Price': sig_df.loc[trade_date, 'close'], 'Cat': is_prof})
            buy_trade = None
            
    segments = alt.Chart(pd.DataFrame(df_segments)).mark_line(size=4).encode(
        x='Date:T', y='Price:Q', detail='tid:N',
        color=alt.condition(alt.datum.Cat == True, alt.value('#00b300'), alt.value('#ff3333'))
    )
    
    return (price_line + indicators + segments).interactive()

# --- Main Logic & UI ---

def main():
    st.set_page_config(layout="wide", page_title="TQQQ v28td_m Pro")
    st.title("⭐ Leveraged Strategy v28td_m (Pro Visuals + Macro)")
    
    last_day = get_last_closed_trading_day()
    
    with st.sidebar:
        st.header("⚙️ Parameters")
        TICKER = st.text_input("Underlying", "QQQ")
        LEVERAGED_TICKER = st.text_input("Leveraged", "TQQQ")
        INVERSE_TICKER = st.text_input("Inverse", "SQQQ")
        start_date = st.date_input("Start Date", date(last_day.year, 1, 1))
        
        adx_thresh = st.number_input("ADX Threshold", 0.0, 30.0, DEFAULT_ADX_THRESHOLD)
        di_spread = st.number_input("DI Spread %", -1.0, 1.0, DEFAULT_DI_SPREAD_PCT)
        steep_thresh = st.number_input("Steepening Threshold", 0.0, 1.0, DEFAULT_STEEPENING_THRESHOLD)
        m_buffer = st.number_input("Macro Buffer", 0.0, 0.1, DEFAULT_MACRO_BUFFER)
        
        slippage = st.number_input("Slippage (%)", 0.0, 1.0, 0.2) / 100
        st_tax = st.number_input("Tax Rate (%)", 0.0, 50.0, 25.0) / 100
        
        if st.button("🚀 Run Analysis", type="primary"):
            st.cache_data.clear()
            st.rerun()

    data = fetch_historical_data(last_day, TICKER, LEVERAGED_TICKER, INVERSE_TICKER)
    if data.empty: return
    
    sig_df = generate_historical_signals(data, LEVERAGED_TICKER, 1.5, adx_thresh, di_spread, steep_thresh, m_buffer)
    
    # Simplified Backtest Engine Integration
    from decimal import Decimal
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
                history.append({'Date': idx.date(), 'Action': f"SELL {current_ticker}", 'Asset': current_ticker, 'Portfolio Value': float(portfolio_value)})
            if target != 'CASH':
                shares = (portfolio_value * (1 - Decimal(str(slippage)))) / prices[target]
                history.append({'Date': idx.date(), 'Action': f"BUY {target}", 'Asset': target, 'Portfolio Value': float(portfolio_value)})
            current_ticker = target
        
        if current_ticker != 'CASH': portfolio_value = shares * prices[current_ticker]
        sim_df.at[idx, 'Portfolio_Value'] = float(portfolio_value)
        sim_df.at[idx, 'BH_QQQ'] = float((prices[TICKER] / q_start) * INITIAL_INVESTMENT)
        sim_df.at[idx, 'BH_TQQQ'] = float((prices[LEVERAGED_TICKER] / t_start) * INITIAL_INVESTMENT)

    hist_df = pd.DataFrame(history)

    # --- UI DISPLAY ---
    st.subheader("🔴 Live Trading Signal")
    c1, c2, c3, c4 = st.columns(4)
    curr_sig = hist_df.iloc[-1]['Asset'] if not hist_df.empty else "CASH"
    c1.markdown(f"## {'🟢' if curr_sig != 'CASH' else '🟡'} **{curr_sig}**")
    c2.metric("Yield Spread", f"{sig_df['Yield_Spread'].iloc[-1]:.4f}")
    c3.metric("Spread Slope", f"{sig_df['Spread_Slope'].iloc[-1]:.4f}")
    c4.metric("ADX", f"{sig_df['ADX'].iloc[-1]:.1f}")

    st.markdown("---")
    st.markdown("## 📊 Benchmarks vs Strategy")
    b1, b2, b3 = st.columns(3)
    final_v = sim_df['Portfolio_Value'].iloc[-1]
    b1.metric("Strategy (Net)", f"${final_v:,.2f}", f"{(final_v/10000-1)*100:+.1f}%")
    b2.metric(f"B&H {TICKER}", f"${sim_df['BH_QQQ'].iloc[-1]:,.2f}")
    b3.metric(f"B&H {LEVERAGED_TICKER}", f"${sim_df['BH_TQQQ'].iloc[-1]:,.2f}")

    st.markdown("## 📉 Pro Interactive Trade Chart")
    st.altair_chart(plot_pro_signals(sig_df, hist_df, TICKER, start_date), use_container_width=True)

    with st.expander("🧾 Detailed Trade History"):
        st.dataframe(hist_df, use_container_width=True)

if __name__ == "__main__":
    main()
