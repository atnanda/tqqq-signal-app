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

# --- Configuration ---
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

# --- Data Fetching ---
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
def fetch_historical_data(end_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER):
    tickers = [TICKER, LEVERAGED_TICKER, INVERSE_TICKER, TNX_TICKER, IRX_TICKER]
    try:
        all_data = yf.download(tickers, start=TQQQ_INCEPTION_DATE, end=end_date + timedelta(days=1), progress=False)
        if all_data.empty: return pd.DataFrame()
        df = pd.DataFrame(index=all_data.index)
        for t in tickers:
            col = ('Adj Close', t) if ('Adj Close', t) in all_data.columns else ('Close', t)
            df[f'{t}_close'] = all_data[col]
            if t == TICKER:
                df['high'], df['low'] = all_data['High'][t], all_data['Low'][t]
        df['close'] = df[f'{TICKER}_close']
        df['Yield_Spread'] = df[f'{TNX_TICKER}_close'] - df[f'{IRX_TICKER}_close']
        df['Spread_Slope'] = df['Yield_Spread'].diff(10)
        return df.dropna(subset=['close', 'high', 'low']).loc[:pd.Timestamp(end_date)]
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()

# --- Signal Engine (v28td_m Logic) ---
@st.cache_data(ttl=3600)
def generate_signals(df, LEVERAGED_TICKER, adx_threshold, di_spread_pct, steepening_threshold, macro_buffer):
    df = df.copy()
    df.ta.sma(length=SMA_PERIOD, append=True)
    df.ta.sma(length=SMA_SHORT_PERIOD, append=True)
    df.ta.ema(length=EMA_PERIOD, append=True)
    adx = pta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'], df['DMP'], df['DMN'] = adx['ADX_14'], adx['DMP_14'], adx['DMN_14']
    df['SMA_20_slope'] = df[f'SMA_{SMA_SHORT_PERIOD}'].diff(1)
    tr = pd.DataFrame({'hl': df['high']-df['low'], 'hpc': np.abs(df['high']-df['close'].shift(1)), 'lpc': np.abs(df['low']-df['close'].shift(1))}).max(axis=1)
    df['ATR'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    
    df['VASL'] = df[f'EMA_{EMA_PERIOD}'] - (2.0 * df['ATR'])
    df['Mini_VASL'] = df[f'EMA_{EMA_PERIOD}'] - (1.5 * df['ATR'])
    
    df['Trade_Ticker'] = 'CASH'
    macro_lockout = False
    for i in range(len(df)):
        row = df.iloc[i]
        if pd.isna(row[f'SMA_{SMA_PERIOD}']): continue
        if macro_lockout and row['Spread_Slope'] <= (steepening_threshold - macro_buffer): macro_lockout = False
        
        if row['close'] >= row[f'SMA_{SMA_PERIOD}']: # BULL
            if row['Yield_Spread'] > 0 and row['Spread_Slope'] > steepening_threshold:
                macro_lockout = True
                ticker = 'CASH'
            elif macro_lockout or row['ADX'] < adx_threshold or row['DMP'] <= (row['DMN']*(1+di_spread_pct)) or row['close'] < row['VASL']:
                ticker = 'CASH'
            else: ticker = LEVERAGED_TICKER
        else: # BEAR
            ticker = LEVERAGED_TICKER if (row[f'EMA_{EMA_PERIOD}'] > row[f'SMA_{SMA_SHORT_PERIOD}'] and row['SMA_20_slope'] > 0) else 'CASH'
        df.iloc[i, df.columns.get_loc('Trade_Ticker')] = ticker
    return df.dropna(subset=[f'SMA_{SMA_PERIOD}'])

# --- Charting ---
def plot_pro_chart(sig_df, history_df, TICKER, start_date):
    plot_data = sig_df.loc[start_date:].reset_index().rename(columns={'index': 'Date'})
    cols = ['close', f'SMA_{SMA_PERIOD}', f'SMA_{SMA_SHORT_PERIOD}', f'EMA_{EMA_PERIOD}', 'VASL', 'Mini_VASL']
    long_df = plot_data.melt('Date', value_vars=cols, var_name='Metric', value_name='Price')
    
    selection = alt.selection_point(fields=['Metric'], bind='legend')
    base = alt.Chart(long_df).encode(x='Date:T').properties(height=550)
    
    lines = base.mark_line().encode(
        y=alt.Y('Price:Q', scale=alt.Scale(zero=False)),
        color=alt.Color('Metric:N', scale=alt.Scale(domain=cols, range=['gray', 'orange', 'blue', 'purple', 'red', 'gold'])),
        opacity=alt.condition(selection, alt.value(1.0), alt.value(0.1))
    ).add_params(selection)
    
    segments = []
    buy = None
    for _, r in history_df.iterrows():
        d = pd.Timestamp(r['Date'])
        if d < pd.Timestamp(start_date): continue
        if 'BUY' in r['Action']: buy = {'d': d, 'p': sig_df.loc[d, 'close'], 'v': r['Portfolio Value']}
        elif 'SELL' in r['Action'] and buy:
            win = r['Portfolio Value'] >= buy['v']
            segments.extend([{'t': i, 'Date': buy['d'], 'Price': buy['p'], 'Win': win}, {'t': i, 'Date': d, 'Price': sig_df.loc[d, 'close'], 'Win': win}])
            buy = None
    
    trade_layer = alt.Chart(pd.DataFrame(segments)).mark_line(size=4).encode(
        x='Date:T', y='Price:Q', detail='t:N', color=alt.condition(alt.datum.Win, alt.value('green'), alt.value('red'))
    )
    return (lines + trade_layer).interactive()

# --- Main UI ---
def main():
    st.set_page_config(layout="wide", page_title="TQQQ v28td_m Fixed")
    last_day = get_last_closed_trading_day()
    
    with st.sidebar:
        st.header("Settings")
        TICKER = st.text_input("Underlying", "QQQ")
        LEV = st.text_input("Leveraged", "TQQQ")
        start_date = st.date_input("Start Date", date(last_day.year, 1, 1))
        adx_t = st.number_input("ADX", 0.0, 30.0, 12.0)
        di_s = st.number_input("DI Spread", -1.0, 1.0, -0.8)
        st_t = st.number_input("Macro", 0.0, 1.0, 0.16)
        m_b = st.number_input("Buffer", 0.0, 0.1, 0.03)
        if st.button("Run"): st.cache_data.clear(); st.rerun()

    df = fetch_historical_data(last_day, TICKER, LEV, "SQQQ")
    if df.empty: return
    
    sig_df = generate_signals(df, LEV, adx_t, di_s, st_t, m_b)
    sim_df = sig_df.loc[start_date:].copy()
    
    # Corrected Backtest Loop
    val, shares, curr = INITIAL_INVESTMENT, Decimal("0"), 'CASH'
    history = []
    for d, r in sim_df.iterrows():
        px = {TICKER: Decimal(str(r[f'{TICKER}_close'])), LEV: Decimal(str(r[f'{LEV}_close']))}
        target = r['Trade_Ticker']
        if target != curr:
            if curr != 'CASH':
                val = shares * px[curr]
                history.append({'Date': d.date(), 'Action': f"SELL {curr}", 'Asset': curr, 'Portfolio Value': float(val)})
            if target != 'CASH':
                shares = (val * Decimal("0.998")) / px[target]
                history.append({'Date': d.date(), 'Action': f"BUY {target}", 'Asset': target, 'Portfolio Value': float(val)})
            curr = target
        if curr != 'CASH': val = shares * px[curr]
        sim_df.at[d, 'Portfolio_Value'] = float(val)

    hist_df = pd.DataFrame(history)
    
    # UI
    st.subheader("🔴 Live Trading Signal")
    c1, c2, c3, c4 = st.columns(4)
    # FIX: Ensure we check Asset key safely
    last_asset = hist_df.iloc[-1]['Asset'] if not hist_df.empty else "CASH"
    c1.markdown(f"## {'🟢' if last_asset != 'CASH' else '🟡'} **{last_asset}**")
    c2.metric("Yield Spread", f"{sig_df['Yield_Spread'].iloc[-1]:.4f}")
    c3.metric("ADX", f"{sig_df['ADX'].iloc[-1]:.1f}")
    c4.metric("Return", f"{((float(val)/10000)-1)*100:+.1f}%")

    st.markdown("## 📊 Performance vs QQQ")
    b1, b2 = st.columns(2)
    b1.metric("Strategy Final", f"${float(val):,.2f}")
    q_ret = (df[f'{TICKER}_close'].iloc[-1] / df[f'{TICKER}_close'].loc[pd.Timestamp(start_date)])
    b2.metric("B&H QQQ", f"${float(INITIAL_INVESTMENT * Decimal(str(q_ret))):,.2f}")

    st.altair_chart(plot_pro_chart(sig_df, hist_df, TICKER, start_date), use_container_width=True)
    st.dataframe(hist_df, use_container_width=True)

if __name__ == "__main__":
    main()
