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

# --- 1. CONFIGURATION (STRICTLY ORIGINAL) ---
getcontext().prec = 50
warnings.filterwarnings("ignore")

EMA_PERIOD = 5
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
SMA_PERIOD = 200
SMA_SHORT_PERIOD = 20

# Macro Tickers for v28 Logic
TNX_TICKER = "^TNX" 
IRX_TICKER = "^IRX" 

INITIAL_INVESTMENT = Decimal("10000.00")
TQQQ_INCEPTION_DATE = date(2010, 2, 9)
SLIPPAGE_PCT = 0.002  

# --- 2. DATA ACQUISITION (UPDATED FOR MACRO DATA) ---

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
    # Added Macro Tickers to the download list
    tickers = [TICKER, LEVERAGED_TICKER, INVERSE_TICKER, TNX_TICKER, IRX_TICKER]
    try:
        all_data = yf.download(tickers, start=TQQQ_INCEPTION_DATE, end=end_date + timedelta(days=1), progress=False)
        if all_data.empty: return pd.DataFrame()
        
        df_combined = pd.DataFrame(index=all_data.index)
        for ticker in tickers:
            col = ('Adj Close', ticker) if ('Adj Close', ticker) in all_data.columns else ('Close', ticker)
            df_combined[f'{ticker}_close'] = all_data[col]
            if ticker == TICKER:
                df_combined['high'] = all_data['High'][ticker]
                df_combined['low'] = all_data['Low'][ticker]
        
        df_combined['close'] = df_combined[f'{TICKER}_close']
        # v28 Macro Calculations
        df_combined['Yield_Spread'] = df_combined[f'{TNX_TICKER}_close'] - df_combined[f'{IRX_TICKER}_close']
        df_combined['Spread_Slope'] = df_combined['Yield_Spread'].diff(10)
        
        return df_combined.dropna(subset=['close', 'high', 'low', 'Yield_Spread'])
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

# --- 3. THE STRATEGY (STRICT v28td_m LOGIC SWAP) ---

@st.cache_data(ttl=3600)
def generate_historical_signals(df, LEVERAGED_TICKER, mini_vasl_mult, 
                                adx_threshold, di_spread_pct, steepening_threshold, macro_buffer):
    df = df.copy()
    # Indicators
    df.ta.sma(length=SMA_PERIOD, append=True)
    df.ta.sma(length=SMA_SHORT_PERIOD, append=True)
    df.ta.ema(length=EMA_PERIOD, append=True)
    adx_df = pta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'], df['DMP'], df['DMN'] = adx_df['ADX_14'], adx_df['DMP_14'], adx_df['DMN_14']
    df['SMA_20_slope'] = df[f'SMA_{SMA_SHORT_PERIOD}'].diff(1)
    
    tr = pd.DataFrame({'hl': df['high']-df['low'], 'hpc': np.abs(df['high']-df['close'].shift(1)), 'lpc': np.abs(df['low']-df['close'].shift(1))}).max(axis=1)
    df['ATR'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
    df['VASL'] = df[f'EMA_{EMA_PERIOD}'] - (ATR_MULTIPLIER * df['ATR'])
    df['Mini_VASL'] = df[f'EMA_{EMA_PERIOD}'] - (mini_vasl_mult * df['ATR'])
    
    # Signal Loop with Hysteresis
    df['Trade_Ticker'] = 'CASH'
    macro_lockout = False
    recovery_level = steepening_threshold - macro_buffer
    
    for i in range(len(df)):
        row = df.iloc[i]
        if pd.isna(row[f'SMA_{SMA_PERIOD}']): continue
        if macro_lockout and row['Spread_Slope'] <= recovery_level: macro_lockout = False
        
        if row['close'] >= row[f'SMA_{SMA_PERIOD}']: # BULL REGIME
            if row['Yield_Spread'] > 0 and row['Spread_Slope'] > steepening_threshold:
                macro_lockout = True
                ticker = 'CASH'
            elif macro_lockout or row['ADX'] < adx_threshold or row['DMP'] <= (row['DMN']*(1+di_spread_pct)) or row['close'] < row['VASL']:
                ticker = 'CASH'
            else: ticker = LEVERAGED_TICKER
        else: # BEAR REGIME
            ticker = LEVERAGED_TICKER if (row[f'EMA_{EMA_PERIOD}'] > row[f'SMA_{SMA_SHORT_PERIOD}'] and row['SMA_20_slope'] > 0) else 'CASH'
        df.iloc[i, df.columns.get_loc('Trade_Ticker')] = ticker
        
    return df.dropna(subset=[f'SMA_{SMA_PERIOD}'])

# --- 4. BACKTEST ENGINE (STRICTLY ORIGINAL v23td) ---

@st.cache_data(ttl=3600)
def run_tax_sim(signals_df, start_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER, tax_rate, slippage_pct):
    # Ensure start_date is a valid trading day in the index
    actual_start = signals_df.index[signals_df.index.searchsorted(pd.Timestamp(start_date))]
    sim_df = signals_df.loc[actual_start:].copy()
    
    portfolio_value, shares, current_ticker = INITIAL_INVESTMENT, Decimal("0"), 'CASH'
    cost_basis, loss_carry, yearly_gain = Decimal("0"), Decimal("0"), Decimal("0")
    history, tax_logs = [], []
    
    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        prices = {TICKER: Decimal(str(row[f'{TICKER}_close'])), LEVERAGED_TICKER: Decimal(str(row[f'{LEVERAGED_TICKER}_close']))}
        target = row['Trade_Ticker']
        
        if target != current_ticker:
            if current_ticker != 'CASH':
                val = shares * prices[current_ticker]
                realized = val - cost_basis
                yearly_gain += realized
                portfolio_value = val
                history.append({'Date': row.name.date(), 'Action': f'SELL {current_ticker}', 'Asset': current_ticker, 'Price': float(prices[current_ticker]), 'Portfolio Value': float(portfolio_value), 'Realized P/L': float(realized)})
            
            if target != 'CASH':
                buy_price = prices[target]
                portfolio_value *= (Decimal("1") - Decimal(str(slippage_pct)))
                shares = portfolio_value / buy_price
                cost_basis = portfolio_value
                history.append({'Date': row.name.date(), 'Action': f'BUY {target}', 'Asset': target, 'Price': float(buy_price), 'Portfolio Value': float(portfolio_value), 'Realized P/L': 0.0})
            current_ticker = target
        
        if current_ticker != 'CASH': portfolio_value = shares * prices[current_ticker]
        
        # End of Year Tax Logic
        if i == len(sim_df)-1 or sim_df.index[i+1].year > row.name.year:
            taxable = max(Decimal("0"), yearly_gain - loss_carry)
            tax_due = taxable * Decimal(str(tax_rate))
            portfolio_value -= tax_due
            if current_ticker != 'CASH': shares = portfolio_value / prices[current_ticker]
            tax_logs.append({'Year': row.name.year, 'Realized Gain': float(yearly_gain), 'Tax Paid': float(tax_due)})
            loss_carry = max(Decimal("0"), loss_carry - yearly_gain) if yearly_gain > 0 else loss_carry + abs(yearly_gain)
            yearly_gain = Decimal("0")
            
        sim_df.at[row.name, 'Portfolio_Value'] = float(portfolio_value)
    
    return sim_df, pd.DataFrame(history), pd.DataFrame(tax_logs)

# --- 5. CHARTING (STRICTLY ORIGINAL v23td STYLE) ---

def plot_original_layered_chart(sig_df, history, TICKER, start_date):
    plot_data = sig_df.loc[start_date:].reset_index().rename(columns={'index':'Date'})
    price_cols = ['close', f'SMA_{SMA_PERIOD}', f'SMA_{SMA_SHORT_PERIOD}', f'EMA_{EMA_PERIOD}', 'VASL', 'Mini_VASL']
    
    long_df = plot_data.melt('Date', value_vars=price_cols, var_name='Metric', value_name='Price')
    selection = alt.selection_point(fields=['Metric'], bind='legend')
    
    base = alt.Chart(long_df).encode(x='Date:T').properties(height=600)
    lines = base.mark_line().encode(
        y=alt.Y('Price:Q', scale=alt.Scale(zero=False)),
        color=alt.Color('Metric:N', scale=alt.Scale(domain=price_cols, range=['gray', 'orange', 'blue', 'purple', 'red', 'gold'])),
        opacity=alt.condition(selection, alt.value(1.0), alt.value(0.1))
    ).add_params(selection)
    
    # Trade P/L Segments
    segments = []
    buy = None
    for _, r in history.iterrows():
        d = pd.Timestamp(r['Date'])
        if d < pd.Timestamp(start_date): continue
        if 'BUY' in r['Action']: buy = {'d': d, 'p': sig_df.loc[d, 'close'], 'v': r['Portfolio Value']}
        elif 'SELL' in r['Action'] and buy:
            win = r['Portfolio Value'] >= buy['v']
            segments.extend([{'t':_, 'Date':buy['d'], 'Price':buy['p'], 'Win':win}, {'t':_, 'Date':d, 'Price':sig_df.loc[d, 'close'], 'Win':win}])
            buy = None
            
    trade_layer = alt.Chart(pd.DataFrame(segments)).mark_line(size=4).encode(
        x='Date:T', y='Price:Q', detail='t:N', color=alt.condition(alt.datum.Win, alt.value('green'), alt.value('red'))
    )
    return (lines + trade_layer).interactive()

# --- 6. MAIN UI (STRICTLY ORIGINAL v23td LAYOUT) ---

def main():
    st.set_page_config(layout="wide", page_title="TQQQ v28td_m OriginalUI")
    st.title("🚀 TQQQ/SQQQ Strategy v28td_m")
    
    last_day = get_last_closed_trading_day()
    
    with st.sidebar:
        st.header("Settings")
        TICKER = st.text_input("Underlying", "QQQ")
        LEV_TICKER = st.text_input("Leveraged", "TQQQ")
        INV_TICKER = st.text_input("Inverse", "SQQQ")
        start_date = st.date_input("Start Date", date(2025, 1, 1))
        
        st.subheader("v28 Strategy Updates")
        adx_t = st.number_input("ADX Threshold", 0.0, 30.0, 12.0)
        di_s = st.number_input("DI Spread %", -1.0, 1.0, -0.8)
        st_t = st.number_input("Macro Steepening", 0.0, 1.0, 0.16)
        m_b = st.number_input("Macro Buffer", 0.0, 0.1, 0.03)
        
        st.subheader("Costs & Taxes")
        tax_r = st.number_input("Tax Rate (%)", 0.0, 50.0, 25.0) / 100
        if st.button("🚀 Run Simulation", type="primary"):
            st.cache_data.clear(); st.rerun()

    data = fetch_historical_data(last_day, TICKER, LEV_TICKER, INV_TICKER)
    if data.empty: return
    
    sig_df = generate_historical_signals(data, LEV_TICKER, 1.5, adx_t, di_s, st_t, m_b)
    results, history, tax_logs = run_tax_sim(sig_df, start_date, TICKER, LEV_TICKER, INV_TICKER, tax_r, SLIPPAGE_PCT)

    # === ORIGINAL UI HEADER ===
    st.subheader("🔴 Live Trading Signal")
    c1, c2, c3, c4 = st.columns(4)
    last_asset = history.iloc[-1]['Asset'] if not history.empty else "CASH"
    c1.markdown(f"## {'🟢' if last_asset != 'CASH' else '🟡'} **{last_asset}**")
    c2.metric("Yield Spread", f"{sig_df['Yield_Spread'].iloc[-1]:.4f}")
    c3.metric("Spread Slope", f"{sig_df['Spread_Slope'].iloc[-1]:.4f}")
    c4.metric("Strategy Return", f"{((results['Portfolio_Value'].iloc[-1]/10000)-1)*100:+.1f}%")

    # === ORIGINAL BENCHMARKS ===
    st.markdown("---")
    b1, b2, b3 = st.columns(3)
    b1.metric("Strategy (Net)", f"${results['Portfolio_Value'].iloc[-1]:,.2f}")
    q_ret = sig_df[f'{TICKER}_close'].iloc[-1] / sig_df[f'{TICKER}_close'].asof(pd.Timestamp(start_date))
    b2.metric(f"B&H {TICKER}", f"${float(INITIAL_INVESTMENT) * float(q_ret):,.2f}")
    t_ret = sig_df[f'{LEV_TICKER}_close'].iloc[-1] / sig_df[f'{LEV_TICKER}_close'].asof(pd.Timestamp(start_date))
    b3.metric(f"B&H {LEV_TICKER}", f"${float(INITIAL_INVESTMENT) * float(t_ret):,.2f}")

    # === ORIGINAL CHART & LOGS ===
    st.altair_chart(plot_original_layered_chart(sig_df, history, TICKER, start_date), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🧾 Trade History"):
            st.dataframe(history, use_container_width=True)
    with col2:
        with st.expander("💰 Tax Summary"):
            st.dataframe(tax_logs, use_container_width=True)

if __name__ == "__main__":
    main()
