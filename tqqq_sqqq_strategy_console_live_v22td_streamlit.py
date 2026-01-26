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
DEFAULT_ADX_THRESHOLD = 12.0 # Hidden

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

def calculate_max_drawdown(series):
    if series.empty: return 0.0
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    return float(drawdown.min() * 100)

@st.cache_data(ttl=24*3600)
def fetch_historical_data(end_date, TICKER, LEVERAGED_TICKER):
    start_date = TQQQ_INCEPTION_DATE
    try:
        all_data = yf.download([TICKER, LEVERAGED_TICKER], start=start_date, end=end_date + timedelta(days=1), progress=False)
        df_combined = pd.DataFrame(index=all_data.index)
        for ticker in [TICKER, LEVERAGED_TICKER]:
            df_combined[f'{ticker}_close'] = all_data['Adj Close'][ticker]
            if ticker == TICKER:
                df_combined['high'] = all_data['High'][ticker]
                df_combined['low'] = all_data['Low'][ticker]
        df_combined['close'] = df_combined[f'{TICKER}_close']
        return df_combined.dropna()
    except Exception as e:
        st.error(f"Data Error: {e}")
        return pd.DataFrame()

# --- Logic & Backtest ---

@st.cache_data(ttl=24*3600)
def run_v22td_strategy(df, start_date, m_vasl, tax_rate):
    df = df.copy()
    # Indicators
    df.ta.sma(length=SMA_PERIOD, append=True)
    df.ta.sma(length=SMA_SHORT_PERIOD, append=True)
    df.ta.ema(length=EMA_PERIOD, append=True)
    adx_df = pta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'] = adx_df['ADX_14']
    df['SMA_20_slope'] = df[f'SMA_{SMA_SHORT_PERIOD}'].diff(1)
    df['ATR'] = pta.atr(df['high'], df['low'], df['close'], length=ATR_PERIOD)

    sim_df = df[df.index.date >= start_date].copy()
    port_val, shares, ticker_state = INITIAL_INVESTMENT, Decimal("0"), 'CASH'
    cost_basis, loss_carry, yearly_gain, total_tax = Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0")
    history = []

    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        prices = {'QQQ': Decimal(str(row['QQQ_close'])), 'TQQQ': Decimal(str(row['TQQQ_close']))}
        
        # Signal
        vasl, mini = row[f'EMA_{EMA_PERIOD}'] - (2.0 * row['ATR']), row[f'EMA_{EMA_PERIOD}'] - (m_vasl * row['ATR'])
        if row['close'] >= row[f'SMA_{SMA_PERIOD}']:
            target = 'TQQQ' if (row['ADX'] >= DEFAULT_ADX_THRESHOLD and row['close'] >= vasl and row['close'] >= mini) else 'CASH'
        else:
            target = 'TQQQ' if (row[f'EMA_{EMA_PERIOD}'] > row[f'SMA_{SMA_SHORT_PERIOD}'] and row['SMA_20_slope'] > 0) else 'CASH'

        if target != ticker_state:
            if ticker_state != 'CASH':
                val = shares * prices[ticker_state]
                yearly_gain += (val - cost_basis)
                port_val = val
                history.append({'Date': row.name.date(), 'Action': f"SELL {ticker_state}", 'Value': float(port_val)})
            if target != 'CASH':
                shares = port_val / prices[target]
                cost_basis = port_val
                history.append({'Date': row.name.date(), 'Action': f"BUY {target}", 'Value': float(port_val)})
            ticker_state = target

        if ticker_state != 'CASH': port_val = shares * prices[ticker_state]

        # Year End Tax
        if (i == len(sim_df)-1) or (sim_df.index[i+1].year > row.name.year):
            tax_due = max(Decimal("0"), (yearly_gain - loss_carry) * Decimal(str(tax_rate)))
            port_val -= tax_due
            total_tax += tax_due
            if ticker_state != 'CASH': shares = port_val / prices[ticker_state]
            loss_carry = max(Decimal("0"), loss_carry - yearly_gain if yearly_gain > 0 else loss_carry + abs(yearly_gain))
            yearly_gain = Decimal("0")
            
        sim_df.at[row.name, 'Strategy'] = float(port_val)

    return sim_df, pd.DataFrame(history), float(total_tax)

# --- UI ---

def main():
    st.set_page_config(layout="wide", page_title="TQQQ v22td Pro")
    st.title("⭐ TQQQ Volatility Strategy (v22td)")
    
    last_day = get_last_closed_trading_day()
    with st.sidebar:
        st.header("Parameters")
        TICKER = st.text_input("Underlying", "QQQ")
        LEV_TICKER = st.text_input("Leveraged", "TQQQ")
        start_date = st.date_input("Start Date", date(date.today().year, 1, 1))
        m_vasl = st.number_input("Mini-VASL Multiplier", 0.0, 3.0, 1.5)
        tax_rate = st.number_input("Tax Rate", 0.0, 0.5, 0.25)
        st.markdown("---")
        use_log = st.checkbox("Logarithmic Scale", value=True)
        if st.button("Run Analysis", type="primary"): st.rerun()

    data = fetch_historical_data(last_day, TICKER, LEV_TICKER)
    if data.empty: return

    results, history, total_tax = run_v22td_strategy(data, start_date, m_vasl, tax_rate)

    # 1. Signal Header
    st.subheader("Live Trading Signal")
    curr_sig = history.iloc[-1]['Action'] if not history.empty else "CASH"
    status = "BUY/HOLD" if LEV_TICKER in curr_sig else "CASH"
    c1, c2 = st.columns([3, 1])
    c1.markdown(f"## :rotating_light: **{status}**")
    c2.markdown(f"## **{LEV_TICKER if status == 'BUY/HOLD' else 'CASH'}**")
    
    m1, m2 = st.columns(2)
    m1.metric(f"{TICKER} Price", f"${data['close'].iloc[-1]:.2f}")
    m2.metric("Market Regime", "BULL" if data['close'].iloc[-1] >= data.ta.sma(length=200).iloc[-1] else "BEAR")
    st.markdown("---")

    # 2. Performance Summary Table
    st.subheader("📊 Performance Summary")
    bench_q = (results[f'{TICKER}_close'] / results[f'{TICKER}_close'].iloc[0]) * 10000
    bench_t = (results[f'{LEV_TICKER}_close'] / results[f'{LEV_TICKER}_close'].iloc[0]) * 10000
    
    sum_df = pd.DataFrame({
        'Metric': ['Strategy', f'B&H {TICKER}', f'B&H {LEV_TICKER}'],
        'Final Value': [results['Strategy'].iloc[-1], bench_q.iloc[-1], bench_t.iloc[-1]],
        'Return': [(results['Strategy'].iloc[-1]/10000-1)*100, (bench_q.iloc[-1]/10000-1)*100, (bench_t.iloc[-1]/10000-1)*100],
        'Max Drawdown': [calculate_max_drawdown(results['Strategy']), calculate_max_drawdown(bench_q), calculate_max_drawdown(bench_t)],
        'Taxes Paid': [total_tax, 0.0, 0.0]
    })
    st.dataframe(sum_df.style.format({'Final Value':'${:,.2f}','Return':'{:.2f}%','Max Drawdown':'{:.2f}%','Taxes Paid':'${:,.2f}'}), hide_index=True)

    # 3. RESTORED ENHANCED GRAPH
    st.subheader("📈 Multi-Asset Equity Curve (Normalized to $10k)")
    
    plot_df = pd.DataFrame({
        'Date': results.index,
        'Strategy': results['Strategy'],
        f'B&H {TICKER}': bench_q,
        f'B&H {LEV_TICKER}': bench_t
    }).melt('Date', var_name='Asset', value_name='Value')

    selection = alt.selection_point(fields=['Asset'], bind='legend')
    chart = alt.Chart(plot_df).mark_line().encode(
        x='Date:T',
        y=alt.Y('Value:Q', scale=alt.Scale(type='log' if use_log else 'linear'), title="Value ($)"),
        color=alt.Color('Asset:N', scale=alt.Scale(range=['#636EFA', '#EF553B', '#00CC96'])),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.1)),
        tooltip=['Date:T', 'Asset:N', alt.Tooltip('Value:Q', format='$,.2f')]
    ).add_params(selection).interactive().properties(height=500)
    
    st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()
