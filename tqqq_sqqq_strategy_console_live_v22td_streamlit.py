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
DEFAULT_ADX_THRESHOLD = 12.0 # Hidden from UI

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
    # return date(today_date.year, 1, 1)
    return date(2025, 1, 1)


def calculate_max_drawdown(series):
    if series.empty: return 0.0
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    return float(drawdown.min() * 100)

@st.cache_data(ttl=24*3600)
def fetch_historical_data(end_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER):
    start_date = TQQQ_INCEPTION_DATE
    tickers = [TICKER, LEVERAGED_TICKER, INVERSE_TICKER]
    try:
        all_data = yf.download(tickers, start=start_date, end=end_date + timedelta(days=1), progress=False, auto_adjust=False)
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
        df_combined.dropna(subset=['close', 'high', 'low'], inplace=True)
        return df_combined[df_combined.index.date <= end_date] 
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return pd.DataFrame()

def calculate_true_range_and_atr(df, atr_period):
    high_minus_low = df['high'] - df['low']
    high_minus_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_minus_prev_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.DataFrame({'hl': high_minus_low, 'hpc': high_minus_prev_close, 'lpc': low_minus_prev_close}).max(axis=1)
    return tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean()

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

# --- Backtest & Simulation ---

@st.cache_data(ttl=24*3600)
def generate_historical_signals(df, LEVERAGED_TICKER, mini_vasl_mult):
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
        vasl, mini_vasl = ema5 - (2.0 * atr), ema5 - (mini_vasl_mult * atr)
        
        if price >= sma200:
            target = LEVERAGED_TICKER if (adx >= DEFAULT_ADX_THRESHOLD and price >= vasl and price >= mini_vasl) else 'CASH'
        else:
            target = LEVERAGED_TICKER if (ema5 > sma20 and slope > 0) else 'CASH'
        df.at[index, 'Trade_Ticker'] = target
    return df.dropna(subset=[f'SMA_{SMA_PERIOD}'])

@st.cache_data(ttl=6*3600)
def run_simulation_with_tax(signals_df, start_date, TICKER, LEVERAGED_TICKER, tax_rate):
    sim_df = signals_df[signals_df.index.date >= start_date].copy()
    portfolio_value, shares, current_ticker = INITIAL_INVESTMENT, Decimal("0"), 'CASH'
    cost_basis, loss_carry, yearly_gain, total_tax_paid = Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0")
    history = []
    
    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        prices = {t: Decimal(str(row[f'{t}_close'])) for t in [TICKER, LEVERAGED_TICKER]}
        target = row['Trade_Ticker']
        
        if target != current_ticker:
            if current_ticker != 'CASH':
                val = shares * prices[current_ticker]
                yearly_gain += (val - cost_basis)
                portfolio_value = val
                history.append({'Date': row.name.date(), 'Action': f"SELL {current_ticker}", 'Portfolio Value': float(portfolio_value)})
            if target != 'CASH':
                shares = portfolio_value / prices[target]
                cost_basis = portfolio_value
                history.append({'Date': row.name.date(), 'Action': f"BUY {target}", 'Portfolio Value': float(portfolio_value)})
            current_ticker = target

        if current_ticker != 'CASH': portfolio_value = shares * prices[current_ticker]
        
        # Yearly Tax Deduction
        if (i == len(sim_df)-1) or (sim_df.index[i+1].year > row.name.year):
            tax_due = max(Decimal("0"), (yearly_gain - loss_carry) * Decimal(str(tax_rate)))
            if tax_due > 0:
                portfolio_value -= tax_due
                total_tax_paid += tax_due
                if current_ticker != 'CASH': shares = portfolio_value / prices[current_ticker]
                history.append({'Date': row.name.date(), 'Action': "TAX PAID", 'Portfolio Value': float(portfolio_value)})
            loss_carry = max(Decimal("0"), loss_carry - yearly_gain if yearly_gain > 0 else loss_carry + abs(yearly_gain))
            yearly_gain = Decimal("0")
            
        sim_df.at[row.name, 'Portfolio_Value'] = float(portfolio_value)
    
    return sim_df, pd.DataFrame(history), float(total_tax_paid)

# --- App Interface ---

def main():
    st.set_page_config(layout="wide", page_title="TQQQ v22td Pro")
    st.title("⭐ TQQQ Volatility Strategy (v22td)")
    st.markdown("---")

    last_day = get_last_closed_trading_day()
    
    with st.sidebar:
        st.header("Parameters")
        TICKER = st.text_input("Underlying", "QQQ")
        LEVERAGED_TICKER = st.text_input("Leveraged (3x)", "TQQQ")
        start_date = st.date_input("Start Date", get_default_ytd_start_date(last_day))
        m_vasl = st.number_input("Mini-VASL Multiplier", 0.0, 3.0, 1.5)
        st_tax = st.number_input("Annual Tax Rate", 0.0, 0.5, 0.25)
        if st.button("Run Analysis", type="primary"): st.rerun()

    data = fetch_historical_data(last_day, TICKER, LEVERAGED_TICKER, "SQQQ")
    if data.empty: return

    inds = calculate_indicators(data, last_day, data['close'].iloc[-1])
    sig_df = generate_historical_signals(data, LEVERAGED_TICKER, m_vasl)
    results, history, total_tax = run_simulation_with_tax(sig_df, start_date, TICKER, LEVERAGED_TICKER, st_tax)

    # 1. LIVE SIGNAL
    st.subheader("Live Trading Signal")
    col_act, col_tick = st.columns([3, 1])
    curr_sig = history.iloc[-1]['Action'] if not history.empty else "CASH"
    status = "BUY/HOLD" if LEVERAGED_TICKER in curr_sig else "CASH"
    col_act.markdown(f"## :rotating_light: **{status}**")
    col_tick.markdown(f"## **{LEVERAGED_TICKER if status == 'BUY/HOLD' else 'CASH'}**")
    
    m1, m2 = st.columns(2)
    m1.metric(f"{TICKER} Price", f"${inds['current_price']:.2f}")
    m2.metric("Market Regime", "BULL" if inds['current_price'] >= inds['sma_200'] else "BEAR")
    st.markdown("---")

    # 2. PERFORMANCE SUMMARY (Including MDD and Tax)
    st.markdown("## 📊 Backtest Performance Summary")
    final_v = results['Portfolio_Value'].iloc[-1]
    
    # Benchmarks
    bh_q_vals = (data[f'{TICKER}_close'].loc[results.index] / data[f'{TICKER}_close'].loc[results.index[0]]) * 10000
    bh_t_vals = (data[f'{LEVERAGED_TICKER}_close'].loc[results.index] / data[f'{LEVERAGED_TICKER}_close'].loc[results.index[0]]) * 10000

    summary_df = pd.DataFrame({
        'Metric': [f"Strategy ({LEVERAGED_TICKER})", f"B&H {TICKER}", f"B&H {LEVERAGED_TICKER}"],
        'Final Value': [final_v, bh_q_vals.iloc[-1], bh_t_vals.iloc[-1]],
        'Total Return': [(final_v/10000-1)*100, (bh_q_vals.iloc[-1]/10000-1)*100, (bh_t_vals.iloc[-1]/10000-1)*100],
        'Max Drawdown': [calculate_max_drawdown(results['Portfolio_Value']), calculate_max_drawdown(bh_q_vals), calculate_max_drawdown(bh_t_vals)],
        'Total Taxes Paid': [total_tax, 0.0, 0.0]
    })
    st.dataframe(summary_df.style.format({
        'Final Value': '${:,.2f}', 
        'Total Return': '{:+.2f}%', 
        'Max Drawdown': '{:,.2f}%',
        'Total Taxes Paid': '${:,.2f}'
    }), hide_index=True, width='stretch')

    # 3. EQUITY CURVE
    st.markdown("## 📈 Equity Curve")
    chart = alt.Chart(results.reset_index()).mark_line(color='blue').encode(
        x='Date:T', y=alt.Y('Portfolio_Value:Q', title="Portfolio Value ($)"),
        tooltip=['Date:T', alt.Tooltip('Portfolio_Value:Q', format='$,.2f')]
    ).interactive()
    st.altair_chart(chart, use_container_width=True)

    with st.expander("🧾 View Trade & Tax History"):
        st.dataframe(history, hide_index=True, width='stretch')

if __name__ == "__main__":
    main()
