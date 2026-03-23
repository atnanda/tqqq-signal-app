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

# Technical Indicator Parameters
EMA_PERIOD = 5
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
SMA_PERIOD = 200
SMA_SHORT_PERIOD = 20

# Macro Tickers
TNX_TICKER = "^TNX" # 10-Year Treasury Yield
IRX_TICKER = "^IRX" # 13-Week Treasury Bill

# Trading Parameters
INITIAL_INVESTMENT = Decimal("10000.00")
TQQQ_INCEPTION_DATE = date(2010, 2, 9)
DEFAULT_ADX_THRESHOLD = 12.0
DEFAULT_STEEPENING_THRESHOLD = 0.16
DEFAULT_MACRO_BUFFER = 0.03
DEFAULT_DI_SPREAD_PCT = -0.8
SLIPPAGE_PCT = 0.002  
COMMISSION_PER_TRADE = 0.0  

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

@st.cache_data(ttl=3600)
def fetch_historical_data(end_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER):
    start_date = TQQQ_INCEPTION_DATE
    tickers = [TICKER, LEVERAGED_TICKER, INVERSE_TICKER, TNX_TICKER, IRX_TICKER]
    try:
        all_data = yf.download(tickers, start=start_date, end=end_date + timedelta(days=1), 
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

def calculate_true_range_and_atr(df, atr_period):
    high_minus_low = df['high'] - df['low']
    high_minus_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_minus_prev_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.DataFrame({'hl': high_minus_low, 'hpc': high_minus_prev_close, 
                       'lpc': low_minus_prev_close}).max(axis=1)
    return tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean()

@st.cache_data(ttl=3600)
def calculate_indicators(data_daily, target_date, current_price):
    df = data_daily[data_daily.index.date <= target_date].copy()
    if len(df) < SMA_PERIOD: return None
    
    df.ta.sma(length=SMA_PERIOD, append=True)
    df.ta.sma(length=SMA_SHORT_PERIOD, append=True)
    df.ta.ema(length=EMA_PERIOD, append=True)
    adx_df = pta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX'], df['DMP'], df['DMN'] = adx_df['ADX_14'], adx_df['DMP_14'], adx_df['DMN_14']
    df['SMA_20_slope'] = df[f'SMA_{SMA_SHORT_PERIOD}'].diff(1)
    df['ATR'] = calculate_true_range_and_atr(df, ATR_PERIOD)
    
    return {
        'current_price': current_price,
        'sma_200': df[f'SMA_{SMA_PERIOD}'].iloc[-1],
        'sma_20': df[f'SMA_{SMA_SHORT_PERIOD}'].iloc[-1],
        'sma_20_slope': df['SMA_20_slope'].iloc[-1],
        'ema_5': df[f'EMA_{EMA_PERIOD}'].iloc[-1],
        'atr': df['ATR'].ffill().iloc[-1],
        'adx': df['ADX'].iloc[-1],
        'dmp': df['DMP'].iloc[-1],
        'dmn': df['DMN'].iloc[-1],
        'yield_spread': df['Yield_Spread'].iloc[-1],
        'spread_slope': df['Spread_Slope'].iloc[-1]
    }

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
    df['ATR'] = calculate_true_range_and_atr(df, ATR_PERIOD)
    
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

@st.cache_data(ttl=3600)
def run_tax_sim(signals_df, start_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER, 
                tax_rate, slippage_pct):
    sim_df = signals_df[signals_df.index.date >= start_date].copy()
    portfolio_value, shares, current_ticker = INITIAL_INVESTMENT, Decimal("0"), 'CASH'
    cost_basis, loss_carry, yearly_gain = Decimal("0"), Decimal("0"), Decimal("0")
    history, tax_logs, daily_values = [], [], []
    
    # Benchmarking data
    q_start = Decimal(str(sim_df[f'{TICKER}_close'].iloc[0]))
    t_start = Decimal(str(sim_df[f'{LEVERAGED_TICKER}_close'].iloc[0]))
    
    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        prices = {t: Decimal(str(row[f'{t}_close'])) for t in [TICKER, LEVERAGED_TICKER, INVERSE_TICKER]}
        target = row['Trade_Ticker']
        
        if target != current_ticker:
            if current_ticker != 'CASH':
                sell_price = prices[current_ticker]
                portfolio_value = shares * sell_price
                realized_gain = portfolio_value - cost_basis
                yearly_gain += realized_gain
                history.append({'Date': row.name.date(), 'Action': f"SELL {current_ticker}", 'Asset': current_ticker, 'Price': float(sell_price), 'Portfolio Value': float(portfolio_value), 'Realized P/L': float(realized_gain), 'Reason': row['Trade_Reason']})
            
            if target != 'CASH':
                buy_price = prices[target]
                effective_capital = portfolio_value * (Decimal("1") - Decimal(str(slippage_pct)))
                shares = effective_capital / buy_price
                cost_basis = portfolio_value
                history.append({'Date': row.name.date(), 'Action': f"BUY {target}", 'Asset': target, 'Price': float(buy_price), 'Portfolio Value': float(portfolio_value), 'Realized P/L': 0.0, 'Reason': row['Trade_Reason']})
            current_ticker = target
        
        if current_ticker != 'CASH': portfolio_value = shares * prices[current_ticker]
        
        if (i == len(sim_df)-1) or (sim_df.index[i+1].year > row.name.year):
            taxable_gain = yearly_gain - loss_carry
            tax_due = max(Decimal("0"), taxable_gain * Decimal(str(tax_rate)))
            if tax_due > 0:
                portfolio_value -= tax_due
                if current_ticker != 'CASH': shares = portfolio_value / prices[current_ticker]
                history.append({'Date': row.name.date(), 'Action': "ANNUAL TAX", 'Asset': 'CASH', 'Portfolio Value': float(portfolio_value), 'Realized P/L': float(-tax_due), 'Reason': 'Tax Payment'})
            tax_logs.append({'Year': row.name.year, 'Realized Gain': float(yearly_gain), 'Tax Paid': float(tax_due)})
            loss_carry = max(Decimal("0"), loss_carry - yearly_gain) if yearly_gain > 0 else loss_carry + abs(yearly_gain)
            yearly_gain = Decimal("0")
            
        sim_df.at[row.name, 'Portfolio_Value'] = float(portfolio_value)
        sim_df.at[row.name, 'BH_QQQ'] = float((prices[TICKER] / q_start) * INITIAL_INVESTMENT)
        sim_df.at[row.name, 'BH_TQQQ'] = float((prices[LEVERAGED_TICKER] / t_start) * INITIAL_INVESTMENT)
        daily_values.append({'Date': row.name.date(), 'Portfolio Value': float(portfolio_value)})
    
    return sim_df, pd.DataFrame(history), pd.DataFrame(tax_logs), pd.DataFrame(daily_values)

def calculate_risk_metrics(daily_values_df, history_df, initial_investment):
    dv = daily_values_df.set_index('Date')
    dv['Daily_Return'] = dv['Portfolio Value'].pct_change()
    final_v = dv['Portfolio Value'].iloc[-1]
    years = len(dv) / 252
    ann_ret = ((final_v / float(initial_investment)) ** (1/years) - 1) * 100 if years > 0 else 0
    max_dd = ((dv['Portfolio Value'] - dv['Portfolio Value'].expanding().max()) / dv['Portfolio Value'].expanding().max()).min() * 100
    
    return {
        'Total Return (%)': (final_v / float(initial_investment) - 1) * 100,
        'Annualized Return (%)': ann_ret,
        'Max Drawdown (%)': max_dd,
        'Total Trades': len(history_df[history_df['Action'].str.startswith('BUY')])
    }

def analyze_trade_pairs(history_df, full_data, TICKER):
    trade_pairs = []
    trades = history_df[history_df['Action'].str.startswith(('BUY', 'SELL'))].copy()
    buy_trade = None
    for _, row in trades.iterrows():
        if row['Action'].startswith('BUY'):
            buy_trade = {'date': pd.to_datetime(row['Date']), 'val': row['Portfolio Value'], 'price': full_data.loc[pd.to_datetime(row['Date']), 'close']}
        elif row['Action'].startswith('SELL') and buy_trade:
            pl = row['Portfolio Value'] - buy_trade['val']
            trade_pairs.append({'buy_date': buy_trade['date'], 'sell_date': pd.to_datetime(row['Date']), 'buy_price': buy_trade['price'], 'sell_price': full_data.loc[pd.to_datetime(row['Date']), 'close'], 'is_profitable': pl >= 0})
            buy_trade = None
    return trade_pairs

def plot_enhanced_signals(sig_df, trade_pairs, TICKER, start_date):
    plot_data = sig_df[sig_df.index.date >= start_date].copy()
    base = alt.Chart(plot_data.reset_index()).encode(x='Date:T')
    line = base.mark_line(color='gray', opacity=0.4).encode(y=alt.Y('close:Q', title='Price'))
    
    df_segments = []
    for i, t in enumerate(trade_pairs):
        df_segments.append({'tid': i, 'Date': t['buy_date'], 'Price': t['buy_price'], 'Cat': t['is_profitable']})
        df_segments.append({'tid': i, 'Date': t['sell_date'], 'Price': t['sell_price'], 'Cat': t['is_profitable']})
    
    segments = alt.Chart(pd.DataFrame(df_segments)).mark_line(size=3).encode(
        x='Date:T', y='Price:Q', detail='tid:N',
        color=alt.condition(alt.datum.Cat == True, alt.value('green'), alt.value('red'))
    )
    return (line + segments).interactive()

def main():
    st.set_page_config(layout="wide", page_title="TQQQ v28td_m Benchmarked")
    st.title("⭐ Leveraged Strategy v28td_m (Macro + Benchmarking)")
    st.markdown("---")
    
    last_day = get_last_closed_trading_day()
    
    with st.sidebar:
        st.header("⚙️ Parameters")
        TICKER = st.text_input("Underlying", "QQQ")
        LEVERAGED_TICKER = st.text_input("Leveraged", "TQQQ")
        INVERSE_TICKER = st.text_input("Inverse", "SQQQ")
        start_date = st.date_input("Start Date", get_default_ytd_start_date(last_day))
        
        st.subheader("Filters")
        adx_thresh = st.number_input("ADX Threshold", 0.0, 30.0, DEFAULT_ADX_THRESHOLD)
        di_spread = st.number_input("DI Spread %", -1.0, 1.0, DEFAULT_DI_SPREAD_PCT)
        steep_thresh = st.number_input("Steepening Threshold", 0.0, 1.0, DEFAULT_STEEPENING_THRESHOLD)
        m_buffer = st.number_input("Macro Buffer", 0.0, 0.1, DEFAULT_MACRO_BUFFER)
        
        st.subheader("Costs & Tax")
        slippage = st.number_input("Slippage (%)", 0.0, 1.0, 0.2) / 100
        st_tax = st.number_input("Tax Rate (%)", 0.0, 50.0, 25.0) / 100
        
        if st.button("🚀 Run Analysis", type="primary"):
            st.cache_data.clear()
            st.rerun()

    data = fetch_historical_data(last_day, TICKER, LEVERAGED_TICKER, INVERSE_TICKER)
    if data.empty: return
    
    inds = calculate_indicators(data, last_day, data['close'].iloc[-1])
    sig_df = generate_historical_signals(data, LEVERAGED_TICKER, 1.5, adx_thresh, di_spread, steep_thresh, m_buffer)
    results, history, tax_logs, daily_values = run_tax_sim(sig_df, start_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER, st_tax, slippage)
    risk_metrics = calculate_risk_metrics(daily_values, history, INITIAL_INVESTMENT)

    # Signal Header
    st.subheader("🔴 Live Trading Signal")
    col1, col2, col3, col4 = st.columns(4)
    curr_ticker = history.iloc[-1]['Asset'] if not history.empty else "CASH"
    status_color = "🟢" if curr_ticker != "CASH" else "🟡"
    col1.markdown(f"## {status_color} **{curr_ticker}**")
    col2.metric("Yield Spread", f"{inds['yield_spread']:.4f}")
    col3.metric("Spread Slope", f"{inds['spread_slope']:.4f}")
    col4.metric("ADX", f"{inds['adx']:.1f}")

    st.markdown("---")
    
    # Performance Summary with Benchmarks
    st.markdown("## 📊 Performance Summary")
    b1, b2, b3 = st.columns(3)
    final_v = results['Portfolio_Value'].iloc[-1]
    bh_q = results['BH_QQQ'].iloc[-1]
    bh_t = results['BH_TQQQ'].iloc[-1]
    
    b1.metric("Strategy (Net)", f"${final_v:,.2f}", f"{(final_v/10000-1)*100:+.1f}%")
    b2.metric(f"B&H {TICKER}", f"${bh_q:,.2f}", f"{(bh_q/10000-1)*100:+.1f}%")
    b3.metric(f"B&H {LEVERAGED_TICKER}", f"${bh_t:,.2f}", f"{(bh_t/10000-1)*100:+.1f}%")
    
    st.markdown("### 📈 Risk Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Annualized Return", f"{risk_metrics['Annualized Return (%)']:.2f}%")
    m2.metric("Max Drawdown", f"{risk_metrics['Max Drawdown (%)']:.2f}%")
    m3.metric("Total Trades", f"{risk_metrics['Total Trades']}")

    st.markdown("## 📉 Interactive Performance Chart")
    trade_pairs = analyze_trade_pairs(history, data, TICKER)
    st.altair_chart(plot_enhanced_signals(sig_df, trade_pairs, TICKER, start_date), use_container_width=True)

    col_h, col_t = st.columns(2)
    with col_h:
        with st.expander("🧾 Trade History"):
            st.dataframe(history, use_container_width=True, hide_index=True)
    with col_t:
        with st.expander("💰 Tax Summary"):
            st.dataframe(tax_logs, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
