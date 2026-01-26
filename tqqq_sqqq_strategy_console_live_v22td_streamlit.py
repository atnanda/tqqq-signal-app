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
DEFAULT_ADX_THRESHOLD = 12.0 # Hidden from UI, kept for strategy logic

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
        'atr': df['ATR'].ffill().iloc[-1]
    }

# --- Visual/Chart Helpers ---

def analyze_trade_pairs(trade_history_df, full_data, TICKER):
    temp_df = trade_history_df.copy()
    trades = temp_df[temp_df['Action'].str.startswith(('BUY', 'SELL'))].copy()
    trades['Date_dt'] = pd.to_datetime(trades['Date'])
    trades = trades.set_index('Date_dt').join(full_data[f'{TICKER}_close'], how='left').reset_index()
    
    trade_pairs, buy_trade = [], None
    for _, row in trades.iterrows():
        if row['Action'].startswith('BUY'):
            buy_trade = {'date': row['Date_dt'], 'price': row[f'{TICKER}_close'], 'val': Decimal(str(row['Portfolio Value'])), 'asset': row['Asset']}
        elif row['Action'].startswith('SELL') and buy_trade:
            pl = float(Decimal(str(row['Portfolio Value'])) - buy_trade['val'])
            trade_pairs.append({
                'buy_date': buy_trade['date'], 'sell_date': row['Date_dt'],
                'buy_price': buy_trade['price'], 'sell_price': row[f'{TICKER}_close'],
                'profit_loss': pl, 'is_profitable_str': "Profit" if pl >= 0 else "Loss"
            })
            buy_trade = None
    return trade_pairs

def plot_v22td_signals(signals_df, trade_pairs, TICKER, start_date):
    plot_data = signals_df[signals_df.index.date >= start_date].copy()
    price_cols = ['close', f'SMA_{SMA_PERIOD}', f'SMA_{SMA_SHORT_PERIOD}', f'EMA_{EMA_PERIOD}']
    plot_long = plot_data.reset_index().rename(columns={'index': 'Date'})[['Date'] + price_cols].melt('Date', var_name='Metric', value_name='Price')

    selection = alt.selection_point(fields=['Metric'], bind='legend', name='MetricSelect') 
    base = alt.Chart(plot_long).encode(x=alt.X('Date:T', title='Date')).properties(height=500)
    
    price_line = base.mark_line(color='gray', opacity=0.4, size=1).encode(
        y=alt.Y('Price:Q', title=f'{TICKER} Price ($)'),
        tooltip=[alt.Tooltip('Price:Q', format='$.2f')]
    ).transform_filter(alt.datum.Metric == 'close')
    
    indicators = base.mark_line().encode(
        y='Price:Q',
        color=alt.Color('Metric:N', scale=alt.Scale(domain=price_cols[1:], range=['orange', 'blue', 'purple'])),
        opacity=alt.condition(selection, alt.value(1.0), alt.value(0.1))
    ).add_params(selection).transform_filter(alt.datum.Metric != 'close')

    df_segments = []
    for i, t in enumerate(trade_pairs):
        df_segments.append({'tid': i, 'Date': t['buy_date'], 'Price': t['buy_price'], 'Cat': t['is_profitable_str']})
        df_segments.append({'tid': i, 'Date': t['sell_date'], 'Price': t['sell_price'], 'Cat': t['is_profitable_str']})
    
    segments = alt.Chart(pd.DataFrame(df_segments)).mark_line(size=3).encode(
        x='Date:T', y='Price:Q', detail='tid:N',
        color=alt.condition(alt.datum.Cat == 'Profit', alt.value('#008000'), alt.value('#d62728'))
    )
    return (price_line + indicators + segments).interactive()

# --- Execution Engine ---

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
def run_tax_sim(signals_df, start_date, TICKER, LEVERAGED_TICKER, tax_rate):
    sim_df = signals_df[signals_df.index.date >= start_date].copy()
    portfolio_value, shares, current_ticker = INITIAL_INVESTMENT, Decimal("0"), 'CASH'
    cost_basis, loss_carry, yearly_gain = Decimal("0"), Decimal("0"), Decimal("0")
    history, tax_logs = [], []
    
    for i in range(len(sim_df)):
        row = sim_df.iloc[i]
        prices = {t: Decimal(str(row[f'{t}_close'])) for t in [TICKER, LEVERAGED_TICKER]}
        target = row['Trade_Ticker']
        
        if target != current_ticker:
            if current_ticker != 'CASH':
                val = shares * prices[current_ticker]
                yearly_gain += (val - cost_basis)
                portfolio_value = val
                history.append({'Date': row.name.date(), 'Action': f"SELL {current_ticker}", 'Asset': current_ticker, 'Portfolio Value': float(portfolio_value)})
            if target != 'CASH':
                shares = portfolio_value / prices[target]
                cost_basis = portfolio_value
                history.append({'Date': row.name.date(), 'Action': f"BUY {target}", 'Asset': target, 'Portfolio Value': float(portfolio_value)})
            current_ticker = target

        if current_ticker != 'CASH': portfolio_value = shares * prices[current_ticker]
        
        if (i == len(sim_df)-1) or (sim_df.index[i+1].year > row.name.year):
            tax_due = max(Decimal("0"), (yearly_gain - loss_carry) * Decimal(str(tax_rate)))
            if tax_due > 0:
                portfolio_value -= tax_due
                if current_ticker != 'CASH': shares = portfolio_value / prices[current_ticker]
                history.append({'Date': row.name.date(), 'Action': "ANNUAL TAX", 'Asset': 'CASH', 'Portfolio Value': float(portfolio_value)})
            tax_logs.append({'Year': row.name.year, 'Realized Gain': float(yearly_gain), 'Tax Paid': float(tax_due)})
            loss_carry = max(Decimal("0"), loss_carry - yearly_gain if yearly_gain > 0 else loss_carry + abs(yearly_gain))
            yearly_gain = Decimal("0")
            
        sim_df.at[row.name, 'Portfolio_Value'] = float(portfolio_value)
    return sim_df, pd.DataFrame(history), pd.DataFrame(tax_logs)

# --- Main App Interface ---

def main():
    st.set_page_config(layout="wide", page_title="TQQQ v22td Pro")
    st.title("⭐ Leveraged Volatility Strategy (v22td)")
    st.markdown("---")

    last_day = get_last_closed_trading_day()
    
    with st.sidebar:
        st.header("Strategy Parameters")
        TICKER = st.text_input("Underlying", "QQQ")
        LEVERAGED_TICKER = st.text_input("Leveraged (3x)", "TQQQ")
        start_date = st.date_input("Start Date", get_default_ytd_start_date(last_day))
        # Mini-VASL ATR multiplier is still useful for risk tuning
        m_vasl = st.number_input("Mini-VASL Multiplier", 0.0, 3.0, 1.5)
        st_tax = st.number_input("Annual Tax Rate", 0.0, 0.5, 0.25)
        if st.button("Run Analysis", type="primary"): st.rerun()

    data = fetch_historical_data(last_day, TICKER, LEVERAGED_TICKER, "SQQQ")
    if data.empty: return

    inds = calculate_indicators(data, last_day, data['close'].iloc[-1])
    sig_df = generate_historical_signals(data, LEVERAGED_TICKER, m_vasl)
    results, history, tax_logs = run_tax_sim(sig_df, start_date, TICKER, LEVERAGED_TICKER, st_tax)

    # 1. LIVE SIGNAL HEADER
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

    # 2. PERFORMANCE SUMMARY
    st.markdown("## 📊 Backtest Performance Summary")
    final_v = results['Portfolio_Value'].iloc[-1]
    bh_q = (data[f'{TICKER}_close'].loc[results.index[-1]] / data[f'{TICKER}_close'].loc[results.index[0]]) * 10000
    bh_t = (data[f'{LEVERAGED_TICKER}_close'].loc[results.index[-1]] / data[f'{LEVERAGED_TICKER}_close'].loc[results.index[0]]) * 10000

    summary_df = pd.DataFrame({
        'Metric': [f"Strategy ({LEVERAGED_TICKER})", f"B&H {TICKER}", f"B&H {LEVERAGED_TICKER}"],
        'Final Value': [final_v, float(bh_q), float(bh_t)],
        'Total Return': [(final_v/10000-1)*100, (float(bh_q)/10000-1)*100, (float(bh_t)/10000-1)*100]
    })
    st.dataframe(summary_df.style.format({'Final Value': '${:,.2f}', 'Total Return': '{:+.2f}%'}), hide_index=True, width='stretch')

    # 3. INTERACTIVE CHART
    st.markdown("## 📈 Interactive Price and Trade Signal Chart")
    trade_pairs = analyze_trade_pairs(history, data, TICKER)
    st.altair_chart(plot_v22td_signals(sig_df, trade_pairs, TICKER, start_date), use_container_width=True)

    # 4. TRADE HISTORY
    with st.expander("🧾 Detailed Trade History"):
        st.dataframe(history.style.format({'Portfolio Value': '${:,.2f}'}), hide_index=True, width='stretch')

if __name__ == "__main__":
    main()
