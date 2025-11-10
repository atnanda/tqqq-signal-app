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

# Strategy Parameters
EMA_PERIOD = 5
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
SMA_PERIOD = 200
SMA_SHORT_PERIOD = 20
INITIAL_INVESTMENT = Decimal("10000.00")
TQQQ_INCEPTION_DATE = date(2010, 2, 9)
MINI_VASL_MULTIPLIER = 1.5 

# --- Helper Functions ---

def get_last_closed_trading_day():
    """Determines the most recent CLOSED trading day."""
    today = date.today()
    last_day = today - timedelta(days=1)
    while last_day.weekday() > 4: # 5 is Sat, 6 is Sun
        last_day -= timedelta(days=1)
    return last_day

def get_default_ytd_start_date(today_date):
    """Returns January 1st of the current year."""
    return date(today_date.year, 1, 1)

@st.cache_data(ttl=24*3600) # Cache data for 24 hours
def fetch_historical_data(end_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER):
    """Fetches historical data for QQQ, TQQQ, and SQQQ up to a specified end_date."""
    start_date = TQQQ_INCEPTION_DATE
    tickers = [TICKER, LEVERAGED_TICKER, INVERSE_TICKER]
    market_end_date = end_date + timedelta(days=1) 
    
    try:
        all_data = yf.download(tickers, start=start_date, end=market_end_date, interval="1d", progress=False, auto_adjust=False, timeout=15)
        
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
        st.error(f"FATAL ERROR during historical data download: {e}")
        return pd.DataFrame()

def fetch_live_price(ticker):
    """Fetches the current live market price."""
    try:
        ticker_info = yf.Ticker(ticker).info
        live_price = ticker_info.get('regularMarketPrice')
        return live_price if live_price and live_price > 0 else None
    except Exception:
        return None
        
def calculate_true_range_and_atr(df, atr_period):
    """Calculates True Range and EMA-based Average True Range (ATR)."""
    high_minus_low = df['high'] - df['low']
    high_minus_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_minus_prev_close = np.abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.DataFrame({'hl': high_minus_low, 'hpc': high_minus_prev_close, 'lpc': low_minus_prev_close}).max(axis=1)
    atr_series = true_range.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean()
    
    return atr_series 

def calculate_indicators(data_daily, target_date, current_price):
    """Calculates all indicators using data only up to target_date (Close Price)."""
    df = data_daily[data_daily.index.date <= target_date].copy() 
    
    if df.empty:
        raise ValueError("No data available for indicator calculation.")
        
    df.ta.sma(length=SMA_PERIOD, append=True)
    df.ta.sma(length=SMA_SHORT_PERIOD, append=True)
    df.ta.ema(length=EMA_PERIOD, append=True)
    
    sma_200_col = f'SMA_{SMA_PERIOD}'
    
    if df[sma_200_col].isnull().all():
        raise ValueError(f"Insufficient data (need {SMA_PERIOD} days) to calculate 200-Day SMA.")
        
    current_sma_200 = df[sma_200_col].iloc[-1]
    latest_sma_20 = df[f'SMA_{SMA_SHORT_PERIOD}'].iloc[-1] 
    latest_ema_5 = df[f'EMA_{EMA_PERIOD}'].iloc[-1]
    df['ATR'] = calculate_true_range_and_atr(df, ATR_PERIOD)
    latest_atr = df['ATR'].ffill().iloc[-1]

    if not all(np.isfinite([current_sma_200, latest_ema_5, latest_atr])): 
        raise ValueError("Indicator calculation resulted in non-numeric values.")
    
    return {
        'current_price': current_price,
        'sma_200': current_sma_200,
        'sma_20': latest_sma_20,
        'ema_5': latest_ema_5,
        'atr': latest_atr
    }

def generate_signal(indicators, LEVERAGED_TICKER, MINI_VASL_MULTIPLIER):
    """Applies the trading strategy logic."""
    price = indicators['current_price'] 
    sma_200 = indicators['sma_200']
    ema_5 = indicators['ema_5']
    atr = indicators['atr']

    vasl_trigger_level = ema_5 - (ATR_MULTIPLIER * atr)           
    mini_vasl_exit_level = ema_5 - (MINI_VASL_MULTIPLIER * atr)                    
    
    trade_ticker = 'CASH'
    conviction_status = "N/A"
    final_signal = "CASH (Default)" 

    dma_bull_price = (price >= sma_200)
    
    if dma_bull_price:
        
        if price < vasl_trigger_level:
            trade_ticker = 'CASH'
            conviction_status = "DMA Bull - VASL Triggered (2.0x ATR Exit)"
            final_signal = f"SELL {LEVERAGED_TICKER} / CASH (DMA Bull - Hard Stop)"
            
        elif price < mini_vasl_exit_level: 
             trade_ticker = 'CASH'
             conviction_status = f"DMA Bull - Mini-VASL Exit ({MINI_VASL_MULTIPLIER:.1f}x ATR Exit)"
             final_signal = f"SELL {LEVERAGED_TICKER} / CASH (DMA Bull - Soft Stop)"
        
        else:
            trade_ticker = LEVERAGED_TICKER
            conviction_status = f"DMA - Bull ({LEVERAGED_TICKER} Entry/Hold)"
            final_signal = f"BUY {LEVERAGED_TICKER} / HOLD {LEVERAGED_TICKER}"
                    
    else: 
        conviction_status = "DMA - Bear (Stay CASH - Inverse Trading Disabled)"
        final_signal = "CASH (Inverse Trading Disabled)"
        trade_ticker = 'CASH'

    return {
        'signal': final_signal,
        'trade_ticker': trade_ticker,
        'conviction': conviction_status,
        'vasl_down': vasl_trigger_level,
        'mini_vasl_down': mini_vasl_exit_level,
    }

def calculate_max_drawdown(series):
    """Calculates the Maximum Drawdown for a given series of values."""
    if series.empty: return 0.0
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    return drawdown.min() * 100

def analyze_trade_pairs(trade_history_df, full_data, TICKER):
    """Processes the trade history to create Buy-Sell pairs with profit/loss."""
    temp_df = trade_history_df.copy()
    temp_df['Portfolio Value Float'] = temp_df['Portfolio Value'].astype(float)
    
    trades = temp_df[temp_df['Action'].str.startswith(('BUY', 'SELL'))].copy()
    trades['Date_dt'] = pd.to_datetime(trades['Date'])

    qqq_close_col = f'{TICKER}_close'
    
    trades = trades.set_index('Date_dt') 
    trades.sort_index(inplace=True)
    trades = trades.join(full_data[qqq_close_col], how='left')
    trades.rename(columns={qqq_close_col: 'QQQ_Price'}, inplace=True)
    trades = trades.reset_index()

    trade_pairs = []
    buy_trade = None
    
    for _, row in trades.iterrows():
        action = row['Action']
        
        if action.startswith('BUY'):
            if buy_trade: buy_trade = None 
            buy_trade = {
                'buy_date': row['Date_dt'],
                'buy_qqq_price': row['QQQ_Price'],
                'buy_portfolio_value': Decimal(str(row['Portfolio Value Float'])),
                'asset': row['Asset']
            }
        elif action.startswith('SELL'):
            if buy_trade:
                sell_portfolio_value = Decimal(str(row['Portfolio Value Float']))
                profit_loss = float(sell_portfolio_value - buy_trade['buy_portfolio_value'])
                
                trade_pairs.append({
                    'buy_date': buy_trade['buy_date'],
                    'sell_date': row['Date_dt'],
                    'buy_qqq_price': buy_trade['buy_qqq_price'],
                    'sell_qqq_price': row['QQQ_Price'],
                    'profit_loss': profit_loss,
                    'is_profitable': profit_loss >= 0, 
                    'asset': buy_trade['asset']
                })
                buy_trade = None
                
    return trade_pairs

def plot_trade_signals(data_daily, trade_pairs, TICKER):
    """Generates an Altair chart for Streamlit display."""
    plot_data = data_daily.copy()
    price_cols = ['close', f'SMA_{SMA_PERIOD}', f'SMA_{SMA_SHORT_PERIOD}', f'EMA_{EMA_PERIOD}'] 
    plot_data_long = plot_data.reset_index().rename(columns={'index': 'Date'})[['Date'] + price_cols].melt('Date', var_name='Metric', value_name='Price')

    trade_segments = []
    for i, trade in enumerate(trade_pairs):
        trade_segments.append({'trade_id': i, 'Date': trade['buy_date'], 'Price': trade['buy_qqq_price'], 'Is_Profitable': trade['is_profitable'], 'Signal': 'Buy', 'P_L': trade['profit_loss']})
        trade_segments.append({'trade_id': i, 'Date': trade['sell_date'], 'Price': trade['sell_qqq_price'], 'Is_Profitable': trade['is_profitable'], 'Signal': 'Sell', 'P_L': trade['profit_loss']})
        
    df_segments = pd.DataFrame(trade_segments)
    
    base = alt.Chart(plot_data_long).encode(x=alt.X('Date:T', title='Date')).properties(title=f'{TICKER} Price and Strategy Signals', width='container', height=500)
    
    price_line = base.mark_line(color='gray', opacity=0.7, size=0.5).encode(
        y=alt.Y('Price:Q', title=f'{TICKER} Price ($)'),
    ).transform_filter(alt.datum.Metric == 'close')
    
    indicator_lines = base.mark_line().encode(
        y=alt.Y('Price:Q'),
        color=alt.Color('Metric:N', scale=alt.Scale(domain=[f'SMA_{SMA_PERIOD}', f'SMA_{SMA_SHORT_PERIOD}', f'EMA_{EMA_PERIOD}'], range=['orange', 'blue', 'purple']), legend=alt.Legend(title="Indicator")), 
        strokeDash=alt.condition(alt.datum.Metric == f'SMA_{SMA_PERIOD}', alt.value([5, 5]), alt.value([2, 2])),
    ).transform_filter((alt.datum.Metric != 'close'))

    segment_lines = alt.Chart(df_segments).mark_line(size=3).encode(
        x=alt.X('Date:T'),
        y=alt.Y('Price:Q'),
        detail='trade_id:N',
        color=alt.Color('Is_Profitable:N', scale=alt.Scale(domain=[True, False], range=['green', 'red']), legend=alt.Legend(title="Trade P/L")),
        tooltip=[alt.Tooltip('P_L:Q', title='P/L', format='+.2f')]
    )

    return (price_line + indicator_lines + segment_lines).interactive()

class BacktestEngine:
    def __init__(self, historical_data, LEVERAGED_TICKER, MINI_VASL_MULTIPLIER):
        self.df = historical_data.copy()
        self.LEVERAGED_TICKER = LEVERAGED_TICKER
        self.MINI_VASL_MULTIPLIER = MINI_VASL_MULTIPLIER
        
    def generate_historical_signals(self):
        self.df.ta.sma(length=SMA_PERIOD, append=True)
        self.df.ta.sma(length=SMA_SHORT_PERIOD, append=True) 
        self.df.ta.ema(length=EMA_PERIOD, append=True)
        self.df['ATR'] = calculate_true_range_and_atr(self.df, ATR_PERIOD)
        
        indicator_cols = [f'SMA_{SMA_PERIOD}', f'SMA_{SMA_SHORT_PERIOD}', f'EMA_{EMA_PERIOD}', 'ATR', 'close'] 
        df_current_day_data = self.df[indicator_cols].copy().rename(columns={'close': 'current_close'})
        self.df['Trade_Ticker'] = 'CASH' 
        
        for index in self.df.index: 
            current_data = df_current_day_data.loc[index]
            
            if pd.isna(current_data[f'SMA_{SMA_PERIOD}']): continue
                
            price = current_data['current_close'] 
            ema_5 = current_data[f'EMA_{EMA_PERIOD}']
            sma_200 = current_data[f'SMA_{SMA_PERIOD}']
            atr = current_data['ATR']

            vasl_trigger_level = ema_5 - (ATR_MULTIPLIER * atr)
            mini_vasl_exit_level = ema_5 - (self.MINI_VASL_MULTIPLIER * atr) 
            
            dma_bull_price = (price >= sma_200)
            
            if dma_bull_price:
                if price < vasl_trigger_level:
                    trade_ticker = 'CASH'
                elif price < mini_vasl_exit_level: 
                     trade_ticker = 'CASH'
                else:
                    trade_ticker = self.LEVERAGED_TICKER
            else: 
                trade_ticker = 'CASH' 
            
            self.df.loc[index, 'Trade_Ticker'] = trade_ticker

        self.df.dropna(subset=[f'SMA_{SMA_PERIOD}', 'Trade_Ticker'], inplace=True)
        return self.df
        
    def run_simulation(self, start_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER):
        sim_df = self.df[self.df.index.date >= start_date].copy()
        if sim_df.empty: 
            return float(INITIAL_INVESTMENT), 0, 0, pd.DataFrame(), 'CASH', pd.DataFrame() 
            
        portfolio_value = INITIAL_INVESTMENT 
        shares = Decimal("0")
        current_ticker = 'CASH'
        trade_history = [] 
        
        trade_history.append({'Date': sim_df.index.min().date(), 'Action': "START", 'Asset': 'CASH', 'Price': 0.0, 'Portfolio Value': float(portfolio_value)})

        # B&H Setup
        qqq_start_price = sim_df.iloc[0][f'{TICKER}_close']
        tqqq_start_price = sim_df.iloc[0][f'{LEVERAGED_TICKER}_close']
        initial_float = float(INITIAL_INVESTMENT)
        sim_df['BH_QQQ_Value'] = initial_float * (sim_df[f'{TICKER}_close'] / qqq_start_price)
        sim_df['BH_TQQQ_Value'] = initial_float * (sim_df[f'{LEVERAGED_TICKER}_close'] / tqqq_start_price)
        
        for i in range(len(sim_df)):
            current_day = sim_df.iloc[i]
            trade_ticker = current_day['Trade_Ticker']
            current_day_prices = {t: Decimal(str(current_day[f'{t}_close'])) for t in [TICKER, LEVERAGED_TICKER, INVERSE_TICKER]}
            
            if trade_ticker != current_ticker:
                # SELL
                if current_ticker != 'CASH':
                    sell_price = current_day_prices.get(current_ticker, Decimal("0"))
                    realized_cash = shares * sell_price if sell_price > 0 else Decimal("0")
                    portfolio_value = realized_cash
                    trade_history.append({'Date': current_day.name.date(), 'Action': f"SELL {current_ticker}", 'Asset': current_ticker, 'Price': float(sell_price), 'Portfolio Value': float(realized_cash)})
                    shares = Decimal("0")
                
                # BUY
                if trade_ticker != 'CASH':
                    buy_price = current_day_prices.get(trade_ticker, Decimal("0"))
                    if buy_price > 0 and portfolio_value > 0:
                        shares = portfolio_value / buy_price
                    else:
                        shares = Decimal("0")
                    portfolio_value = shares * buy_price if shares > 0 else portfolio_value
                    trade_history.append({'Date': current_day.name.date(), 'Action': f"BUY {trade_ticker}", 'Asset': trade_ticker, 'Price': float(buy_price), 'Portfolio Value': float(portfolio_value)})
                        
                current_ticker = trade_ticker

            # TRACKING
            if current_ticker != 'CASH' and shares > 0:
                current_price = current_day_prices.get(current_ticker, Decimal("0"))
                portfolio_value = shares * current_price
            
            sim_df.loc[current_day.name, 'Portfolio_Value'] = float(portfolio_value) 

        final_holding = current_ticker
        final_value = float(portfolio_value)

        return final_value, sim_df['BH_QQQ_Value'].iloc[-1], sim_df['BH_TQQQ_Value'].iloc[-1], pd.DataFrame(trade_history), final_holding, sim_df
# --- Streamlit Application ---

def run_analysis(backtest_start_date, target_signal_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER, current_mini_vasl_multiplier):
    """Encapsulates the entire backtest and rendering process."""
    with st.spinner("Fetching data and running backtest..."):
        
        # 1. Fetch Data
        full_data = fetch_historical_data(target_signal_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER)
        
        if full_data.empty:
            st.error("Could not fetch necessary historical data. Check tickers or try a different date range.")
            return

        # 2. Get Live Price and Indicators
        live_price = fetch_live_price(TICKER)
        signal_price = live_price if live_price is not None else full_data['close'].iloc[-1].item()
        price_source = "LIVE PRICE" if live_price is not None else "HISTORICAL CLOSE"

        try:
            indicators = calculate_indicators(full_data, target_signal_date, signal_price)
            signal_results = generate_signal(indicators, LEVERAGED_TICKER, current_mini_vasl_multiplier)
        except ValueError as e:
            st.error(f"Error in signal generation: {e}")
            return

        # 3. Run Backtest
        backtester = BacktestEngine(full_data, LEVERAGED_TICKER, current_mini_vasl_multiplier)
        signals_df = backtester.generate_historical_signals()

        start_of_tradable_data = signals_df.index.min().date() 
        backtest_start = max(backtest_start_date, start_of_tradable_data) 
        
        final_value, bh_qqq, bh_tqqq, trade_history_df, final_holding, sim_df = backtester.run_simulation(backtest_start, TICKER, LEVERAGED_TICKER, INVERSE_TICKER)

        # 4. Performance Metrics Calculation
        last_trade_day = signals_df.index.max().date() 
        days_held = (last_trade_day - backtest_start).days
        years_held = days_held / 365.25 if days_held > 0 else 1 
        initial_float = float(INITIAL_INVESTMENT)

        strategy_mdd = calculate_max_drawdown(sim_df['Portfolio_Value'])
        bh_qqq_mdd = calculate_max_drawdown(sim_df['BH_QQQ_Value'])
        bh_tqqq_mdd = calculate_max_drawdown(sim_df['BH_TQQQ_Value'])
        
        calc_cagr = lambda final, initial, years: ((final / initial) ** (1 / years) - 1) * 100 if final > 0 and initial > 0 and years > 0 else 0.0
        strategy_cagr = calc_cagr(final_value, initial_float, years_held)
        bh_qqq_cagr = calc_cagr(bh_qqq, initial_float, years_held)
        bh_tqqq_cagr = calc_cagr(bh_tqqq, initial_float, years_held)

    st.markdown("## 📊 Strategy Results & Signal")
    st.markdown(f"**Backtest Period:** `{backtest_start.strftime('%Y-%m-%d')}` to `{last_trade_day.strftime('%Y-%m-%d')}`")
    st.markdown("---")
    
    # --- Action Section (MOVED AND BOLDED) ---
    st.markdown("### **Trade Action**")
    st.markdown(f"## :rotating_light: **{signal_results['signal']}**")
    st.caption(f"**Conviction:** {signal_results['conviction']}")
    st.markdown("---")

    # --- Live Signal Section ---
    st.subheader("Live Trading Signal Details")
    
    col1, col2 = st.columns(2)
    col1.metric("Price Used", f"${indicators['current_price']:.2f}", help=f"Source: {price_source}")
    col2.metric("Recommended Ticker", signal_results['trade_ticker'])

    st.markdown("---")

    # --- Performance Summary ---
    st.markdown("## 💰 Backtest Performance Summary")
    
    summary_data = {
        'Metric': [f"Strategy ({LEVERAGED_TICKER})", f"{TICKER} Buy & Hold", f"{LEVERAGED_TICKER} Buy & Hold"], 
        'Final Value': [final_value, bh_qqq, bh_tqqq],
        'Total Return': [(final_value/initial_float - 1) * 100, (bh_qqq/initial_float - 1) * 100, (bh_tqqq/initial_float - 1) * 100], 
        'CAGR': [strategy_cagr, bh_qqq_cagr, bh_tqqq_cagr],
        'Max Drawdown': [strategy_mdd, bh_qqq_mdd, bh_tqqq_mdd] 
    }
    df_summary = pd.DataFrame(summary_data)
    
    st.dataframe(df_summary.style.format({
        'Final Value': '${:,.2f}', 
        'Total Return': '{:+.2f}%', 
        'CAGR': '{:+.2f}%', 
        'Max Drawdown': '{:,.2f}%'
    }), hide_index=True, use_container_width=True)

    # --- Interactive Plot ---
    st.markdown("## 📈 Interactive Price and Trade Signal Chart")
    
    if len(trade_history_df[trade_history_df['Action'].str.startswith('BUY')]) > 0:
        trade_pairs = analyze_trade_pairs(trade_history_df, full_data, TICKER)
        plot_data_full = signals_df[signals_df.index.date >= backtest_start].copy()
        
        st.altair_chart(plot_trade_signals(plot_data_full, trade_pairs, TICKER), use_container_width=True)
        st.caption(f"Chart shows {TICKER} price with trade segments. Green = profitable trade, Red = losing trade.")
    else:
        st.warning("No trades were executed in the selected backtest period.")

    # --- Detailed Trade History (OPTIMIZATION: Use Expander) ---
    with st.expander("📜 Detailed Trade History (Click to Expand)"):
        trade_history_display = trade_history_df.copy()
        trade_history_display['Date'] = trade_history_display['Date'].astype(str)
        
        st.dataframe(trade_history_display.style.format({
            'Price': '${:,.2f}', 
            'Portfolio Value': '${:,.2f}'
        }), hide_index=True, use_container_width=True)
        st.caption("Trades are executed at the Close price of the Date shown.")

def main_app():
    
    st.set_page_config(layout="wide", page_title="TQQQ Volatility Strategy Backtester")
    st.title("📈 Leveraged ETF Volatility Strategy Backtester")
    st.markdown("---")

    # Get default dates
    last_closed_day = get_last_closed_trading_day()
    default_ytd_start = get_default_ytd_start_date(last_closed_day)

    # --- Sidebar Inputs ---
    with st.sidebar:
        st.header("Strategy Parameters")
        
        # Tickers
        st.subheader("Tickers")
        TICKER = st.text_input("Underlying ETF", "QQQ")
        LEVERAGED_TICKER = st.text_input("Leveraged ETF (3X)", "TQQQ")
        INVERSE_TICKER = st.text_input("Inverse ETF (3X)", "SQQQ")
        
        # Dates (Default is YTD)
        st.subheader("Backtest Period")
        backtest_start_date = st.date_input("Start Date", default_ytd_start, min_value=TQQQ_INCEPTION_DATE)
        target_signal_date = st.date_input("End Date / Signal Date", last_closed_day)
        
        # Strategy Customization
        st.subheader("Risk Control")
        current_mini_vasl_multiplier = st.number_input(
            f"Soft Stop Multiplier ({TICKER} Mini-VASL ATR)", 
            min_value=0.5, 
            max_value=3.0, 
            value=MINI_VASL_MULTIPLIER, 
            step=0.1
        )
        st.caption(f"Hard Stop Multiplier: {ATR_MULTIPLIER}x ATR (Fixed)")
        
        st.markdown("---")
        if st.button("Re-Run Analysis", type="primary"):
            pass 
            
    # --- Auto-Run Logic ---
    run_analysis(backtest_start_date, target_signal_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER, current_mini_vasl_multiplier)


if __name__ == "__main__":
    main_app()
