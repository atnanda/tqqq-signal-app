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
    """Returns January 1st of the current year based on the end date."""
    return date(today_date.year, 1, 1)

@st.cache_data(ttl=24*3600) # Cache data for 24 hours
def fetch_historical_data(end_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER):
    """
    Fetches historical data for QQQ, TQQQ, and SQQQ up to a specified end_date.
    Cached: Only re-runs if end_date or ticker config changes.
    """
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

@st.cache_data(ttl=6*3600) # Cache for 6 hours
def calculate_indicators(data_daily, target_date, current_price):
    """
    Calculates signal-day indicators using data only up to target_date (Close Price).
    Cached: Re-runs only if data/target date changes.
    """
    df = data_daily[data_daily.index.date <= target_date].copy() 
    
    if df.empty:
        raise ValueError("No data available for indicator calculation.")
        
    df.ta.sma(length=SMA_PERIOD, append=True)
    df.ta.ema(length=EMA_PERIOD, append=True)
    
    sma_200_col = f'SMA_{SMA_PERIOD}'
    
    if df[sma_200_col].isnull().all():
        raise ValueError(f"Insufficient data (need {SMA_PERIOD} days) to calculate 200-Day SMA.")
        
    current_sma_200 = df[sma_200_col].iloc[-1]
    latest_ema_5 = df[f'EMA_{EMA_PERIOD}'].iloc[-1]
    df['ATR'] = calculate_true_range_and_atr(df, ATR_PERIOD)
    latest_atr = df['ATR'].ffill().iloc[-1]

    if not all(np.isfinite([current_sma_200, latest_ema_5, latest_atr])): 
        raise ValueError("Indicator calculation resulted in non-numeric values.")
    
    return {
        'current_price': current_price,
        'sma_200': current_sma_200,
        'ema_5': latest_ema_5,
        'atr': latest_atr
    }
    
@st.cache_data(ttl=24*3600) # Cache for 24 hours
def generate_all_historical_signals(data_daily, LEVERAGED_TICKER, MINI_VASL_MULTIPLIER):
    """
    Generates all historical signals (Technical Analysis) for the backtest data.
    Cached: Prevents recalculating all indicators on every Streamlit rerun.
    """
    df = data_daily.copy()
    
    df.ta.sma(length=SMA_PERIOD, append=True)
    df.ta.ema(length=EMA_PERIOD, append=True)
    df['ATR'] = calculate_true_range_and_atr(df, ATR_PERIOD)
    
    indicator_cols = [f'SMA_{SMA_PERIOD}', f'EMA_{EMA_PERIOD}', 'ATR', 'close'] 
    df_current_day_data = df[indicator_cols].copy().rename(columns={'close': 'current_close'})
    df['Trade_Ticker'] = 'CASH' 
    
    for index in df.index: 
        current_data = df_current_day_data.loc[index]
        
        if pd.isna(current_data[f'SMA_{SMA_PERIOD}']): continue
            
        price = current_data['current_close'] 
        ema_5 = current_data[f'EMA_{EMA_PERIOD}']
        sma_200 = current_data[f'SMA_{SMA_PERIOD}']
        atr = current_data['ATR']

        vasl_trigger_level = ema_5 - (ATR_MULTIPLIER * atr)
        mini_vasl_exit_level = ema_5 - (MINI_VASL_MULTIPLIER * atr) 
        
        dma_bull_price = (price >= sma_200)
        
        if dma_bull_price:
            if price < vasl_trigger_level:
                trade_ticker = 'CASH'
            elif price < mini_vasl_exit_level: 
                 trade_ticker = 'CASH'
            else:
                trade_ticker = LEVERAGED_TICKER
        else: 
            trade_ticker = 'CASH' 
        
        df.loc[index, 'Trade_Ticker'] = trade_ticker

    df.dropna(subset=[f'SMA_{SMA_PERIOD}', 'Trade_Ticker'], inplace=True)
    return df

@st.cache_data(ttl=6*3600) # Cache the backtest simulation results
def run_simulation(signals_df: pd.DataFrame, start_date: date, TICKER: str, LEVERAGED_TICKER: str, INVERSE_TICKER: str):
    """
    Performs the backtest simulation using pre-calculated signals.
    Cached: Re-runs only if signals_df (data/parameters) or start_date changes.
    """
    # Use signals_df as the primary input for cache hashing
    sim_df = signals_df[signals_df.index.date >= start_date].copy()
    if sim_df.empty: 
        return float(INITIAL_INVESTMENT), 0, 0, pd.DataFrame(), 'CASH', pd.DataFrame() 
        
    # Use native float for faster calculations in the loop
    portfolio_value = float(INITIAL_INVESTMENT)
    shares = 0.0
    current_ticker = 'CASH'
    trade_history = [] 
    
    trade_history.append({'Date': sim_df.index.min().date(), 'Action': "START", 'Asset': 'CASH', 'Price': 0.0, 'Portfolio Value': portfolio_value})

    # B&H Setup
    qqq_start_price = sim_df.iloc[0][f'{TICKER}_close']
    tqqq_start_price = sim_df.iloc[0][f'{LEVERAGED_TICKER}_close']
    initial_float = float(INITIAL_INVESTMENT)
    sim_df['BH_QQQ_Value'] = initial_float * (sim_df[f'{TICKER}_close'] / qqq_start_price)
    sim_df['BH_TQQQ_Value'] = initial_float * (sim_df[f'{LEVERAGED_TICKER}_close'] / tqqq_start_price)
    
    for i in range(len(sim_df)):
        current_day = sim_df.iloc[i]
        trade_ticker = current_day['Trade_Ticker'] 
        current_day_prices = {t: current_day[f'{t}_close'] for t in [TICKER, LEVERAGED_TICKER, INVERSE_TICKER]}
        
        if trade_ticker != current_ticker:
            # SELL
            if current_ticker != 'CASH':
                sell_price = current_day_prices.get(current_ticker, 0.0)
                realized_cash = shares * sell_price if sell_price > 0 else 0.0
                portfolio_value = realized_cash
                trade_history.append({'Date': current_day.name.date(), 'Action': f"SELL {current_ticker}", 'Asset': current_ticker, 'Price': sell_price, 'Portfolio Value': realized_cash})
                shares = 0.0
            
            # BUY
            if trade_ticker != 'CASH':
                buy_price = current_day_prices.get(trade_ticker, 0.0)
                if buy_price > 0 and portfolio_value > 0:
                    shares = portfolio_value / buy_price
                else:
                    shares = 0.0
                portfolio_value = shares * buy_price if shares > 0 else portfolio_value
                trade_history.append({'Date': current_day.name.date(), 'Action': f"BUY {trade_ticker}", 'Asset': trade_ticker, 'Price': buy_price, 'Portfolio Value': portfolio_value})
                    
            current_ticker = trade_ticker

        # TRACKING
        if current_ticker != 'CASH' and shares > 0:
            current_price = current_day_prices.get(current_ticker, 0.0)
            portfolio_value = shares * current_price
        
        sim_df.loc[current_day.name, 'Portfolio_Value'] = portfolio_value 

    final_holding = current_ticker
    final_value = portfolio_value

    return final_value, sim_df['BH_QQQ_Value'].iloc[-1], sim_df['BH_TQQQ_Value'].iloc[-1], pd.DataFrame(trade_history), final_holding, sim_df

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
    """
    Processes the trade history to create Buy-Sell pairs with profit/loss
    AND identifies the last open Buy trade, if one exists.
    """
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
            if buy_trade: buy_trade = None # Reset if last trade was a Buy and current is a Buy (should not happen in this sim)
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
                
                is_profitable = profit_loss >= 0
                p_l_category = ("Profit" if is_profitable else "Loss").strip() 
                sort_key = 1 if is_profitable else 0
                
                trade_pairs.append({
                    'buy_date': buy_trade['buy_date'],
                    'sell_date': row['Date_dt'],
                    'buy_qqq_price': buy_trade['buy_qqq_price'],
                    'sell_qqq_price': row['QQQ_Price'],
                    'profit_loss': profit_loss,
                    'is_profitable_str': p_l_category, 
                    'sort_key': sort_key,
                    'asset': buy_trade['asset']
                })
                buy_trade = None
                
    # --- Check for Open Trade ---
    open_trade = None
    if buy_trade:
        # If buy_trade is not None after the loop, it's an open position
        last_date = full_data.index.max()
        # Use the price of the actual asset held (TQQQ/SQQQ) for the P/L calculation's basis
        asset_close_col = f'{buy_trade["asset"]}_close'
        current_asset_price = full_data[asset_close_col].loc[last_date]
        
        # We need the asset price at the time of the BUY to determine the P/L
        buy_asset_price = full_data[asset_close_col].loc[buy_trade['buy_date']]
        
        # Calculate P/L ratio to determine color
        if buy_asset_price > 0:
            current_profit_loss = current_asset_price - buy_asset_price
            is_profitable = current_profit_loss >= 0
            p_l_category = ("Profit" if is_profitable else "Loss").strip() 
        else:
            p_l_category = "Loss"

        # The chart plots QQQ price, so we use the corresponding QQQ prices for plotting data points
        qqq_buy_price = buy_trade['buy_qqq_price']
        qqq_current_price = full_data[qqq_close_col].loc[last_date]

        open_trade = {
            'buy_date': buy_trade['buy_date'],
            'buy_qqq_price': qqq_buy_price,
            'current_date': last_date,
            'current_qqq_price': qqq_current_price,
            'asset': buy_trade['asset'],
            'is_profitable_str': p_l_category, # Profit/Loss status used for color
            'Trade_Type': 'Open Position' # Identifier for the detail channel
        }
                
    return trade_pairs, open_trade 

def plot_trade_signals(signals_df, trade_pairs, TICKER, backtest_start, ytd_start_date, open_trade=None):
    """
    Generates an Altair chart, including trade segments for completed trades
    and a dotted projection for the open trade, with independently toggleable indicators.
    """
    
    # --- Dynamic Chart Range Logic ---
    if backtest_start >= ytd_start_date:
        chart_start_date = ytd_start_date
        chart_title_suffix = " (YTD View)"
    else:
        chart_start_date = backtest_start
        chart_title_suffix = " (Full Backtest View)"
        
    plot_data = signals_df.copy()
    plot_data = plot_data[plot_data.index.date >= chart_start_date].copy()
    plot_data.ta.sma(length=SMA_SHORT_PERIOD, append=True)
    
    price_cols = ['close', f'SMA_{SMA_PERIOD}', f'SMA_{SMA_SHORT_PERIOD}', f'EMA_{EMA_PERIOD}'] 
    plot_data_long = plot_data.reset_index().rename(columns={'index': 'Date'})[['Date'] + price_cols].melt('Date', var_name='Metric', value_name='Price')

    # --- Trade Segment Data (Only drawn for the backtest period) ---
    trade_segments = []
    for i, trade in enumerate(trade_pairs):
        if trade['buy_date'].date() >= chart_start_date:
            p_l_category = trade['is_profitable_str']
            sort_key = trade['sort_key'] 
            trade_segments.append({'trade_id': i, 'Date': trade['buy_date'], 'Price': trade['buy_qqq_price'], 'P_L_Category': p_l_category, 'Signal': 'Buy', 'P_L': trade['profit_loss'], 'Sort_Key': sort_key}) 
            trade_segments.append({'trade_id': i, 'Date': trade['sell_date'], 'Price': trade['sell_qqq_price'], 'P_L_Category': p_l_category, 'Signal': 'Sell', 'P_L': trade['profit_loss'], 'Sort_Key': sort_key}) 
        
    df_segments = pd.DataFrame(trade_segments)
    
    # --- Altair Chart Composition ---
    
    # 1. Selection for Independent Indicator Toggling (by default, Altair points/legends are multi-select)
    # We use empty initialization to start with all selected.
    selection = alt.selection_point(fields=['Metric'], bind='legend')
    
    base = alt.Chart(plot_data_long).encode(x=alt.X('Date:T', title='Date')).properties(title=f'{TICKER} Price and Strategy Signals{chart_title_suffix}', height=500)
    
    # 2. Base Price Line
    price_line = base.mark_line(color='gray', opacity=0.7, size=0.5).encode(
        y=alt.Y('Price:Q', title=f'{TICKER} Price ($)'),
        color=alt.value('gray'), 
        tooltip=[alt.Tooltip('Price:Q', format='$.2f', title=f'{TICKER} Price')],
    ).transform_filter(alt.datum.Metric == 'close')
    
    # 3. Indicator Lines (TOGGLEABLE)
    indicator_lines = base.mark_line().encode(
        y=alt.Y('Price:Q'),
        # Color encoding binds the selection for the toggle
        color=alt.Color('Metric:N', 
            scale=alt.Scale(domain=[f'SMA_{SMA_PERIOD}', f'SMA_{SMA_SHORT_PERIOD}', f'EMA_{EMA_PERIOD}'], range=['orange', 'blue', 'purple']), 
            legend=alt.Legend(title="Indicator (Click to Toggle)")
        ), 
        strokeDash=alt.condition(alt.datum.Metric == f'SMA_{SMA_PERIOD}', alt.value([5, 5]), alt.value([2, 2])),
        # Opacity condition uses the selection filter
        opacity=alt.condition(selection, alt.value(1.0), alt.value(0.1)), 
        tooltip=[alt.Tooltip('Metric:N', title='Indicator'), alt.Tooltip('Price:Q', format='$.2f', title='Value')],
    ).add_params(selection).transform_filter((alt.datum.Metric != 'close'))

    # 4. Trade Segments (Completed Trades - Solid Line) 
    segment_lines = alt.Chart(df_segments).mark_line(size=3).encode(
        x=alt.X('Date:T'),
        y=alt.Y('Price:Q'),
        detail='trade_id:N',
        order=alt.Order('Sort_Key:Q'),
        color=alt.condition(
            alt.Predicate(field='P_L_Category', equal='Profit'),
            alt.value('#008000'), # Green
            alt.value('#d62728')  # Red
        ),
        tooltip=[
            alt.Tooltip('Date:T', title='Trade Date'),
            alt.Tooltip('P_L:Q', title='P/L', format='+.2f'),
            alt.Tooltip('P_L_Category:N', title='Outcome'),
        ]
    )

    # 5. Open Trade Projection (Dotted Line)
    open_line = alt.Chart(pd.DataFrame()).mark_line(size=2, strokeDash=[5, 5]).encode(
        x='Date:T',
        y='Price:Q'
    ).properties(title="Open Trade") 

    if open_trade and open_trade['asset'] != 'CASH':
        # Create data points for the open trade line using QQQ prices for plotting
        open_trade_data = pd.DataFrame([
            {'Date': open_trade['buy_date'], 'Price': open_trade['buy_qqq_price'], 'P_L_Category': open_trade['is_profitable_str'], 'Trade_Type': open_trade['Trade_Type']},
            {'Date': open_trade['current_date'], 'Price': open_trade['current_qqq_price'], 'P_L_Category': open_trade['is_profitable_str'], 'Trade_Type': open_trade['Trade_Type']}
        ])
        
        # Filter for the visible chart range
        open_trade_data = open_trade_data[open_trade_data['Date'].dt.date >= chart_start_date]
        
        if not open_trade_data.empty:
            open_line = alt.Chart(open_trade_data).mark_line(size=3, strokeDash=[5, 5]).encode(
                x=alt.X('Date:T'),
                y=alt.Y('Price:Q'),
                detail='Trade_Type:N',
                color=alt.condition(
                    alt.Predicate(field='P_L_Category', equal='Profit'),
                    alt.value('#008000'), # Green (Profit)
                    alt.value('#d62728')  # Red (Loss)
                ),
                tooltip=[
                    alt.Tooltip('Date:T', title='Start Date'),
                    alt.Tooltip('Price:Q', title='QQQ Price', format='$.2f'),
                    alt.Tooltip('P_L_Category:N', title='Current Status'),
                    alt.Tooltip('Trade_Type:N', title='Trade Type'),
                ]
            )

    
    return (price_line + indicator_lines + segment_lines + open_line).interactive()


# --- Streamlit Application (Remaining functions) ---

def run_analysis(backtest_start_date, target_signal_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER, current_mini_vasl_multiplier):
    """Encapsulates the entire backtest and rendering process."""
    with st.spinner("Fetching data and running backtest..."):
        
        # 1. Fetch Data (Cached)
        full_data = fetch_historical_data(target_signal_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER)
        
        if full_data.empty:
            st.error("Could not fetch necessary historical data. Check tickers or try a different date range.")
            return

        # 2. Get Live Price and Signal Indicators (Cached)
        live_price = fetch_live_price(TICKER)
        signal_price = live_price if live_price is not None else full_data['close'].iloc[-1].item()
        price_source = "LIVE PRICE" if live_price is not None else "HISTORICAL CLOSE"

        try:
            # Use cached function for signal indicators
            indicators = calculate_indicators(full_data, target_signal_date, signal_price)
            signal_results = generate_signal(indicators, LEVERAGED_TICKER, current_mini_vasl_multiplier)
        except ValueError as e:
            st.error(f"Error in signal generation: {e}")
            return

        # 3. Generate Historical Signals (Cached)
        signals_df = generate_all_historical_signals(full_data, LEVERAGED_TICKER, current_mini_vasl_multiplier)
        
        # 4. Run Backtest Simulation (Cached)
        start_of_tradable_data = signals_df.index.min().date() 
        backtest_start = max(backtest_start_date, start_of_tradable_data) 
        
        # Call the standalone, cached function
        final_value, bh_qqq, bh_tqqq, trade_history_df, final_holding, sim_df = run_simulation(
            signals_df, backtest_start, TICKER, LEVERAGED_TICKER, INVERSE_TICKER
        )

        # 5. Performance Metrics Calculation
        last_trade_day = signals_df.index.max().date() 
        days_held = (last_trade_day - backtest_start).days
        years_held = days_held / 365.25 if days_held > 0 else 1 
        initial_float = float(INITIAL_INVESTMENT)

        strategy_mdd = calculate_max_drawdown(sim_df['Portfolio_Value'])
        
        calc_cagr = lambda final, initial, years: ((final / initial) ** (1 / years) - 1) * 100 if final > 0 and initial > 0 and years > 0 else 0.0
        strategy_cagr = calc_cagr(final_value, initial_float, years_held)

        # 6. Calculate YTD start date for plotting
        ytd_start_date = get_default_ytd_start_date(target_signal_date)

    st.markdown("## 📊 Strategy Results & Signal")
    st.markdown(f"**Backtest Period:** `{backtest_start.strftime('%Y-%m-%d')}` to `{last_trade_day.strftime('%Y-%m-%d')}`")
    st.markdown("---")
    
    # --- COMBINED Action and Live Signal Section ---
    st.subheader("Live Trading Signal")
    
    # Row 1: Action (Largest and most important)
    col_action, col_ticker = st.columns([3, 1])
    
    col_action.markdown(f"## :rotating_light: **{signal_results['signal']}**")
    col_ticker.markdown(f"## **{signal_results['trade_ticker']}**")
    
    # Row 2: Details (Smaller metrics and conviction)
    col_price, col_conviction = st.columns([1, 1])
    
    col_price.metric("Price Used", f"${indicators['current_price']:.2f}", help=f"Source: {price_source}")
    col_conviction.caption(f"**Conviction:** {signal_results['conviction']}")

    st.markdown("---")

    # --- Performance Summary (Only show Strategy) ---
    st.markdown("## 💰 Backtest Performance Summary")
    
    summary_data = {
        'Metric': [f"Strategy ({LEVERAGED_TICKER})"], 
        'Final Value': [final_value],
        'Total Return': [(final_value/initial_float - 1) * 100], 
        'CAGR': [strategy_cagr],
        'Max Drawdown': [strategy_mdd] 
    }
    df_summary = pd.DataFrame(summary_data)
    
    st.dataframe(df_summary.style.format({
        'Final Value': '${:,.2f}', 
        'Total Return': '{:+.2f}%', 
        'CAGR': '{:+.2f}%', 
        'Max Drawdown': '{:,.2f}%'
    }), hide_index=True, width='stretch')

    # --- Interactive Plot ---
    
    # Logic for dynamically setting the chart description
    if backtest_start >= ytd_start_date:
        chart_description = f"**Interactive Price and Trade Signal Chart (YTD View)**"
        chart_caption = f"Chart shows YTD price with strategy indicators (toggleable by legend click) and completed trade segments overlaid for the backtest period (starting {backtest_start.strftime('%Y-%m-%d')}). Open positions are shown as a dotted line."
    else:
        chart_description = f"**Interactive Price and Trade Signal Chart (Full Backtest View)**"
        chart_caption = f"Chart shows the full backtest period price with strategy indicators (toggleable by legend click) and completed trade segments. Open positions are shown as a dotted line."
        
    st.markdown(f"## 📈 {chart_description}")
    
    if len(trade_history_df[trade_history_df['Action'].str.startswith('BUY')]) > 0:
        # Generate trade pairs and check for open trade
        trade_pairs, open_trade = analyze_trade_pairs(trade_history_df, full_data, TICKER)
        
        # Use the plotting function with the new open_trade argument
        st.altair_chart(plot_trade_signals(signals_df, trade_pairs, TICKER, backtest_start, ytd_start_date, open_trade), use_container_width=True)
        st.caption(chart_caption)
    else:
        # Handle case with no trades
        plot_data_for_display = signals_df.copy()
        if backtest_start >= ytd_start_date:
            plot_data_for_display = plot_data_for_display[plot_data_for_display.index.date >= ytd_start_date].copy()
            
        plot_data_for_display.ta.sma(length=SMA_SHORT_PERIOD, append=True)
        price_cols = ['close', f'SMA_{SMA_PERIOD}', f'SMA_{SMA_SHORT_PERIOD}', f'EMA_{EMA_PERIOD}'] 
        plot_data_long_no_trades = plot_data_for_display.reset_index().rename(columns={'index': 'Date'})[['Date'] + price_cols].melt('Date', var_name='Metric', value_name='Price')
        
        # Redefine base chart for no-trade scenario
        base = alt.Chart(plot_data_long_no_trades).encode(x=alt.X('Date:T', title='Date')).properties(title=f'{TICKER} Price and Strategy Signals (No Trades){chart_title_suffix}', height=500)
        
        # 1. Selection for Indicator Toggling (even in no-trade scenario)
        selection = alt.selection_point(fields=['Metric'], bind='legend')
        
        price_line = base.mark_line(color='gray', opacity=0.7, size=0.5).encode(
            y=alt.Y('Price:Q', title=f'{TICKER} Price ($)'),
             color=alt.value('gray'), 
             tooltip=[alt.Tooltip('Price:Q', format='$.2f', title=f'{TICKER} Price')],
        ).transform_filter(alt.datum.Metric == 'close')
        
        indicator_lines = base.mark_line().encode(
            y=alt.Y('Price:Q'),
            color=alt.Color('Metric:N', 
                scale=alt.Scale(domain=[f'SMA_{SMA_PERIOD}', f'SMA_{SMA_SHORT_PERIOD}', f'EMA_{EMA_PERIOD}'], range=['orange', 'blue', 'purple']), 
                legend=alt.Legend(title="Indicator (Click to Toggle)")
            ), 
            strokeDash=alt.condition(alt.datum.Metric == f'SMA_{SMA_PERIOD}', alt.value([5, 5]), alt.value([2, 2])),
            opacity=alt.condition(selection, alt.value(1.0), alt.value(0.1)),
            tooltip=[alt.Tooltip('Metric:N', title='Indicator'), alt.Tooltip('Price:Q', format='$.2f', title='Value')],
        ).add_params(selection).transform_filter((alt.datum.Metric != 'close'))


        st.altair_chart((price_line + indicator_lines).interactive(), use_container_width=True)
        st.warning("No trades were executed in the selected backtest period. Displaying price and indicators only.")

    # --- Detailed Trade History (OPTIMIZATION: Use Expander) ---
    with st.expander("📜 Detailed Trade History (Click to Expand)"):
        trade_history_display = trade_history_df.copy()
        trade_history_display['Date'] = trade_history_display['Date'].astype(str)
        
        st.dataframe(trade_history_display.style.format({
            'Price': '${:,.2f}', 
            'Portfolio Value': '${:,.2f}'
        }), hide_index=True, width='stretch')
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
        # Ensure the date inputs are used as cache keys
        backtest_start_date = st.date_input("Start Date", default_ytd_start, min_value=TQQQ_INCEPTION_DATE, key='start_date')
        target_signal_date = st.date_input("End Date / Signal Date", last_closed_day, key='end_date')
        
        # Strategy Customization
        st.subheader("Risk Control")
        # Ensure multiplier is used as a cache key
        current_mini_vasl_multiplier = st.number_input(
            f"Soft Stop Multiplier ({TICKER} Mini-VASL ATR)", 
            min_value=0.5, 
            max_value=3.0, 
            value=MINI_VASL_MULTIPLIER, 
            step=0.1,
            key='mini_vasl_mult'
        )
        st.caption(f"Hard Stop Multiplier: {ATR_MULTIPLIER}x ATR (Fixed)")
        
        st.markdown("---")
        if st.button("Re-Run Analysis", type="primary"):
            # A dummy action to force a rerun if needed
            st.session_state['rerun_trigger'] = st.session_state.get('rerun_trigger', 0) + 1
            st.rerun()
            
    # --- Auto-Run Logic ---
    run_analysis(backtest_start_date, target_signal_date, TICKER, LEVERAGED_TICKER, INVERSE_TICKER, current_mini_vasl_multiplier)


if __name__ == "__main__":
    main_app()
