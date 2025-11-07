import pandas as pd
import yfinance as yf
import pandas_ta as pta
from datetime import datetime, timedelta
import warnings
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pytz
from decimal import Decimal, getcontext

# Set precision for Decimal operations to high value (e.g., 50 places)
getcontext().prec = 50

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration (Constants) ---
TICKER = "QQQ"
EMA_PERIOD = 5
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
SMA_PERIOD = 200
# Convert INITIAL_INVESTMENT to Decimal
INITIAL_INVESTMENT = Decimal("10000.00")
LEVERAGED_TICKER = "TQQQ"
INVERSE_TICKER = "SQQQ"

# TQQQ Inception Date (February 9, 2010)
TQQQ_INCEPTION_DATE = datetime(2010, 2, 9).date()

# Initialize Session State for date/price overrides
if 'override_price' not in st.session_state:
    st.session_state['override_price'] = None
if 'override_enabled' not in st.session_state:
    st.session_state['override_enabled'] = False

# --- Timezone-Aware Date Function (FIX: Corrected to New York Timezone) ---

def get_most_recent_trading_day():
    """Determines the most recent CLOSED trading day using New York (ET) timezone."""
    new_york_tz = pytz.timezone('America/New_York') # FIX: Use New York for market time
    now_et = datetime.now(new_york_tz)
    market_close_hour = 16 # 4:00 PM ET
    market_close_minute = 0 # FIX: Market closes at 4:00 PM ET (not 4:30 PM)
    
    target_date = now_et.date()

    cutoff_time = now_et.replace(hour=market_close_hour, minute=market_close_minute, second=0, microsecond=0)
    
    # If before market close today, the target date is yesterday
    if now_et < cutoff_time:
        target_date -= timedelta(days=1)

    # Move back if target date is a weekend
    while target_date.weekday() > 4:
        target_date -= timedelta(days=1)
        
    return target_date

# --- Core Helper Function for Data Fetching and Cleaning ---

@st.cache_data(ttl=60*60*4) 
def fetch_historical_data(): 
    """
    Fetches historical data for QQQ, TQQQ, and SQQQ starting from TQQQ's inception date.
    """
    today_for_fetch = get_most_recent_trading_day()
    market_end_date = today_for_fetch + timedelta(days=1) 
    
    start_date = TQQQ_INCEPTION_DATE
    tickers = [TICKER, LEVERAGED_TICKER, INVERSE_TICKER]
    
    try:
        all_data = yf.download(
            tickers, 
            start=start_date, 
            end=market_end_date, 
            interval="1d", 
            progress=False,
            auto_adjust=False, 
            timeout=15 
        )
        
        if all_data.empty: return pd.DataFrame()

        df_combined = pd.DataFrame(index=all_data.index)
        
        for ticker in tickers:
            # Prioritize Adj Close, fallback to Close
            if ('Adj Close', ticker) in all_data.columns:
                df_combined[f'{ticker}_close'] = all_data['Adj Close'][ticker]
            elif ('Close', ticker) in all_data.columns:
                df_combined[f'{ticker}_close'] = all_data['Close'][ticker]
            
            # Get Open Price for execution
            if ('Open', ticker) in all_data.columns:
                 df_combined[f'{ticker}_open'] = all_data['Open'][ticker]
                
            # Get QQQ high/low/open for ATR/Charting
            if ticker == TICKER:
                for metric in ['High', 'Low', 'Volume', 'Open']: 
                    if (metric, ticker) in all_data.columns:
                        df_combined[metric.lower()] = all_data[metric][ticker] 

        # Map QQQ columns to generic names required by functions like calculate_indicators & create_chart
        if f'{TICKER}_close' in df_combined.columns:
            df_combined['close'] = df_combined[f'{TICKER}_close']
        
        if f'{TICKER}_open' in df_combined.columns:
            df_combined['open'] = df_combined[f'{TICKER}_open']
            
        required_cols = ['open', 'high', 'low', 'close', 
                         f'{LEVERAGED_TICKER}_close', f'{INVERSE_TICKER}_close',
                         f'{TICKER}_open', f'{LEVERAGED_TICKER}_open', f'{INVERSE_TICKER}_open']
                         
        df_combined.dropna(subset=required_cols, inplace=True)
        
        return df_combined 

    except Exception as e:
        st.error(f"FATAL ERROR during data download: {e}")
        return pd.DataFrame()

# --- Indicator Calculations ---

def calculate_true_range_and_atr(df, atr_period):
    high_minus_low = df['high'] - df['low']
    high_minus_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_minus_prev_close = np.abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.DataFrame({'hl': high_minus_low, 'hpc': high_minus_prev_close, 'lpc': low_minus_prev_close}).max(axis=1)
    atr_series = true_range.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean()
    
    return atr_series 

def calculate_indicators(data_daily, target_date, current_price):
    """Calculates all indicators using data only up to target_date (Close Price)."""
    # Filter up to and including the target date
    df = data_daily[data_daily.index.date <= target_date].copy() 
    
    if df.empty:
        raise ValueError("No data available for indicator calculation on or before target date.")
        
    df.ta.sma(length=SMA_PERIOD, append=True)
    df.ta.ema(length=EMA_PERIOD, append=True)
    
    sma_col = f'SMA_{SMA_PERIOD}'
    
    if sma_col not in df.columns or df[sma_col].isnull().all():
        raise ValueError(f"Indicator calculation failed: Insufficient data (need {SMA_PERIOD} days) to calculate 200-Day SMA for {target_date}.")
    
    # Get the last valid indicator values (based on data up to target_date's close)
    current_sma_200 = df[sma_col].iloc[-1]
    latest_ema_5 = df[f'EMA_{EMA_PERIOD}'].iloc[-1]
    df['ATR'] = calculate_true_range_and_atr(df, ATR_PERIOD)
    latest_atr = df['ATR'].ffill().iloc[-1]

    # Clean up numpy types for consistent metric display
    current_sma_200 = current_sma_200.item() if hasattr(current_sma_200, 'item') else current_sma_200
    latest_ema_5 = latest_ema_5.item() if hasattr(latest_ema_5, 'item') else latest_ema_5
    latest_atr = latest_atr.item() if hasattr(latest_atr, 'item') else latest_atr
    
    if not all(np.isfinite([current_sma_200, latest_ema_5, latest_atr])):
        raise ValueError("Indicator calculation resulted in infinite or non-numeric values.")

    df['VASL_Level'] = df[f'EMA_{EMA_PERIOD}'] - (ATR_MULTIPLIER * df['ATR'])
    
    return {
        'current_price': current_price,
        'sma_200': current_sma_200,
        'ema_5': latest_ema_5,
        'atr': latest_atr
    }, df

def generate_signal(indicators, current_holding_ticker=None):
    """
    Applies the refined trading strategy logic, including the Mini-VASL exit.
    """
    price = indicators['current_price']
    sma_200 = indicators['sma_200']
    ema_5 = indicators['ema_5']
    atr = indicators['atr']

    vasl_trigger_level = ema_5 - (ATR_MULTIPLIER * atr)
    mini_vasl_exit_level = ema_5 - (0.5 * atr) # New Mini-VASL level
    
    # 1. PRIORITY 1: VASL (Volatility Stop-Loss)
    if price < vasl_trigger_level:
        final_signal = "**SELL TQQQ/SQQQ / CASH (VASL Triggered)**"
        trade_ticker = 'CASH'
        conviction_status = "VASL Triggered - Move to CASH"
    else:
        # 2. PRIORITY 2: DMA (Directional Conviction - with strict TQQQ entry)
        dma_bull_price = (price >= sma_200)
        
        if dma_bull_price:
            # Price is Bullish (Above 200-SMA)
            
            # **NEW RULE: Mini-VASL Exit (DMA Bull, but price drops near 5-EMA minus 0.5*ATR)**
            if current_holding_ticker == LEVERAGED_TICKER and price < mini_vasl_exit_level:
                 final_signal = "**SELL TQQQ / CASH (Mini-VASL Exit Triggered)**"
                 trade_ticker = 'CASH'
                 conviction_status = "Mini-VASL Triggered - Exiting TQQQ position"
            else:
                # Continue with original DMA Bull logic if Mini-VASL is NOT triggered or not holding TQQQ
                dma_bull_ema = (ema_5 >= sma_200) # EMA CONFIRMATION FILTER
                
                if dma_bull_ema:
                    # DMA: BULL (Price AND EMA confirm momentum) -> ENTRY/HOLD TQQQ
                    conviction_status = "DMA - Bull (LONG TQQQ confirmed by 5EMA)"
                    final_signal = "**BUY TQQQ**"
                    trade_ticker = LEVERAGED_TICKER
                else:
                    # DMA: Neutral (Price is bull, but 5EMA is lagging 200SMA)
                    
                    if current_holding_ticker == LEVERAGED_TICKER:
                        # RETAIN: If already in TQQQ, this condition (EMA lag) is NOT an exit rule.
                        conviction_status = "DMA - Hold TQQQ (Price Bullish, EMA lagging but not exit condition)"
                        final_signal = "**HOLD TQQQ**"
                        trade_ticker = LEVERAGED_TICKER
                    else:
                        # CASH: If not in TQQQ, the strict entry rule fails.
                        conviction_status = "DMA - Neutral (TQQQ Entry Filter Failed: 5EMA lagging)"
                        final_signal = "**CASH (TQQQ Entry Filter Failed)**"
                        trade_ticker = 'CASH'
        else: # dma_bull_price is False (Price < 200 SMA)
            # DMA: BEAR - Check PRIORITY 3
            
            # 3. PRIORITY 3: Inverse EMA Exit (Only applies if DMA is BEAR)
            if price > ema_5:
                # DMA Bear, but short-term bounce above 5-Day EMA
                conviction_status = "DMA - Bear, but Price > 5-Day EMA (Exit SQQQ)"
                final_signal = "**SELL SQQQ / CASH (Inverse Position Exit)**"
                trade_ticker = 'CASH'
            else:
                # DMA Bear, and no short-term strength (Price <= 5-Day EMA)
                conviction_status = "DMA - Bear (BUY SQQQ)"
                final_signal = "**BUY SQQQ**"
                trade_ticker = INVERSE_TICKER

    return final_signal, trade_ticker, conviction_status, vasl_trigger_level

# --- Backtesting Engine ---

class BacktestEngine:
    """Runs a backtest simulation for a given historical dataset."""
    
    def __init__(self, historical_data):
        self.df = historical_data.copy()
        
    def generate_historical_signals(self):
        """
        Generates the trading signal for every day in the history using lookahead-free logic.
        """
        # 1. Calculate all indicators on the existing data
        self.df.ta.sma(length=SMA_PERIOD, append=True)
        self.df.ta.ema(length=EMA_PERIOD, append=True)
        self.df['ATR'] = calculate_true_range_and_atr(self.df, ATR_PERIOD)
        
        # 2. Shift indicators and the QQQ Close price one day forward to simulate using the *prior* day's close data
        indicator_cols = [f'SMA_{SMA_PERIOD}', f'EMA_{EMA_PERIOD}', 'ATR', 'close']
        
        # DataFrame containing indicator values (and QQQ Close Price) from the previous day, aligned to the current day
        df_prev_day_data = self.df[indicator_cols].shift(1)
        df_prev_day_data.rename(columns={'close': 'prev_close'}, inplace=True)
        
        # 3. Initialize signal and tracking
        self.df['Trade_Ticker'] = 'CASH' # Default is CASH
        current_ticker_historical = 'CASH'
        
        # Start from the second day, as the first day's shifted indicators are NaN
        for index in self.df.index[1:]: 
            row = self.df.loc[index]
            prev_data = df_prev_day_data.loc[index]
            
            # If indicators are not yet calculated (i.e., less than SMA_PERIOD days)
            if pd.isna(prev_data[f'SMA_{SMA_PERIOD}']):
                current_ticker_historical = 'CASH'
                continue
                
            # Previous day's QQQ Close price (The trigger price - Lookahead Free)
            price = prev_data['prev_close'] 
            
            # Previous day's indicator values
            ema_5 = prev_data[f'EMA_{EMA_PERIOD}']
            sma_200 = prev_data[f'SMA_{SMA_PERIOD}']
            atr = prev_data['ATR']

            vasl_trigger_level = ema_5 - (ATR_MULTIPLIER * atr)
            
            # --- Signal Logic Based on Day N-1 Data ---
            
            # 1. PRIORITY 1: VASL
            if price < vasl_trigger_level:
                trade_ticker = 'CASH'
            else:
                # 2. PRIORITY 2: DMA (MODIFIED ENTRY)
                dma_bull_price = (price >= sma_200)
                
                if dma_bull_price:
                    # --- START DMA BULL REGIME ---
                    
                    # **NEW RULE: Mini-VASL Exit (Only if holding TQQQ)**
                    mini_vasl_exit_level = ema_5 - (0.5 * atr)
                    if current_ticker_historical == LEVERAGED_TICKER and price < mini_vasl_exit_level:
                         trade_ticker = 'CASH'
                    else:
                        # Continue with original DMA Bull logic
                        dma_bull_ema = (ema_5 >= sma_200) 
                        
                        if dma_bull_ema:
                            # TQQQ ENTRY/HOLD
                            trade_ticker = LEVERAGED_TICKER
                        else:
                            # EMA LAGGING: If already in TQQQ, HOLD. Otherwise, CASH.
                            trade_ticker = LEVERAGED_TICKER if current_ticker_historical == LEVERAGED_TICKER else 'CASH'
                else: # dma_bull_price is False (Price < 200 SMA)
                    # DMA: BEAR - Check PRIORITY 3
                    
                    # 3. PRIORITY 3: Inverse EMA Exit
                    if price > ema_5:
                        trade_ticker = 'CASH' 
                    else:
                        trade_ticker = INVERSE_TICKER 
            
            self.df.loc[index, 'Trade_Ticker'] = trade_ticker
            current_ticker_historical = trade_ticker # Update holding for the next day's check

        self.df.dropna(subset=[f'SMA_{SMA_PERIOD}', 'Trade_Ticker'], inplace=True)
        return self.df
        
    def run_simulation(self, start_date):
        """
        Runs the $10k simulation and logs all trades. 
        Execution: Trades execute at the current day's Open price.
        """
        sim_df = self.df[self.df.index.date >= start_date].copy()
        if sim_df.empty: return float(INITIAL_INVESTMENT), 0, pd.DataFrame() 
            
        # Initialize with Decimal
        portfolio_value = INITIAL_INVESTMENT 
        shares = Decimal("0")
        current_ticker = 'CASH'
        trade_history = [] 

        for i in range(len(sim_df)):
            current_day = sim_df.iloc[i]
            trade_ticker = current_day['Trade_Ticker']
            
            # FIX: Ensure all prices are converted to Decimal at entry to simulation loop
            current_day_prices = {}
            for t in [TICKER, LEVERAGED_TICKER, INVERSE_TICKER]:
                # CLOSE price for P/L tracking (unrealized value)
                current_day_prices[f'{t}_close_dec'] = Decimal(str(current_day[f'{t}_close']))
                # OPEN price for execution
                current_day_prices[f'{t}_open_dec'] = Decimal(str(current_day[f'{t}_open']))
            
            # --- 1. HANDLE TRADE SIGNAL CHANGE ---
            if trade_ticker != current_ticker:
                
                # A. SELL (Exit Current Position)
                if current_ticker != 'CASH':
                    # Execute SELL at the current day's Open price
                    sell_price = current_day_prices.get(f'{current_ticker}_open_dec', Decimal("0"))
                    if sell_price > 0:
                         realized_cash = shares * sell_price
                    else:
                        realized_cash = Decimal("0")

                    # Log the SELL trade.
                    trade_history.append({
                        'Date': current_day.name.date(), 
                        'Action': f"SELL {current_ticker}", 
                        'Asset': current_ticker, 
                        'Price': float(sell_price), # Store as float for display
                        'Portfolio Value': float(realized_cash) # Store as float for display
                    })
                    
                    # Update portfolio state to CASH.
                    portfolio_value = realized_cash
                    shares = Decimal("0")
                
                # B. BUY (Enter New Position)
                if trade_ticker != 'CASH':
                    # Execute BUY at the current day's Open price
                    buy_price = current_day_prices.get(f'{trade_ticker}_open_dec', Decimal("0"))
                    
                    if buy_price > 0:
                        # Use the entire CASH amount (portfolio_value) to buy shares
                        shares = portfolio_value / buy_price
                    else:
                        shares = Decimal("0")
                        portfolio_value = Decimal("0")
                        
                    # Log the BUY trade.
                    trade_history.append({
                        'Date': current_day.name.date(), 
                        'Action': f"BUY {trade_ticker}", 
                        'Asset': trade_ticker, 
                        'Price': float(buy_price),
                        'Portfolio Value': float(portfolio_value)
                    })
                        
                current_ticker = trade_ticker

            # --- 2. TRACKING: UPDATE PORTFOLIO VALUE FOR THE CURRENT DAY'S CLOSE ---
            if current_ticker != 'CASH' and shares > 0:
                # Use CLOSE price for tracking P/L (unrealized value)
                current_price = current_day_prices.get(f'{current_ticker}_close_dec', Decimal("0"))
                # Update the portfolio value based on the current close price (unrealized P/L)
                portfolio_value = shares * current_price
            
            # Log the daily portfolio value
            sim_df.loc[current_day.name, 'Portfolio_Value'] = float(portfolio_value) 

        # --- Add final row to trade history for current holding value ---
        if current_ticker != 'CASH' and shares > 0:
            last_date = sim_df.index[-1].date()
            final_price = current_day_prices.get(f'{current_ticker}_close_dec', Decimal("0"))
            
            trade_history.append({
                'Date': last_date, 
                'Action': f"HOLDING VALUE", 
                'Asset': current_ticker, 
                'Price': float(final_price),
                'Portfolio Value': float(portfolio_value)
            })


        # Final B&H calculation (Use float for final comparison)
        qqq_close_col = f'{TICKER}_close'
        qqq_start_price = sim_df.iloc[0][qqq_close_col]
        qqq_end_price = sim_df.iloc[-1][qqq_close_col]
        buy_and_hold_qqq = float(INITIAL_INVESTMENT) * (qqq_end_price / qqq_start_price)

        return float(portfolio_value), buy_and_hold_qqq, pd.DataFrame(trade_history)

def run_backtests(full_data, target_date):
    """Defines timeframes and runs the backtesting engine."""
    
    backtester = BacktestEngine(full_data)
    signals_df = backtester.generate_historical_signals()
    
    if signals_df.empty:
        st.error("Backtest failed: Could not generate historical signals.")
        return [], pd.DataFrame() 

    last_signal_date = signals_df.index.max().date()
    # Use the earliest date where indicators are valid for 'Full History' backtest
    start_of_tradable_data = signals_df.index.min().date() 
    start_of_year = datetime(last_signal_date.year, 1, 1).date()
    ytd_start_date = max(start_of_year, start_of_tradable_data) 
    
    three_months_back = (last_signal_date - timedelta(days=90))
    one_week_back = (last_signal_date - timedelta(days=7))

    timeframes = [
        ("Full History", start_of_tradable_data), 
        ("YTD (Since First Valid Signal)", ytd_start_date),
        ("3 Months Back", three_months_back),
        ("1 Week Back", one_week_back),
        ("Signal Date to Today", target_date), 
    ]
    
    results = []
    trade_history_for_signal_date = pd.DataFrame() 
    
    for label, start_date in timeframes:
        if start_date > last_signal_date and label != "Signal Date to Today": continue
        
        # Determine the first valid trading day on or after the start_date
        relevant_dates = signals_df.index[signals_df.index.date >= start_date]
        first_trade_day = relevant_dates.min().date() if not relevant_dates.empty else None

        if first_trade_day is None:
            # FIX: If the start_date is not in the index but is <= the last signal date, 
            # we need to find the next available trading day.
            if start_date <= last_signal_date:
                # Find the first index date >= start_date
                try:
                    first_trade_day = signals_df.index[signals_df.index.date >= start_date].min().date()
                except:
                    continue # Skip if no data
            else:
                continue
            
        signal_row_index = signals_df.index[signals_df.index.date >= first_trade_day].min()
        initial_trade = signals_df.loc[signal_row_index, 'Trade_Ticker'] if not pd.isna(signal_row_index) else "N/A"
        
        final_value, buy_and_hold_qqq, trade_history_df = backtester.run_simulation(first_trade_day)

        if label == "Signal Date to Today":
            # --- FIX for KeyError: 'Action' when no trades are executed ---
            if not trade_history_df.empty and 'Action' in trade_history_df.columns:
                # Filter out the 'HOLDING VALUE' row for the main backtest table which only logs trades
                trade_history_for_signal_date = trade_history_df[trade_history_df['Action'] != 'HOLDING VALUE'].copy()
                # Then add the holding value row back for display purposes
                if trade_history_df['Action'].str.contains('HOLDING VALUE').any():
                     trade_history_for_signal_date = pd.concat([trade_history_for_signal_date, trade_history_df[trade_history_df['Action'] == 'HOLDING VALUE'].copy()], ignore_index=True)
            else:
                 trade_history_for_signal_date = pd.DataFrame() # Ensure empty if no trades were logged
            
        initial_float = float(INITIAL_INVESTMENT)
        profit_loss = final_value - initial_float
        bh_profit_loss = buy_and_hold_qqq - initial_float

        # --- CAGR CALCULATION ---
        end_date = last_signal_date # Use the date of the last signal for calculation
        
        # Calculate the number of years (using 365.25 for average days per year)
        days_held = (end_date - first_trade_day).days
        years_held = days_held / 365.25 if days_held > 0 else 1 
        
        # Strategy CAGR
        if final_value > 0 and initial_float > 0 and years_held > 0:
            strategy_cagr = ( (final_value / initial_float) ** (1 / years_held) - 1 ) * 100
        else:
            strategy_cagr = 0.0

        # B&H QQQ CAGR
        if buy_and_hold_qqq > 0 and initial_float > 0 and years_held > 0:
            bh_qqq_cagr = ( (buy_and_hold_qqq / initial_float) ** (1 / years_held) - 1 ) * 100
        else:
            bh_qqq_cagr = 0.0
            
        results.append({
            "Timeframe": label,
            "Start Date": first_trade_day.strftime('%Y-%m-%d'),
            "First Trade": initial_trade,
            "Strategy Value": final_value,
            "B&H QQQ Value": buy_and_hold_qqq,
            "P/L": profit_loss,
            "B&H P/L": bh_profit_loss,
            "Strategy CAGR": strategy_cagr, 
            "B&H CAGR": bh_qqq_cagr
        })
        
    return results, trade_history_for_signal_date

# --- Plotly Charting Function ---

def calculate_true_range_and_atr_for_chart(df, atr_period):
    """Re-implemented helper for chart logic, which uses float math."""
    high_minus_low = df['high'] - df['low']
    high_minus_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_minus_prev_close = np.abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.DataFrame({'hl': high_minus_low, 'hpc': high_minus_prev_close, 'lpc': low_minus_prev_close}).max(axis=1)
    atr_series = true_range.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean()
    
    return atr_series 

def create_chart(df, indicators):
    """Creates a Plotly candlestick chart with indicators and date markers."""
    
    df_plot = df.copy() 
    
    fig = go.Figure(data=[
        go.Candlestick(x=df_plot.index, open=df_plot['open'], high=df_plot['high'], low=df_plot['low'], close=df_plot['close'], name=TICKER)
    ])

    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[f'SMA_{SMA_PERIOD}'], mode='lines', name=f'{SMA_PERIOD}-Day SMA', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[f'EMA_{EMA_PERIOD}'], mode='lines', name=f'{EMA_PERIOD}-Day EMA', line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['VASL_Level'], mode='lines', name='VASL Level', line=dict(color='red', width=1, dash='dot')))

    # 1. Marker for the Price/Indicator on the Historical Signal Date
    signal_date = indicators['signal_date']
    signal_price = indicators['signal_price']
    
    signal_date_index = df_plot.index[df_plot.index.date == signal_date].max()

    if pd.notna(signal_date_index):
        # 1a. Add vertical line for the signal date (FIX: Added name to shape for legend)
        fig.add_shape(
            type="line",
            x0=signal_date_index,
            y0=df_plot['low'].min(),
            x1=signal_date_index,
            y1=df_plot['high'].max(),
            line=dict(color="yellow", width=2, dash="dash"),
            layer="below", 
            # Adding a trace with mode='markers' and setting hoverinfo='skip' is the standard Plotly way to add a legend entry for a shape.
        )
        # FIX: Add an invisible scatter trace for the vertical line legend
        fig.add_trace(go.Scatter(
            x=[signal_date_index], 
            y=[signal_price],
            mode='markers',
            marker=dict(size=0), # Invisible marker
            showlegend=True,
            name=f'Signal Date ({signal_date.strftime("%Y-%m-%d")})' # Legend entry
        ))
        
        # 1b. Add the marker
        fig.add_trace(go.Scatter(
            x=[signal_date_index], 
            y=[signal_price], 
            mode='markers', 
            marker=dict(symbol='star', size=12, color='yellow', line=dict(width=2, color='black')),
            name=f'Signal Price'
        ))

    # 2. Marker for the Latest Price (end of the chart)
    latest_price = indicators['final_price']
    last_date = df_plot.index[-1]
    
    fig.add_trace(go.Scatter(
        x=[last_date], y=[latest_price], 
        mode='markers', 
        marker=dict(symbol='circle', size=10, color='lime', line=dict(width=2, color='black')),
        name='Latest Close Price'
    ))

    fig.update_layout(
        title=f'{TICKER} - Trading Strategy Indicators (Last ~{len(df_plot)} Trading Days)',
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        height=600,
        template='plotly_dark'
    )
    
    return fig

# --- Streamlit Application Layout ---

def display_app():
    
    st.set_page_config(page_title="TQQQ/SQQQ Signal", layout="wide")
    st.title("📈 TQQQ/SQQQ Daily Signal Generator & Full History Backtester")
    st.markdown("Strategy priority: **1. VASL** $\implies$ **2. DMA** (TQQQ exit via **Mini-VASL**) $\implies$ **3. Inverse EMA Exit**.")
    st.markdown("---")

    # 1. Data Fetch 
    data_for_backtest = fetch_historical_data()

    if data_for_backtest.empty:
        st.error("FATAL ERROR: Signal calculation aborted due to insufficient or missing data.")
        st.stop()
        
    latest_available_date = data_for_backtest.index.max().date()
    first_available_date = data_for_backtest.index.min().date()
    
    min_tradeable_date_index = SMA_PERIOD 
    if len(data_for_backtest) > min_tradeable_date_index:
        min_tradeable_date = data_for_backtest.iloc[min_tradeable_date_index].name.date() 
    else:
        min_tradeable_date = latest_available_date

    # FIX: Holiday/Weekend Handling (Determine the actual latest trading day)
    target_date_initial = latest_available_date
    if target_date_initial not in data_for_backtest.index.date:
        # Fallback to the last available date in the index
        target_date_initial = data_for_backtest.index[-1].date() 
        st.sidebar.warning(f"Note: Using **{target_date_initial.strftime('%Y-%m-%d')}** as the latest trading day due to weekend/holiday.")

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("1. Signal Date & Override Price")
        
        target_date = st.date_input("Select Signal Date (Close Price)", value=target_date_initial, min_value=min_tradeable_date, max_value=target_date_initial)
        
        # FIX: Checkbox for override price
        override_enabled = st.checkbox("Enable QQQ Price Override", value=st.session_state['override_enabled'])
        st.session_state['override_enabled'] = override_enabled

        # FIX: number_input bug fixed by providing a safe default of 0.00
        override_value = 0.00
        if override_enabled:
             override_value = st.number_input("Override Signal Price (QQQ Close $)", value=float(data_for_backtest[f'{TICKER}_close'].iloc[-1]), min_value=0.01, format="%.2f", help="Manually test the strategy at a specific QQQ Close price level.")
             st.session_state['override_price'] = override_value if override_value > 0.01 else None
        else:
            st.session_state['override_price'] = None

        if st.button("Clear Data Cache & Rerun", help="Forces a fresh download of market data."):
            st.cache_data.clear()
            st.rerun()
            
        st.header("2. Strategy Parameters")
        st.metric("Ticker", TICKER)
        st.metric("SMA Period (DMA)", f"{SMA_PERIOD} days")
        st.metric("EMA Period (VASL/Exit)", f"{EMA_PERIOD} days")
        st.metric("ATR Period (VASL)", f"{ATR_PERIOD} days")
        st.metric("ATR Multiplier (VASL)", ATR_MULTIPLIER)
        st.metric("Backtest Capital", f"${float(INITIAL_INVESTMENT):,.2f}")
        # FIX: UX polish
        st.caption(f"Full Data start: **{first_available_date.strftime('%Y-%m-%d')}**")


    # FIX: UX polish - clearer info box
    st.info(f"Analysis running for **{target_date.strftime('%A, %B %d, %Y')}**'s closing data.")
    st.markdown("---")
    
    # 2. Determine Final Signal Price 
    final_signal_price = st.session_state['override_price']
    qqq_close_col_name = f'{TICKER}_close'
    
    if final_signal_price is None:
        try:
            # Get the last available close price on or before the target date
            data_for_signal_price = data_for_backtest[data_for_backtest.index.date <= target_date].copy()
            if data_for_signal_price.empty:
                 st.error(f"No market data available on or before {target_date.strftime('%Y-%m-%d')}.")
                 st.stop()
            # Use the actual close price of the last valid day
            final_signal_price = data_for_signal_price[qqq_close_col_name].iloc[-1].item()
        except Exception as e:
            st.error(f"FATAL ERROR: Could not find the Adjusted Close price for the target date. Error: {e}")
            st.stop()

    # 3. Calculate and Generate Signal
    try:
        # indicators are calculated using data UP TO target_date's close.
        indicators, data_with_indicators = calculate_indicators(data_for_backtest, target_date, final_signal_price)
        final_signal, trade_ticker, conviction_status, vasl_trigger_level = generate_signal(indicators) 
    except ValueError as e: 
        st.error(f"FATAL ERROR: {e}")
        st.stop()
    except Exception as e:
        st.error(f"FATAL ERROR during indicator calculation or signal generation: {e}")
        st.stop()

    # 4. Run Backtests
    backtest_results, trade_history_df = run_backtests(data_for_backtest, target_date)
    
    # ... (Display Signal and Metrics) ...
    st.header(f"Daily Signal: {target_date.strftime('%Y-%m-%d')}")
    
    if "BUY TQQQ" in final_signal: st.success(f"## {final_signal}")
    elif "SQQQ" in final_signal: st.warning(f"## {final_signal}")
    else: st.error(f"## {final_signal}")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Signal Price (QQQ Close)", f"${indicators['current_price']:.2f}")
    col2.metric("200-Day SMA", f"${indicators['sma_200']:.2f}")
    col3.metric("DMA Conviction", conviction_status.split('(')[0].strip()) 

    st.subheader("Volatility Stop-Loss (VASL) Details")
    col_vasl_level, col_vasl_status = st.columns(2)
    col_vasl_level.metric("5-Day EMA", f"${indicators['ema_5']:.2f}")
    col_vasl_level.metric("14-Day ATR", f"${indicators['atr']:.2f}")
    
    level_format = col_vasl_status.error if "VASL Triggered" in conviction_status or "Mini-VASL Triggered" in conviction_status else col_vasl_status.success
    level_format(f"**VASL Trigger Level:** ${vasl_trigger_level:.2f}")

    st.markdown("---")
    
    # --- 6. Display Results: INTERACTIVE CHART ---
    st.header("📈 Interactive Indicator Chart")
    
    # Fetch 400 days to ensure 200-day SMA is calculated for the visible 200-day range
    chart_data = data_for_backtest.iloc[-400:].copy() 
    
    if not chart_data.empty:
        # Recalculate indicators on the full visible range for accurate display
        chart_data.ta.sma(length=SMA_PERIOD, append=True)
        chart_data.ta.ema(length=EMA_PERIOD, append=True)
        # Use the float-based helper for the chart data
        chart_data['ATR'] = calculate_true_range_and_atr_for_chart(chart_data, ATR_PERIOD)
        chart_data['VASL_Level'] = chart_data[f'EMA_{EMA_PERIOD}'] - (ATR_MULTIPLIER * chart_data['ATR'])
        
        # Only display the last 200 trading days
        chart_data = chart_data.iloc[-200:].dropna(subset=[f'SMA_{SMA_PERIOD}', 'open']) 

        chart_indicators = {
            'signal_date': target_date,
            'signal_price': final_signal_price,
            'final_price': data_for_backtest['close'].iloc[-1] 
        }
        
        try:
            chart_fig = create_chart(chart_data, chart_indicators) 
            st.plotly_chart(chart_fig, width='stretch')
        except Exception as e:
            st.error(f"Could not generate chart. Error: {e}")
    else:
        st.warning("Insufficient data to display the chart.")

    st.markdown("---")
    
    # --- 7. Display Results: AGGREGATE BACKTESTING ---
    st.header("⏱️ Backtest Performance (vs. QQQ Buy & Hold)")
    st.markdown(f"**Simulation:** ${float(INITIAL_INVESTMENT):,.2f} initial investment traded based on **lookahead-free historical daily signals**, executed at the **next day's Open price**.")

    if backtest_results:
        df_results = pd.DataFrame(backtest_results)
        df_results['Strategy Value'] = df_results['Strategy Value'].map('${:,.2f}'.format)
        df_results['B&H QQQ Value'] = df_results['B&H QQQ Value'].map('${:,.2f}'.format)
        df_results['P/L'] = df_results['P/L'].map(lambda x: f"{'+' if x >= 0 else ''}${x:,.2f}")
        df_results['B&H P/L'] = df_results['B&H P/L'].map(lambda x: f"{'+' if x >= 0 else ''}${x:,.2f}")
        
        # Format the new CAGR columns
        df_results['Strategy CAGR'] = df_results['Strategy CAGR'].map(lambda x: f"{'+' if x >= 0 else ''}{x:,.2f}%")
        df_results['B&H CAGR'] = df_results['B&H CAGR'].map(lambda x: f"{'+' if x >= 0 else ''}{x:,.2f}%")
        
        df_results = df_results.rename(columns={"B&H QQQ Value": "B&H Value", "B&H P/L": "B&H P/L", "Initial Trade": "First Trade"})
        
        # Updated column order to include CAGR
        column_order = ["Timeframe", "Start Date", "First Trade", "Strategy Value", "Strategy CAGR", "B&H Value", "B&H CAGR"]
        df_results = df_results[column_order]
        st.dataframe(df_results, hide_index=True)

    st.markdown("---")
    
    # --- 8. Display Detailed Trade History ---
    st.header(f"📜 Detailed Trade History (From {target_date.strftime('%Y-%m-%d')} to Today)")
    st.caption("**Trades listed here are executed at the Open price of the Date shown.**")
    
    if not trade_history_df.empty:
        
        # Apply formatting
        trade_history_df['Price'] = trade_history_df['Price'].map('${:,.2f}'.format)
        trade_history_df['Portfolio Value'] = trade_history_df['Portfolio Value'].map('${:,.2f}'.format)
        
        # Display the dataframe without custom row styling
        st.dataframe(
            trade_history_df, 
            column_config={
                "Date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"), 
                "Action": st.column_config.Column("Action", help="Trade action", width="small"), 
                "Asset": st.column_config.Column("Asset", width="small")
            },
            hide_index=True
        )
            
        st.caption("This table logs every time the strategy transitions to a new position. The **final row** shows the unrealized value of the last asset held (calculated at the final day's close price).")
    else:
        st.info("No trades were executed during the selected 'Signal Date to Today' backtest period.")

if __name__ == "__main__":
    display_app()
