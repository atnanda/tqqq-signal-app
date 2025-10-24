import pandas as pd
import yfinance as yf
import pandas_ta as pta 
from datetime import datetime, timedelta
import warnings
import streamlit as st
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration (Constants) ---
TICKER = "QQQ"
EMA_PERIOD = 5
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0
SMA_PERIOD = 200
INITIAL_INVESTMENT = 10000.00
LEVERAGED_TICKER = "TQQQ"
INVERSE_TICKER = "SQQQ"

# Initialize Session State for date/price overrides
if 'override_price' not in st.session_state:
    st.session_state['override_price'] = None

# --- Core Helper Function for Data Fetching and Cleaning ---

@st.cache_data(ttl=60*60*4) 
def fetch_historical_data(target_date, lookback_days=400):
    """
    Fetches historical data for QQQ, TQQQ, and SQQQ, and ensures the required 
    prefixed columns ('QQQ_close', 'TQQQ_close', 'SQQQ_close') and 
    simple columns ('close', 'high', 'low') are present.
    """
    market_end_date = target_date + timedelta(days=1)
    start_date = target_date - timedelta(days=lookback_days)
    
    # st.info(f"Fetching historical data for {TICKER}, {LEVERAGED_TICKER}, {INVERSE_TICKER} up to {target_date.strftime('%Y-%m-%d')}...")
    
    try:
        tickers = [TICKER, LEVERAGED_TICKER, INVERSE_TICKER]
        
        all_data = yf.download(
            tickers, 
            start=start_date, 
            end=market_end_date, 
            interval="1d", 
            progress=False,
            auto_adjust=False, 
            timeout=15 
        )
        
        if all_data.empty:
            st.error("yfinance returned an empty dataset. Check tickers and date range.")
            return pd.DataFrame()

        # --- 1. Flatten MultiIndex and Consolidate Prices ---
        df_combined = pd.DataFrame(index=all_data.index)
        
        # Iterate through the required columns and tickers
        for ticker in tickers:
            # 1a. Prioritize Adjusted Close as the 'close' price
            close_col_name = f"{ticker}_close"
            if ('Adj Close', ticker) in all_data.columns:
                df_combined[close_col_name] = all_data['Adj Close'][ticker]
            elif ('Close', ticker) in all_data.columns:
                # Fallback to non-adjusted Close if Adj Close is missing (rare for QQQ, but safe)
                df_combined[close_col_name] = all_data['Close'][ticker]
            
            # 1b. Get OHLCV data for QQQ (needed for indicator calculation)
            if ticker == TICKER:
                for metric in ['Open', 'High', 'Low', 'Volume']:
                    if (metric, ticker) in all_data.columns:
                        df_combined[metric.lower()] = all_data[metric][ticker]
        
        # --- 2. Create Simple QQQ Columns for Indicators ---
        if f'{TICKER}_close' in df_combined.columns:
            df_combined['close'] = df_combined[f'{TICKER}_close']
            
        # --- 3. Final Validation and Cleanup ---
        
        required_cols = ['high', 'low', 'close', f'{LEVERAGED_TICKER}_close', f'{INVERSE_TICKER}_close']
        df_combined.dropna(subset=required_cols, inplace=True)

        missing_qqq_cols = [c for c in ['high', 'low', 'close'] if c not in df_combined.columns]
        if missing_qqq_cols:
             st.error(f"Data cleaning failed: Missing QQQ OHLC columns {missing_qqq_cols}.")
             return pd.DataFrame()

        if df_combined.shape[0] < SMA_PERIOD:
            st.error(f"FATAL ERROR: Insufficient clean data ({df_combined.shape[0]} rows) to calculate {SMA_PERIOD}-Day SMA.")
            return pd.DataFrame() 
            
        return df_combined[df_combined.index.date <= target_date]

    except Exception as e:
        st.error(f"FATAL ERROR during data download: {e}")
        return pd.DataFrame()

# --- Manual ATR Calculation ---

def calculate_true_range_and_atr(df, atr_period):
    """Calculates True Range and Average True Range using native Pandas/Numpy."""
    
    high_minus_low = df['high'] - df['low']
    high_minus_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_minus_prev_close = np.abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.DataFrame({'hl': high_minus_low, 
                               'hpc': high_minus_prev_close, 
                               'lpc': low_minus_prev_close}).max(axis=1)
    
    atr_series = true_range.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean()
    
    latest_atr = atr_series.ffill().iloc[-1]

    if pd.isna(latest_atr) or not np.isfinite(latest_atr):
         raise ValueError("Manual ATR calculation failed to yield a finite value.")
         
    return latest_atr

# --- Calculation and Signal Functions ---

def calculate_indicators(data_daily, current_price):
    """Calculates all indicators using pandas-ta for SMA/EMA and manual calculation for ATR."""
    
    df = data_daily.copy() 
    
    # 1. 200-Day SMA
    df.ta.sma(length=SMA_PERIOD, append=True)
    current_sma_200 = df[f'SMA_{SMA_PERIOD}'].iloc[-1]
    
    # 2. 5-Day EMA
    df.ta.ema(length=EMA_PERIOD, append=True)
    latest_ema_5 = df[f'EMA_{EMA_PERIOD}'].iloc[-1]

    # 3. 14-Day ATR
    latest_atr = calculate_true_range_and_atr(df, ATR_PERIOD)
    
    # Final check and conversion
    current_sma_200 = current_sma_200.item() if hasattr(current_sma_200, 'item') else current_sma_200
    latest_ema_5 = latest_ema_5.item() if hasattr(latest_ema_5, 'item') else latest_ema_5

    if not all(np.isfinite([current_sma_200, latest_ema_5, latest_atr])):
        raise ValueError("Indicator calculation resulted in infinite or non-numeric values.")

    return {
        'current_price': current_price,
        'sma_200': current_sma_200,
        'ema_5': latest_ema_5,
        'atr': latest_atr
    }


def generate_signal(indicators):
    """Applies the defined trading strategy logic: VASL and DMA."""
    price = indicators['current_price']
    sma_200 = indicators['sma_200']
    ema_5 = indicators['ema_5']
    atr = indicators['atr']

    # --- 1. Volatility Stop-Loss (VASL) ---
    vasl_trigger_level = ema_5 - (ATR_MULTIPLIER * atr)
    
    if price < vasl_trigger_level:
        final_signal = "**SELL TQQQ / CASH (Exit)**"
        trade_ticker = 'CASH'
        conviction_status = "VASL Triggered"
    else:
        # --- 2. Conviction Filter (DMA) ---
        dma_bull = (price >= sma_200)
        
        if dma_bull:
            conviction_status = "DMA - Bull (LONG TQQQ Default)"
            final_signal = "**BUY TQQQ**"
            trade_ticker = LEVERAGED_TICKER
        else:
            conviction_status = "DMA - Bear (CASH/SQQQ Default)"
            final_signal = "**BUY SQQQ**"
            trade_ticker = INVERSE_TICKER

    return final_signal, trade_ticker, conviction_status, vasl_trigger_level


# --- Backtesting Engine ---

class BacktestEngine:
    """Runs a backtest simulation for a given historical dataset."""
    
    def __init__(self, historical_data):
        self.df = historical_data.copy()
        
    def generate_historical_signals(self):
        """Generates the trading signal and conviction status for every day in the history."""
        
        # 1. Calculate SMA and EMA (once)
        self.df.ta.sma(length=SMA_PERIOD, append=True)
        self.df.ta.ema(length=EMA_PERIOD, append=True)
        
        # 2. Manually calculate ATR 
        high_minus_low = self.df['high'] - self.df['low']
        high_minus_prev_close = np.abs(self.df['high'] - self.df['close'].shift(1))
        low_minus_prev_close = np.abs(self.df['low'] - self.df['close'].shift(1))
        true_range = pd.DataFrame({'hl': high_minus_low, 'hpc': high_minus_prev_close, 'lpc': low_minus_prev_close}).max(axis=1)
        self.df['ATR'] = true_range.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
        
        # Drop initial NaNs created by indicators. This sets the first tradable day.
        self.df.dropna(subset=[f'SMA_{SMA_PERIOD}', f'EMA_{EMA_PERIOD}', 'ATR'], inplace=True)

        if self.df.empty:
            return pd.DataFrame()

        # 3. Apply the trading logic row by row
        signals = []
        for index, row in self.df.iterrows():
            ema_5 = row[f'EMA_{EMA_PERIOD}']
            sma_200 = row[f'SMA_{SMA_PERIOD}']
            atr = row['ATR']
            price = row['close'] # This is QQQ's adjusted close
            
            vasl_trigger_level = ema_5 - (ATR_MULTIPLIER * atr)
            
            if price < vasl_trigger_level:
                trade_ticker = 'CASH'
            else:
                dma_bull = (price >= sma_200)
                trade_ticker = LEVERAGED_TICKER if dma_bull else INVERSE_TICKER
            
            signals.append(trade_ticker)

        self.df['Trade_Ticker'] = signals
        return self.df
        
    def run_simulation(self, start_date):
        """Runs the $10k simulation from the start date to the end of the data."""
        
        # Filter by date (index.date is used for comparison with the start_date which is a date object)
        sim_df = self.df[self.df.index.date >= start_date].copy()
        
        if sim_df.empty:
            return INITIAL_INVESTMENT, 0
            
        initial_investment = INITIAL_INVESTMENT
        portfolio_value = initial_investment
        shares = 0
        current_ticker = 'CASH'

        # Loop through trading days
        for i in range(len(sim_df)):
            current_day = sim_df.iloc[i]
            trade_ticker = current_day['Trade_Ticker']
            
            # --- 1. Rebalance (Sell Old, Buy New) ---
            if trade_ticker != current_ticker:
                # Sell current position for CASH (using today's closing price)
                if current_ticker != 'CASH':
                    sell_price_col = f'{current_ticker}_close'
                    portfolio_value = shares * current_day[sell_price_col] 
                    shares = 0
                
                # Buy new position (using today's closing price)
                if trade_ticker != 'CASH':
                    buy_price_col = f'{trade_ticker}_close'
                    shares = portfolio_value / current_day[buy_price_col]
                    # Update value after purchase (shares are held)
                    portfolio_value = shares * current_day[buy_price_col] 
                
                current_ticker = trade_ticker

            # --- 2. Track Value ---
            # If still holding shares, update portfolio value based on today's close
            if shares > 0:
                current_price_col = f'{current_ticker}_close'
                portfolio_value = shares * current_day[current_price_col]
            
            sim_df.loc[current_day.name, 'Portfolio_Value'] = portfolio_value

        # Calculate final buy-and-hold value for comparison
        qqq_close_col = f'{TICKER}_close'
        qqq_start_price = sim_df.iloc[0][qqq_close_col]
        qqq_end_price = sim_df.iloc[-1][qqq_close_col]
        buy_and_hold_qqq = initial_investment * (qqq_end_price / qqq_start_price)

        return portfolio_value, buy_and_hold_qqq


def run_backtests(full_data, target_date):
    """Defines timeframes and runs the backtesting engine."""
    
    backtester = BacktestEngine(full_data)
    signals_df = backtester.generate_historical_signals()
    
    if signals_df.empty:
        st.error("Backtest failed: Could not generate historical signals.")
        return []

    # 'today' is a datetime.date object from the st.date_input
    today = target_date 
    
    # Get the actual earliest tradable day from the cleaned data
    start_of_tradable_data = signals_df.index.min().date() 
    
    # Set YTD start date to Jan 1
    start_of_year = datetime(today.year, 1, 1).date()
    
    # Use the later date between Jan 1 and the first tradable day
    ytd_start_date = max(start_of_year, start_of_tradable_data) 
    
    three_months_back = (today - timedelta(days=90))
    one_week_back = (today - timedelta(days=7))
    one_day_back = (today - timedelta(days=1))

    timeframes = [
        # OPTION 1 IMPLEMENTED: Use the mathematically correct start date 
        # but update the label to reflect the constraint.
        ("YTD (Since First Valid Signal)", ytd_start_date),
        ("3 Months Back", three_months_back),
        ("1 Week Back", one_week_back),
        ("1 Day Back", one_day_back),
    ]
    
    results = []
    
    for label, start_date in timeframes:
        if start_date >= today:
             results.append({"Timeframe": label, "Start Date": start_of_year.strftime('%Y-%m-%d'), "Strategy Value": INITIAL_INVESTMENT, "B&H QQQ Value": INITIAL_INVESTMENT, "P/L": 0.0, "B&H P/L": 0.0})
             continue
             
        # Use the calculated start_date (which for YTD is the first valid signal day)
        relevant_dates = signals_df.index[signals_df.index.date >= start_date]
        first_trade_day = relevant_dates.min().date() if not relevant_dates.empty else None

        if first_trade_day is None:
            st.warning(f"Skipping {label}: No trading data available on or after {start_date.strftime('%Y-%m-%d')}.")
            continue
            
        final_value, buy_and_hold_qqq = backtester.run_simulation(first_trade_day)
        
        profit_loss = final_value - INITIAL_INVESTMENT
        bh_profit_loss = buy_and_hold_qqq - INITIAL_INVESTMENT
        
        results.append({
            "Timeframe": label,
            "Start Date": first_trade_day.strftime('%Y-%m-%d'),
            "Strategy Value": final_value,
            "B&H QQQ Value": buy_and_hold_qqq,
            "P/L": profit_loss,
            "B&H P/L": bh_profit_loss
        })
        
    return results

# --- Streamlit Application Layout ---

def get_most_recent_trading_day():
    today = datetime.today().date()
    target_date = today
    while target_date.weekday() > 4: # 5=Saturday, 6=Sunday
        target_date -= timedelta(days=1)
    return target_date

def display_app():
    
    st.set_page_config(page_title="TQQQ/SQQQ Signal", layout="wide")
    st.title("📈 TQQQ/SQQQ Daily Signal Generator & Backtester")
    st.markdown("Strategy based on **200-Day SMA** (Conviction) and **5-Day EMA/14-Day ATR** (Volatility Stop-Loss).")
    st.markdown("---")

    # --- Sidebar for Inputs ---
    with st.sidebar:
        st.header("1. Target Date & Price")
        
        default_date = get_most_recent_trading_day()
        target_date = st.date_input(
            "Select Signal Date",
            value=default_date,
            max_value=default_date
        )
        
        if target_date.weekday() > 4:
            st.error(f"**{target_date.strftime('%Y-%m-%d')} is a weekend.** Please select a trading day.")
            st.stop()
        
        st.session_state['override_price'] = st.number_input(
            "Optional: Override Signal Price ($)",
            value=None,
            min_value=0.01,
            format="%.2f",
            help="Enter a price here to manually test the strategy at a specific level."
        )

        st.header("2. Strategy Parameters")
        st.metric("Ticker", TICKER)
        st.metric("SMA Period (DMA)", f"{SMA_PERIOD} days")
        st.metric("EMA Period (VASL)", f"{EMA_PERIOD} days")
        st.metric("ATR Period (VASL)", f"{ATR_PERIOD} days")
        st.metric("ATR Multiplier (VASL)", ATR_MULTIPLIER)
        st.metric("Backtest Capital", f"${INITIAL_INVESTMENT:,.2f}")


    # --- Main Logic ---

    # 1. Data Fetch
    data_for_indicators = fetch_historical_data(target_date, lookback_days=400)

    if data_for_indicators.empty:
        st.error("FATAL ERROR: Signal calculation aborted due to insufficient or missing data.")
        st.stop()

    # 2. Determine Final Signal Price
    final_signal_price = st.session_state['override_price']
    
    # --- HARDENED FIX FOR PRICE SELECTION ---
    qqq_close_col_name = f'{TICKER}_close'
    
    if final_signal_price is None:
        try:
            # 2a. Explicitly use .loc to find the QQQ Adjusted Close for the target date
            # Ensure the index is converted to date objects for comparison if necessary, 
            # but using the date index directly is usually safest.
            price_row = data_for_indicators.loc[data_for_indicators.index.date == target_date]
            
            if not price_row.empty:
                final_signal_price = price_row[qqq_close_col_name].iloc[0].item()
            else:
                 # Fallback to the last available price if the exact date is missing (e.g., if market closed early)
                 final_signal_price = data_for_indicators[qqq_close_col_name].iloc[-1].item()
        
        except KeyError:
             st.error(f"FATAL ERROR: Required price column '{qqq_close_col_name}' missing from data.")
             st.stop()
        except Exception:
             st.error(f"FATAL ERROR: Could not find the Adjusted Close price for {target_date.strftime('%Y-%m-%d')} in fetched data.")
             st.stop()

    # 3. Calculate and Generate Signal
    try:
        indicators = calculate_indicators(data_for_indicators, final_signal_price)
        final_signal, trade_ticker, conviction_status, vasl_trigger_level = generate_signal(indicators) 
    except Exception as e:
        st.error(f"FATAL ERROR during indicator calculation or signal generation: {e}")
        st.stop()

    # 4. Run Backtests
    backtest_results = run_backtests(data_for_indicators, target_date)
    
    # --- 5. Display Results: CURRENT SIGNAL ---
    
    st.header(f"Today's Signal based on {TICKER} Price at: {target_date.strftime('%Y-%m-%d')}")
    
    # Signal Box Display
    if "BUY TQQQ" in final_signal:
        st.success(f"## {final_signal}")
    elif "BUY SQQQ" in final_signal:
        st.warning(f"## {final_signal}")
    else: # SELL / CASH
        st.error(f"## {final_signal}")

    st.markdown("---")

    # Detail Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Signal Price (QQQ)", f"${indicators['current_price']:.2f}")
    col2.metric("200-Day SMA", f"${indicators['sma_200']:.2f}")
    col3.metric("DMA Conviction", conviction_status.split('(')[0].strip()) 

    st.subheader("Volatility Stop-Loss (VASL) Details")
    col_vasl_level, col_vasl_status = st.columns(2)

    col_vasl_level.metric("5-Day EMA", f"${indicators['ema_5']:.2f}")
    col_vasl_level.metric("14-Day ATR", f"${indicators['atr']:.2f}")

    if "Triggered" in conviction_status:
        col_vasl_status.error(f"**VASL Trigger Level:** ${vasl_trigger_level:.2f}") 
    else:
        col_vasl_status.success(f"**VASL Trigger Level:** ${vasl_trigger_level:.2f}")

    st.markdown("---")
    
    # --- 6. Display Results: BACKTESTING ---
    st.header("⏱️ Backtest Performance (vs. QQQ Buy & Hold)")
    st.markdown(f"**Simulation:** $10,000 initial investment traded based on historical daily signals.")

    if backtest_results:
        # Prepare DataFrame for display
        df_results = pd.DataFrame(backtest_results)
        
        # Format the numbers for readability
        df_results['Strategy Value'] = df_results['Strategy Value'].map('${:,.2f}'.format)
        df_results['B&H QQQ Value'] = df_results['B&H QQQ Value'].map('${:,.2f}'.format)
        df_results['P/L'] = df_results['P/L'].map(lambda x: f"{'+' if x >= 0 else ''}${x:,.2f}")
        df_results['B&H P/L'] = df_results['B&H P/L'].map(lambda x: f"{'+' if x >= 0 else ''}${x:,.2f}")
        
        df_results = df_results.rename(columns={
            "B&H QQQ Value": "B&H Value",
            "B&H P/L": "B&H P/L",
        })
        
        st.dataframe(df_results, 
            column_config={
                "Start Date": st.column_config.DatetimeColumn("Start Date", format="YYYY-MM-DD"),
                "P/L": st.column_config.Column("Strategy P/L", help="Strategy Profit/Loss", width="small"),
                "B&H P/L": st.column_config.Column("B&H P/L", help="Buy & Hold Profit/Loss", width="small")
            },
            hide_index=True)
    else:
        st.warning("No backtesting results generated for the selected date range.")


if __name__ == "__main__":
    display_app()
