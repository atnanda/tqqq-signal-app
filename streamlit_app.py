import pandas as pd
import yfinance as yf
import pandas_ta as pta 
from datetime import datetime, timedelta
import warnings
import streamlit as st
import numpy as np
import plotly.graph_objects as go 
import pytz 

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

# --- Timezone-Aware Date Function ---

def get_most_recent_trading_day():
    """
    Determines the most recent CLOSED trading day based on the California 
    (Pacific Time) timezone and a 4:30 PM PT cutoff.
    """
    
    # 1. Define the California Time Zone
    california_tz = pytz.timezone('America/Los_Angeles')
    
    # 2. Get the current California time
    now_ca = datetime.now(california_tz)
    
    # 3. Define market close time (4:30 PM PT)
    market_close_hour = 16
    market_close_minute = 30
    
    target_date = now_ca.date()

    # Create a timezone-aware cutoff time for comparison
    cutoff_time = now_ca.replace(hour=market_close_hour, minute=market_close_minute, second=0, microsecond=0)
    
    if now_ca < cutoff_time:
        target_date -= timedelta(days=1)

    # 4. Handle Weekends: Wind back to the previous Friday if needed
    while target_date.weekday() > 4: # 5=Saturday, 6=Sunday
        target_date -= timedelta(days=1)
        
    return target_date

# --- Core Helper Function for Data Fetching and Cleaning ---

@st.cache_data(ttl=60*60*4) 
def fetch_historical_data(lookback_days=400):
    """
    Fetches historical data for QQQ, TQQQ, and SQQQ up to the 
    most recent trading day.
    """
    # Ensure data is fetched up to the most recent day for backtesting purposes
    today_for_fetch = get_most_recent_trading_day()
    market_end_date = today_for_fetch + timedelta(days=1) 
    start_date = today_for_fetch - timedelta(days=lookback_days)
    
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

        df_combined = pd.DataFrame(index=all_data.index)
        
        for ticker in tickers:
            close_col_name = f"{ticker}_close"
            if ('Adj Close', ticker) in all_data.columns:
                df_combined[close_col_name] = all_data['Adj Close'][ticker]
            elif ('Close', ticker) in all_data.columns:
                df_combined[close_col_name] = all_data['Close'][ticker]
                
            if ticker == TICKER:
                for metric in ['Open', 'High', 'Low', 'Volume']:
                    if (metric, ticker) in all_data.columns:
                        df_combined[metric.lower()] = all_data[metric][ticker]
        
        if f'{TICKER}_close' in df_combined.columns:
            df_combined['close'] = df_combined[f'{TICKER}_close']
            
        required_cols = ['high', 'low', 'close', f'{LEVERAGED_TICKER}_close', f'{INVERSE_TICKER}_close']
        df_combined.dropna(subset=required_cols, inplace=True)

        if df_combined.shape[0] < SMA_PERIOD:
            # This is a general error, but calculation itself will handle it better below
            st.warning(f"Warning: Data only has {df_combined.shape[0]} rows. May not be enough for a full {SMA_PERIOD}-Day SMA.")
            
        # Return the full dataset up to the most recent trading day
        return df_combined 

    except Exception as e:
        st.error(f"FATAL ERROR during data download: {e}")
        return pd.DataFrame()

# --- Manual ATR Calculation (Returns the full series) ---

def calculate_true_range_and_atr(df, atr_period):
    """Calculates True Range and Average True Range series using native Pandas/Numpy."""
    
    high_minus_low = df['high'] - df['low']
    high_minus_prev_close = np.abs(df['high'] - df['close'].shift(1))
    low_minus_prev_close = np.abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.DataFrame({'hl': high_minus_low, 
                               'hpc': high_minus_prev_close, 
                               'lpc': low_minus_prev_close}).max(axis=1)
    
    atr_series = true_range.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean()
    
    return atr_series 

# --- Calculation and Signal Functions (FIXED ERROR CHECKING) ---

def calculate_indicators(data_daily, target_date, current_price):
    """Calculates all indicators using data only up to target_date."""
    
    # Filter to only use data up to the target_date for indicator calculation
    df = data_daily[data_daily.index.date <= target_date].copy() 
    
    if df.empty:
        raise ValueError("No data available for indicator calculation on or before target date.")
        
    df.ta.sma(length=SMA_PERIOD, append=True)
    df.ta.ema(length=EMA_PERIOD, append=True)
    
    sma_col = f'SMA_{SMA_PERIOD}'
    ema_col = f'EMA_{EMA_PERIOD}'

    # Check for SMA calculation failure (the root cause of the previous error)
    if sma_col not in df.columns or df[sma_col].isnull().all():
        raise ValueError(f"Indicator calculation failed: Insufficient data (need {SMA_PERIOD} days) to calculate 200-Day SMA for {target_date}.")
    
    current_sma_200 = df[sma_col].iloc[-1]
    latest_ema_5 = df[ema_col].iloc[-1]

    df['ATR'] = calculate_true_range_and_atr(df, ATR_PERIOD)
    
    latest_atr = df['ATR'].ffill().iloc[-1]
    
    if pd.isna(latest_atr) or not np.isfinite(latest_atr):
        if len(df['ATR'].ffill()) >= 2:
            latest_atr = df['ATR'].ffill().iloc[-2]
        else:
            raise ValueError("ATR calculation failed to yield a finite value after two attempts.")

    current_sma_200 = current_sma_200.item() if hasattr(current_sma_200, 'item') else current_sma_200
    latest_ema_5 = latest_ema_5.item() if hasattr(latest_ema_5, 'item') else latest_ema_5

    if not all(np.isfinite([current_sma_200, latest_ema_5, latest_atr])):
        raise ValueError("Indicator calculation resulted in infinite or non-numeric values.")

    df['VASL_Level'] = df[ema_col] - (ATR_MULTIPLIER * df['ATR'])
    
    return {
        'current_price': current_price,
        'sma_200': current_sma_200,
        'ema_5': latest_ema_5,
        'atr': latest_atr
    }, df

def generate_signal(indicators):
    """Applies the defined trading strategy logic: VASL and DMA."""
    price = indicators['current_price']
    sma_200 = indicators['sma_200']
    ema_5 = indicators['ema_5']
    atr = indicators['atr']

    vasl_trigger_level = ema_5 - (ATR_MULTIPLIER * atr)
    
    if price < vasl_trigger_level:
        final_signal = "**SELL TQQQ / CASH (Exit)**"
        trade_ticker = 'CASH'
        conviction_status = "VASL Triggered - Move to CASH"
    else:
        dma_bull = (price >= sma_200)
        
        if dma_bull:
            conviction_status = "DMA - Bull (LONG TQQQ)"
            final_signal = "**BUY TQQQ**"
            trade_ticker = LEVERAGED_TICKER
        else:
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
        """Generates the trading signal and conviction status for every day in the history."""
        
        self.df.ta.sma(length=SMA_PERIOD, append=True)
        self.df.ta.ema(length=EMA_PERIOD, append=True)
        
        high_minus_low = self.df['high'] - self.df['low']
        high_minus_prev_close = np.abs(self.df['high'] - self.df['close'].shift(1))
        low_minus_prev_close = np.abs(self.df['low'] - self.df['close'].shift(1))
        true_range = pd.DataFrame({'hl': high_minus_low, 'hpc': high_minus_prev_close, 'lpc': low_minus_prev_close}).max(axis=1)
        self.df['ATR'] = true_range.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
        
        self.df.dropna(subset=[f'SMA_{SMA_PERIOD}', f'EMA_{EMA_PERIOD}', 'ATR'], inplace=True)

        if self.df.empty:
            return pd.DataFrame()

        signals = []
        for index, row in self.df.iterrows():
            ema_5 = row[f'EMA_{EMA_PERIOD}']
            sma_200 = row[f'SMA_{SMA_PERIOD}']
            atr = row['ATR']
            price = row['close']
            
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
        
        # Simulates from start_date to the end of the full dataset
        sim_df = self.df[self.df.index.date >= start_date].copy()
        
        if sim_df.empty:
            return INITIAL_INVESTMENT, 0
            
        initial_investment = INITIAL_INVESTMENT
        portfolio_value = initial_investment
        shares = 0
        current_ticker = 'CASH'

        for i in range(len(sim_df)):
            current_day = sim_df.iloc[i]
            trade_ticker = current_day['Trade_Ticker']
            
            if trade_ticker != current_ticker:
                if current_ticker != 'CASH':
                    sell_price_col = f'{current_ticker}_close'
                    portfolio_value = shares * current_day[sell_price_col] 
                    shares = 0
                
                if trade_ticker != 'CASH':
                    buy_price_col = f'{trade_ticker}_close'
                    if current_day[buy_price_col] > 0:
                        shares = portfolio_value / current_day[buy_price_col]
                        portfolio_value = shares * current_day[buy_price_col]
                    else:
                        shares = 0
                        portfolio_value = 0 
                
                current_ticker = trade_ticker

            if shares > 0:
                current_price_col = f'{current_ticker}_close'
                portfolio_value = shares * current_day[current_price_col]
            
            sim_df.loc[current_day.name, 'Portfolio_Value'] = portfolio_value 

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

    # Get the last date in the fully calculated signal dataset
    last_signal_date = signals_df.index.max().date()
    
    start_of_tradable_data = signals_df.index.min().date() 
    start_of_year = datetime(last_signal_date.year, 1, 1).date()
    ytd_start_date = max(start_of_year, start_of_tradable_data) 
    
    three_months_back = (last_signal_date - timedelta(days=90))
    one_week_back = (last_signal_date - timedelta(days=7))

    timeframes = [
        ("YTD (Since First Valid Signal)", ytd_start_date),
        ("3 Months Back", three_months_back),
        ("1 Week Back", one_week_back),
        # This runs from the user's selected date (target_date) to the end of the data (last_signal_date)
        ("Signal Date to Today", target_date), 
    ]
    
    results = []
    
    for label, start_date in timeframes:
        # For long-term backtests, ensure start date doesn't exceed the last available data day
        if start_date > last_signal_date and label != "Signal Date to Today":
            continue
            
        relevant_dates = signals_df.index[signals_df.index.date >= start_date]
        first_trade_day = relevant_dates.min().date() if not relevant_dates.empty else None

        if first_trade_day is None:
            if label == "Signal Date to Today" and start_date <= last_signal_date:
                # Use the target_date if it's within the data range but no trade day is found immediately
                first_trade_day = start_date 
            else:
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

# --- Plotly Charting Function ---

def create_chart(df, indicators):
    """Creates a Plotly candlestick chart with indicators (SMA, EMA, VASL)."""
    
    df_plot = df.iloc[-200:].copy() 
    
    fig = go.Figure(data=[
        go.Candlestick(
            x=df_plot.index,
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name=TICKER
        )
    ])

    fig.add_trace(go.Scatter(
        x=df_plot.index, 
        y=df_plot[f'SMA_{SMA_PERIOD}'], 
        mode='lines', 
        name=f'{SMA_PERIOD}-Day SMA',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df_plot.index, 
        y=df_plot[f'EMA_{EMA_PERIOD}'], 
        mode='lines', 
        name=f'{EMA_PERIOD}-Day EMA',
        line=dict(color='orange', width=1)
    ))

    fig.add_trace(go.Scatter(
        x=df_plot.index, 
        y=df_plot['VASL_Level'], 
        mode='lines', 
        name='VASL Level',
        line=dict(color='red', width=1, dash='dot')
    ))

    current_price = indicators['current_price']
    last_date = df_plot.index[-1]
    
    fig.add_trace(go.Scatter(
        x=[last_date], y=[current_price], 
        mode='markers', 
        marker=dict(symbol='circle', size=10, color='lime', line=dict(width=2, color='black')),
        name='Signal Price'
    ))

    fig.update_layout(
        title=f'{TICKER} - Trading Strategy Indicators (Last ~200 Days)',
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

        if st.button("Clear Data Cache & Rerun", help="Click this if the price or indicators look outdated. Forces a fresh download."):
            st.cache_data.clear()
            st.rerun()

        st.header("2. Strategy Parameters")
        st.metric("Ticker", TICKER)
        st.metric("SMA Period (DMA)", f"{SMA_PERIOD} days")
        st.metric("EMA Period (VASL)", f"{EMA_PERIOD} days")
        st.metric("ATR Period (VASL)", f"{ATR_PERIOD} days")
        st.metric("ATR Multiplier (VASL)", ATR_MULTIPLIER)
        st.metric("Backtest Capital", f"${INITIAL_INVESTMENT:,.2f}")


    # --- Main Logic ---

    st.info(f"Analysis running for market close data on **{target_date.strftime('%A, %B %d, %Y')}**.")
    st.markdown("---")
    
    # 1. Data Fetch - Now fetches up to the most recent trading day
    data_for_backtest = fetch_historical_data(lookback_days=400)

    if data_for_backtest.empty:
        st.error("FATAL ERROR: Signal calculation aborted due to insufficient or missing data.")
        st.stop()

    # 2. Determine Final Signal Price (using data only up to target_date)
    final_signal_price = st.session_state['override_price']
    qqq_close_col_name = f'{TICKER}_close'
    
    if final_signal_price is None:
        try:
            # Filter data to get the signal price only up to the target_date
            data_for_signal_price = data_for_backtest[data_for_backtest.index.date <= target_date].copy()
            # Use the close price of the *last day in the filtered set*
            final_signal_price = data_for_signal_price[qqq_close_col_name].iloc[-1].item()
        
        except Exception as e:
            st.error(f"FATAL ERROR: Could not find the Adjusted Close price for the target date. Error: {e}")
            st.stop()

    # 3. Calculate and Generate Signal
    try:
        # Pass the full data set to calculate_indicators (it filters internally)
        indicators, data_with_indicators = calculate_indicators(data_for_backtest, target_date, final_signal_price)
        final_signal, trade_ticker, conviction_status, vasl_trigger_level = generate_signal(indicators) 
    except ValueError as e: # Catch the specific ValueError for insufficient data
        st.error(f"FATAL ERROR: {e}")
        st.stop()
    except Exception as e:
        st.error(f"FATAL ERROR during indicator calculation or signal generation: {e}")
        st.stop()


    # 4. Run Backtests - Use the full data set for the simulation
    backtest_results = run_backtests(data_for_backtest, target_date)
    
    # --- 5. Display Results: CURRENT SIGNAL ---
    
    st.header(f"Daily Signal: {target_date.strftime('%Y-%m-%d')}")
    
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
    
    # --- 6. Display Results: INTERACTIVE CHART ---
    st.header("📈 Interactive Indicator Chart")
    
    try:
        # Pass the DataFrame that was returned by calculate_indicators (data_with_indicators)
        chart_fig = create_chart(data_with_indicators, indicators) 
        st.plotly_chart(chart_fig, width='stretch')
    except Exception as e:
        st.error(f"Could not generate chart. Error: {e}")

    st.markdown("---")
        
    # --- 7. Display Results: BACKTESTING ---
    st.header("⏱️ Backtest Performance (vs. QQQ Buy & Hold)")
    st.markdown(f"**Simulation:** ${INITIAL_INVESTMENT:,.2f} initial investment traded based on historical daily signals.")

    if backtest_results:
        df_results = pd.DataFrame(backtest_results)
        
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
