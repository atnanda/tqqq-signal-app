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
import re

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

# --- Timezone-Aware Date Functions ---

def get_calendar_today():
    """Returns today's date in the New York timezone."""
    ny_tz = pytz.timezone('America/New_York')
    return datetime.now(ny_tz).date()

def get_last_closed_trading_day(end_date):
    """
    Finds the last date where market data is fully closed, based on the end_date.
    If the market is still open or hasn't closed yet, it uses the previous day.
    """
    ny_tz = pytz.timezone('America/New_York')
    now_ny = datetime.now(ny_tz)

    # Check if the requested end_date is today
    if end_date == now_ny.date():
        # Check if the current time is after market close (4:00 PM ET)
        if now_ny.hour >= 16:
            # Market is closed, data for today is available
            return end_date
        else:
            # Market is still open or pre-market, use yesterday's close
            return end_date - timedelta(days=1)
    
    # If a historical date is selected, assume data is available
    return end_date

# --- Data Fetching ---

@st.cache_data
def fetch_historical_data():
    """
    Fetches QQQ, TQQQ, and SQQQ data, ensuring all required flat column names exist.
    """
    
    tickers = [TICKER, LEVERAGED_TICKER, INVERSE_TICKER]
    multi_index_data = yf.download(tickers, start=TQQQ_INCEPTION_DATE, actions=True)
    
    data = pd.DataFrame(index=multi_index_data.index)
    
    def get_close_col(ticker):
        if ('Adj Close', ticker) in multi_index_data.columns:
            return ('Adj Close', ticker)
        elif ('Close', ticker) in multi_index_data.columns:
            return ('Close', ticker)
        else:
            raise ValueError(f"Failed to find 'Close' or 'Adj Close' data for {ticker}.")

    # --- QQQ Columns ---
    data[f'{TICKER}_open'] = multi_index_data[('Open', TICKER)]
    data[f'{TICKER}_high'] = multi_index_data[('High', TICKER)]
    data[f'{TICKER}_low'] = multi_index_data[('Low', TICKER)]
    data[f'{TICKER}_close'] = multi_index_data[('Close', TICKER)]
    
    qqq_adj_col = get_close_col(TICKER)
    if qqq_adj_col == ('Close', TICKER):
         data[f'{TICKER}_adj_close'] = data[f'{TICKER}_close']
    else:
        data[f'{TICKER}_adj_close'] = multi_index_data[qqq_adj_col]

    # --- TQQQ & SQQQ Columns ---
    data[f'{LEVERAGED_TICKER}_open'] = multi_index_data[('Open', LEVERAGED_TICKER)]
    data[f'{LEVERAGED_TICKER}_close'] = multi_index_data[get_close_col(LEVERAGED_TICKER)]
    data[f'{INVERSE_TICKER}_open'] = multi_index_data[('Open', INVERSE_TICKER)]
    data[f'{INVERSE_TICKER}_close'] = multi_index_data[get_close_col(INVERSE_TICKER)]


    data = data.dropna()
    return data

# --- Indicator Calculation (Pre-calculation for Efficiency & Charting) ---

@st.cache_data
def calculate_all_indicators(data):
    """Calculates all required indicators for the entire dataset."""
    
    data_with_indicators = data.copy()

    # Calculate indicators on the full dataset
    data_with_indicators['SMA_200'] = pta.sma(data_with_indicators[f'{TICKER}_adj_close'], length=SMA_PERIOD)
    data_with_indicators['EMA_5'] = pta.ema(data_with_indicators[f'{TICKER}_adj_close'], length=EMA_PERIOD)
    
    # ATR needs High, Low, and Close
    atr = pta.atr(
        data_with_indicators[f'{TICKER}_high'],
        data_with_indicators[f'{TICKER}_low'],
        data_with_indicators[f'{TICKER}_close'],
        length=ATR_PERIOD
    )
    data_with_indicators['ATR_14'] = atr
    
    # Calculate Stop-Loss Levels
    data_with_indicators['VASL_LEVEL'] = data_with_indicators['EMA_5'] - (data_with_indicators['ATR_14'] * ATR_MULTIPLIER)
    data_with_indicators['MINI_VASL_LEVEL'] = data_with_indicators['EMA_5'] - (data_with_indicators['ATR_14'] * 0.5)

    # Drop the rows that don't have enough history for indicators (like the first 200 days for SMA)
    return data_with_indicators.dropna(subset=['SMA_200'])


# --- Signal Logic ---

def generate_signal(indicators, current_holding_ticker=None):
    """
    Generates the trade signal (TQQQ, SQQQ, or CASH) based on DMA, VASL, and Mini-VASL logic.
    """
    
    # Check for necessary indicators
    if indicators is None or pd.isna(indicators['SMA_200']):
        return "N/A"

    # Get the QQQ Close Price
    qqq_close = indicators[f'{TICKER}_adj_close']
    
    # Get Indicator Values
    sma_200 = indicators['SMA_200']
    ema_5 = indicators['EMA_5']
    vasl_level = indicators['VASL_LEVEL']
    mini_vasl_level = indicators['MINI_VASL_LEVEL']
    
    # --- 1. VASL Exit (Absolute Stop-Loss) ---
    if qqq_close < vasl_level:
        return "CASH (VASL)"

    # --- Trend Status ---
    is_dma_bull_regime = qqq_close >= sma_200
    
    # --- 2. Mini-VASL Exit (Stricter Exit in a Bull Market) ---
    if is_dma_bull_regime and qqq_close < mini_vasl_level:
        return "CASH (Mini-VASL)"

    # --- 3. DMA Entry/Hold (Primary Logic) ---
    if is_dma_bull_regime:
        if ema_5 >= sma_200:
            return LEVERAGED_TICKER # TQQQ Entry/Hold (High Conviction)
        
        elif current_holding_ticker == LEVERAGED_TICKER:
            return LEVERAGED_TICKER # TQQQ Retention
        
        else:
            return "CASH"

    # --- 4. DMA Bear Regime (Inverse Logic) ---
    else: # qqq_close < sma_200 (Bear Regime)
        if qqq_close <= ema_5:
            return INVERSE_TICKER # SQQQ Entry
            
        elif current_holding_ticker == INVERSE_TICKER:
            return "CASH" # Inverse Exit

        else:
            return "CASH"

# --- Risk Calculation Helper ---

def calculate_risk_metrics(value_series):
    """
    Calculates Max Drawdown (MDD) and Sharpe Ratio (annualized).
    """
    if value_series.empty or len(value_series) < 2:
        return 0.0, 0.0

    # 1. Maximum Drawdown (MDD)
    cumulative_max = value_series.cummax()
    if (cumulative_max <= 0).any():
         mdd = 0.0
    else:
        drawdown = (value_series / cumulative_max) - 1
        mdd = drawdown.min()

    # 2. Sharpe Ratio
    daily_returns = value_series.pct_change().dropna()
    
    if len(daily_returns) < 2:
        return float(mdd), 0.0

    trading_days_per_year = 252
    annualized_return = daily_returns.mean() * trading_days_per_year
    annualized_std_dev = daily_returns.std() * np.sqrt(trading_days_per_year)

    sharpe_ratio = 0.0
    if annualized_std_dev > 0:
        sharpe_ratio = annualized_return / annualized_std_dev
    
    return float(mdd), float(sharpe_ratio)


# --- Backtesting Engine ---

class BacktestEngine:
    """Handles the simulation of the trading strategy using pre-calculated indicators."""
    
    def __init__(self, data_with_indicators):
        # Data passed here already includes SMA_200, EMA_5, etc.
        self.data = data_with_indicators 
        self.qqq_close_col = f'{TICKER}_adj_close'
        self.tqqq_close_col = f'{LEVERAGED_TICKER}_close'
        self.sqqq_close_col = f'{INVERSE_TICKER}_close'
        
    def _execute_trade(self, current_asset, next_asset, date, open_prices):
        """Calculates portfolio value change and returns new asset/value."""
        
        exit_value = Decimal("0.0")
        trade_record = {'Date': date, 'Action': None, 'Asset': None, 'Price': 0.0, 'Portfolio Value': 0.0}
        
        # --- 1. Exit Current Position ---
        if current_asset != "CASH":
            exit_price = open_prices.get(current_asset, Decimal("0.0"))
            exit_value = self.portfolio_shares * exit_price
            
            trade_record.update({
                'Action': f"SELL {current_asset}",
                'Asset': current_asset,
                'Price': float(exit_price),
                'Portfolio Value': float(exit_value)
            })
            self.portfolio_shares = Decimal("0.0")
            self.portfolio_value = exit_value
            
        else:
            exit_value = self.portfolio_value

        # --- 2. Enter New Position ---
        if next_asset != "CASH":
            if next_asset != current_asset or current_asset == "CASH":
                
                entry_price = open_prices.get(next_asset, Decimal("0.0"))
                if entry_price > 0:
                    next_asset_shares = exit_value / entry_price
                    next_asset_value = next_asset_shares * entry_price
                    
                    entry_record = {
                        'Date': date, 'Action': f"BUY {next_asset}", 'Asset': next_asset,
                        'Price': float(entry_price), 'Portfolio Value': float(next_asset_value)
                    }

                    if trade_record['Action'] is not None:
                         self.trade_history.append(entry_record)
                    else:
                        trade_record.update(entry_record)

                    self.portfolio_shares = next_asset_shares
                    self.portfolio_value = next_asset_value
                    self.current_asset = next_asset
                    
                    if trade_record['Action'] is not None and trade_record not in self.trade_history:
                        self.trade_history.append(trade_record)
                else:
                    self.portfolio_value = exit_value
                    self.current_asset = "CASH"
        else:
            self.portfolio_value = exit_value
            self.current_asset = "CASH"
        
        if trade_record['Action'] is not None and trade_record not in self.trade_history:
            self.trade_history.append(trade_record)


    def run_simulation(self, start_date):
        """Runs the backtest simulation from start_date."""
        
        # Sim_data is the data with all indicators pre-calculated
        sim_data = self.data[self.data.index.date >= start_date].copy()
        
        if sim_data.empty or len(sim_data) < 2:
            return float(INITIAL_INVESTMENT), float(INITIAL_INVESTMENT), pd.DataFrame(), pd.DataFrame({'Portfolio_Value': [float(INITIAL_INVESTMENT)]})
            
        # Initial State
        self.portfolio_value = INITIAL_INVESTMENT
        self.portfolio_shares = Decimal("0.0")
        self.current_asset = "CASH" 
        self.trade_history = []
        
        # Simulation tracking
        sim_df = pd.DataFrame(index=sim_data.index)
        
        # Start from the second day to use the previous day's close for the signal
        for i in range(1, len(sim_data)):
            current_date = sim_data.index[i].date()
            
            # --- Signal Generation (Look-ahead Free) ---
            # Indicators are taken from the PREVIOUS day's close (sim_data.iloc[i-1])
            indicators = sim_data.iloc[i-1]
            
            if pd.isna(indicators['SMA_200']):
                sim_df.loc[sim_data.index[i], 'Portfolio_Value'] = float(self.portfolio_value)
                continue

            # Generate the target asset for the *current day's open*
            target_asset = generate_signal(indicators, self.current_asset)
            
            # --- Trade Execution (at Current Day's Open) ---
            open_prices = {
                LEVERAGED_TICKER: Decimal(sim_data.iloc[i][f'{LEVERAGED_TICKER}_open']),
                INVERSE_TICKER: Decimal(sim_data.iloc[i][f'{INVERSE_TICKER}_open']),
                "CASH": Decimal("1.0")
            }

            if target_asset != self.current_asset:
                self._execute_trade(self.current_asset, target_asset, current_date, open_prices)
                
            # --- Update Portfolio Value (at Current Day's Close) ---
            if self.current_asset != "CASH":
                close_price_col = f'{self.current_asset}_close'
                close_price = Decimal(sim_data.iloc[i][close_price_col])
                self.portfolio_value = self.portfolio_shares * close_price
            
            sim_df.loc[sim_data.index[i], 'Portfolio_Value'] = float(self.portfolio_value)
            
        # --- Buy & Hold QQQ Benchmark ---
        qqq_close_col = f'{TICKER}_adj_close'
        qqq_start_price = sim_data.iloc[0][qqq_close_col]
        qqq_final_price = sim_data.iloc[-1][qqq_close_col]
        buy_and_hold_qqq = float(INITIAL_INVESTMENT) * (qqq_final_price / qqq_start_price)
        
        # Finalization
        if self.current_asset != "CASH" and not sim_data.empty:
            final_close_price = Decimal(sim_data.iloc[-1][f'{self.current_asset}_close'])
            final_value = self.portfolio_shares * final_close_price
            self.trade_history.append({
                'Date': sim_data.index[-1].date(), 'Action': f"HOLDING VALUE",
                'Asset': self.current_asset, 'Price': float(final_close_price),
                'Portfolio Value': float(final_value)
            })
            self.portfolio_value = final_value
            
        return float(self.portfolio_value), buy_and_hold_qqq, pd.DataFrame(self.trade_history), sim_df


# --- Backtest Runner ---

def run_backtests(data_with_indicators, target_date):
    """Runs the backtest simulation across defined timeframes."""
    
    backtester = BacktestEngine(data_with_indicators)
    results = []
    
    # Use the last date available in the data for calculating durations
    last_signal_date = data_with_indicators.index.max().date()

    timeframes = {
        "Full History": TQQQ_INCEPTION_DATE,
        "YTD": datetime(last_signal_date.year, 1, 1).date(),
        "1 Year": last_signal_date - timedelta(days=365),
        "3 Months": last_signal_date - timedelta(days=90),
        "1 Week": last_signal_date - timedelta(days=7)
    }
    
    # Initialize trade history for the final table display
    trade_history_df = pd.DataFrame()
    
    for label, start_date in timeframes.items():
        
        first_trade_day = max(start_date, TQQQ_INCEPTION_DATE)
        
        # Ensure the start date is before the end date
        if first_trade_day > last_signal_date:
            continue

        final_value, buy_and_hold_qqq, current_trade_history_df, sim_df = backtester.run_simulation(first_trade_day)

        # --- QQQ B&H Value Series Calculation (for risk metrics) ---
        qqq_close_col = f'{TICKER}_adj_close'
        sim_data = data_with_indicators[data_with_indicators.index.date >= first_trade_day].copy()
        
        if not sim_data.empty and sim_data.iloc[0][qqq_close_col] != 0:
            qqq_start_price = sim_data.iloc[0][qqq_close_col]
            qqq_value_series = (sim_data[qqq_close_col] / qqq_start_price) * float(INITIAL_INVESTMENT)
        else:
            qqq_value_series = pd.Series([float(INITIAL_INVESTMENT)])
        
        # --- Risk Metrics Calculation ---
        if not sim_df.empty:
            strategy_mdd, strategy_sharpe = calculate_risk_metrics(sim_df['Portfolio_Value'])
            bh_mdd, bh_sharpe = calculate_risk_metrics(qqq_value_series)
        else:
            strategy_mdd, strategy_sharpe = 0.0, 0.0
            bh_mdd, bh_sharpe = 0.0, 0.0
        
        initial_float = float(INITIAL_INVESTMENT)
        profit_loss = final_value - initial_float
        bh_profit_loss = buy_and_hold_qqq - initial_float
        
        # --- CAGR CALCULATION ---
        # Determine the actual duration of the trade/holding period
        duration_days = (sim_data.index[-1].date() - first_trade_day).days if not sim_data.empty else 0
        duration_years = duration_days / 365.25
        
        strategy_cagr = 0.0
        bh_qqq_cagr = 0.0
        if duration_years > 0 and final_value > 0 and buy_and_hold_qqq > 0:
            # CAGR formula: (Final Value / Initial Value)^(1/Years) - 1
            strategy_cagr = (((final_value / initial_float) ** (1 / duration_years)) - 1) * 100
            bh_qqq_cagr = (((buy_and_hold_qqq / initial_float) ** (1 / duration_years)) - 1) * 100

        initial_trade = current_trade_history_df.iloc[0]['Asset'] if not current_trade_history_df.empty else "CASH"

        results.append({
            "Timeframe": label, "Start Date": first_trade_day.strftime('%Y-%m-%d'),
            "First Trade": initial_trade, "Strategy Value": final_value,
            "B&H QQQ Value": buy_and_hold_qqq, "P/L": profit_loss,
            "B&H P/L": bh_profit_loss, "Strategy CAGR": strategy_cagr, 
            "B&H CAGR": bh_qqq_cagr, "Strategy MDD": strategy_mdd,
            "B&H MDD": bh_mdd, "Strategy Sharpe": strategy_sharpe,
            "B&H Sharpe": bh_sharpe
        })
        
        # Ensure that the trade history for the longest/most relevant period is saved
        if label == "Full History":
             trade_history_df = current_trade_history_df


    return results, trade_history_df


# --- Charting ---

def create_chart(data_with_indicators, target_date, target_price=None):
    """Creates a Plotly candlestick chart for QQQ with indicators."""
    
    # Filter for the last 6 months for better chart visibility by default
    start_date = target_date - timedelta(days=180)
    data_filtered = data_with_indicators[data_with_indicators.index.date >= start_date].copy()
    data_filtered = data_filtered[data_filtered.index.date <= target_date].copy()
    
    if data_filtered.empty:
        return go.Figure()

    fig = go.Figure(data=[go.Candlestick(
        x=data_filtered.index,
        open=data_filtered[f'{TICKER}_open'],
        high=data_filtered[f'{TICKER}_high'],
        low=data_filtered[f'{TICKER}_low'],
        close=data_filtered[f'{TICKER}_close'],
        name='QQQ Candlestick'
    )])
    
    # Add 200-Day SMA 
    fig.add_trace(go.Scatter(
        x=data_filtered.index,
        y=data_filtered['SMA_200'],
        line=dict(color='blue', width=2),
        name=f'{SMA_PERIOD}-Day SMA'
    ))

    # Add 5-Day EMA
    fig.add_trace(go.Scatter(
        x=data_filtered.index,
        y=data_filtered['EMA_5'],
        line=dict(color='orange', width=1),
        name=f'{EMA_PERIOD}-Day EMA'
    ))

    # Add VASL Level (ATR * 2.0)
    fig.add_trace(go.Scatter(
        x=data_filtered.index,
        y=data_filtered['VASL_LEVEL'],
        line=dict(color='red', width=1, dash='dash'),
        name=f'VASL Stop (x{ATR_MULTIPLIER} ATR)'
    ))
    
    # If an override price is used, mark it on the chart
    if target_price is not None:
        fig.add_annotation(
            x=data_filtered.index[-1],
            y=target_price,
            text=f"Signal Price: ${target_price:,.2f}",
            showarrow=True,
            arrowhead=1,
            ax=-50,
            ay=-50,
            font=dict(size=12, color="green"),
            bgcolor="white"
        )
        
    fig.update_layout(
        title=f'QQQ Candlestick and Dual Moving Average ({target_date.strftime("%Y-%m-%d")})',
        xaxis_rangeslider_visible=False,
        yaxis_title='Price (USD)',
        height=600
    )
    
    return fig


# --- Streamlit App ---

def display_app():
    
    st.set_page_config(layout="wide")
    st.title("TQQQ/SQQQ Dual Moving Average (DMA) Signal")
    st.caption("Strategy based on QQQ Price, 200-Day SMA, 5-Day EMA, and Volatility-Adjusted Stop-Loss (VASL).")
    
    # --- 1. Sidebar for Inputs ---
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Date Picker
        st.subheader("Signal Date")
        today = get_calendar_today()
        default_date = get_last_closed_trading_day(today)
        
        selected_date = st.date_input(
            "Select the day to generate the signal for (based on its close price):",
            value=default_date,
            max_value=today,
            key="signal_date"
        )
        
        # Price Override Toggle
        st.subheader("Price Override (What-If)")
        st.session_state['override_enabled'] = st.toggle("Use a custom QQQ Price", value=False)
        
        if st.session_state['override_enabled']:
            st.session_state['override_price'] = st.number_input(
                "Enter QQQ Close Price (e.g., live price)",
                min_value=1.0,
                value=float(st.session_state.get('override_price') or 350.0),
                format="%.2f",
                key="price_override"
            )
        else:
            st.session_state['override_price'] = None
            
        st.markdown(f"***")
        st.caption(f"Initial Investment: ${float(INITIAL_INVESTMENT):,.2f}")
        st.caption(f"DMA: {EMA_PERIOD}-EMA vs {SMA_PERIOD}-SMA")
        st.caption(f"VASL: {ATR_PERIOD}-ATR x {ATR_MULTIPLIER:.1f}")


    # --- 2. Data Fetching & Indicator Pre-calculation ---
    try:
        full_data = fetch_historical_data()
        data_with_indicators = calculate_all_indicators(full_data)
        
        if data_with_indicators.empty:
            st.error("Not enough historical data (at least 200 days) to run the analysis.")
            return

    except Exception as e:
        st.error(f"Error fetching/processing data: {e}")
        return

    target_date = get_last_closed_trading_day(selected_date)
    
    # Extract the indicators for the target date's close
    daily_indicators_df = data_with_indicators[data_with_indicators.index.date == target_date]

    if not daily_indicators_df.empty:
        daily_indicators = daily_indicators_df.iloc[-1]
        final_price = daily_indicators[f'{TICKER}_adj_close']
    else:
        # Fallback to the latest available day if the target date is unavailable
        if not data_with_indicators.empty:
             daily_indicators = data_with_indicators.iloc[-1]
             final_price = daily_indicators[f'{TICKER}_adj_close']
             target_date = daily_indicators.name.date()
        else:
            st.error(f"No indicator data available for selected date: {target_date.strftime('%Y-%m-%d')}.")
            return


    # --- 3. Price Override Logic ---
    if st.session_state['override_enabled'] and st.session_state['override_price'] is not None:
        final_price = st.session_state['override_price']
        
        # Create a mutable copy for the override calculation
        indicators_override = daily_indicators.copy()
        indicators_override[f'{TICKER}_adj_close'] = final_price # Update the price
        
        # Recalculate levels based on the new price and the old ATR/EMA levels
        ema_5 = indicators_override['EMA_5']
        atr_14 = indicators_override['ATR_14']
        indicators_override['VASL_LEVEL'] = ema_5 - (atr_14 * ATR_MULTIPLIER)
        indicators_override['MINI_VASL_LEVEL'] = ema_5 - (atr_14 * 0.5)
        
        final_indicators = indicators_override
        
    else:
        final_indicators = daily_indicators
        
    # --- 4. Signal Generation ---
    final_signal = generate_signal(final_indicators)

    # --- 5. Display Signal ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("🎯 Daily Signal")
        if final_signal == LEVERAGED_TICKER:
            st.success(f"**{final_signal} (BUY)**", icon="🚀")
        elif final_signal == INVERSE_TICKER:
            st.warning(f"**{final_signal} (BUY)**", icon="📉")
        elif "CASH" in final_signal or final_signal == "N/A":
            st.info(f"**{final_signal}**", icon="💰")
        else:
            st.error(f"**{final_signal}**", icon="❓")
            
        st.markdown(f"**Signal Date (Close):** {target_date.strftime('%Y-%m-%d')}")
        st.markdown(f"**QQQ Price:** **${final_price:,.2f}**")
        if st.session_state['override_enabled']:
            st.markdown(f"*(Price is overridden)*")
            
        st.subheader("Key Indicator Levels")
        st.metric(f"{SMA_PERIOD}-Day SMA", f"${final_indicators['SMA_200']:,.2f}")
        st.metric(f"{EMA_PERIOD}-Day EMA", f"${final_indicators['EMA_5']:,.2f}")
        # Delta shows distance to the stop-loss level
        st.metric("VASL Stop (x2.0 ATR)", f"${final_indicators['VASL_LEVEL']:,.2f}", delta=f"{final_price - final_indicators['VASL_LEVEL']:,.2f}", delta_color="inverse")
        st.metric("Mini-VASL Exit (x0.5 ATR)", f"${final_indicators['MINI_VASL_LEVEL']:,.2f}", delta=f"{final_price - final_indicators['MINI_VASL_LEVEL']:,.2f}", delta_color="inverse")
            
    # --- 6. Display Chart ---
    with col2:
        fig = create_chart(data_with_indicators, target_date, target_price=final_price)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # --- 7. Backtesting Results ---
    st.header("🧪 Strategy Backtest Performance (vs. QQQ Buy & Hold)")
    
    backtest_results, trade_history_df = run_backtests(data_with_indicators, target_date)
    
    if backtest_results:
        df_results = pd.DataFrame(backtest_results)
        
        # Format columns
        df_results['Strategy Value'] = df_results['Strategy Value'].map('${:,.2f}'.format)
        df_results['B&H QQQ Value'] = df_results['B&H QQQ Value'].map('${:,.2f}'.format)
        df_results['Strategy CAGR'] = df_results['Strategy CAGR'].map(lambda x: f"{'+' if x >= 0 else ''}{x:,.2f}%")
        df_results['B&H CAGR'] = df_results['B&H CAGR'].map(lambda x: f"{'+' if x >= 0 else ''}{x:,.2f}%")
        
        # New Formatting for Risk Metrics
        df_results['Strategy MDD'] = df_results['Strategy MDD'].map('{:,.2%}'.format)
        df_results['B&H MDD'] = df_results['B&H MDD'].map('{:,.2%}'.format)
        df_results['Strategy Sharpe'] = df_results['Strategy Sharpe'].map('{:,.2f}'.format)
        df_results['B&H Sharpe'] = df_results['B&H Sharpe'].map('{:,.2f}'.format)
        
        df_results = df_results.rename(columns={
            "B&H QQQ Value": "B&H Value", 
            "Initial Trade": "First Trade",
            "Strategy MDD": "Strategy Max Drawdown",
            "B&H MDD": "B&H Max Drawdown"
        })
        
        column_order = [
            "Timeframe", "Start Date", "First Trade", "Strategy Value", 
            "Strategy CAGR", "Strategy Sharpe", "Strategy Max Drawdown",
            "B&H Value", "B&H CAGR", "B&H Sharpe", "B&H Max Drawdown"
        ]
        df_results = df_results[column_order]
        st.dataframe(df_results, hide_index=True)

    st.markdown("---")
    
    # --- 8. Display Detailed Trade History ---
    
    if not trade_history_df.empty:
        # Determine the start date for the header dynamically
        start_date_for_header = trade_history_df['Date'].min() if 'Date' in trade_history_df.columns and not trade_history_df['Date'].empty else target_date
        
        st.header(f"📜 Detailed Trade History (From {start_date_for_header.strftime('%Y-%m-%d')} to Today)")
        st.caption("**Trades listed here are executed at the Open price of the Date shown.**")
        
        # Apply formatting
        trade_history_df['Price'] = trade_history_df['Price'].map('${:,.2f}'.format)
        trade_history_df['Portfolio Value'] = trade_history_df['Portfolio Value'].map('${:,.2f}'.format)
        
        # Use the original date format (YYYY-MM-DD)
        st.dataframe(
            trade_history_df, 
            column_config={
                "Date": st.column_config.DatetimeColumn("Date", format="%Y-%m-%d"), 
                "Action": st.column_config.Column("Action", help="Trade action", width="small"), 
                "Asset": st.column_config.Column("Asset", width="small")
            },
            hide_index=True
        )
            
        st.caption("This table logs every time the strategy transitions to a new position. The **final row** shows the unrealized value of the last asset held (calculated at the final day's close price).")
    else:
        st.info("No trades executed in the selected timeframe.")

# Run the app
if __name__ == "__main__":
    display_app()
