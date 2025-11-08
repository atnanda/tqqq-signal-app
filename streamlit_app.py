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
import re # Added for pattern matching

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

# --- Data Handling ---

@st.cache_data
def fetch_historical_data():
    """Fetches QQQ, TQQQ, and SQQQ data from yfinance."""
    
    # Use adjusted close prices for accurate returns
    # Fetch all data for QQQ (needed for ATR)
    qqq_data = yf.download(TICKER, start=TQQQ_INCEPTION_DATE, actions=False)
    
    # Fetch only Adjusted Close for leveraged ETFs
    leveraged_data = yf.download(LEVERAGED_TICKER, start=TQQQ_INCEPTION_DATE, actions=False)['Adj Close'].rename(f'{LEVERAGED_TICKER}_close')
    inverse_data = yf.download(INVERSE_TICKER, start=TQQQ_INCEPTION_DATE, actions=False)['Adj Close'].rename(f'{INVERSE_TICKER}_close')

    # Prepare QQQ data
    qqq_data = qqq_data[['Open', 'High', 'Low', 'Close', 'Adj Close']]
    qqq_data.columns = [f'{TICKER}_open', f'{TICKER}_high', f'{TICKER}_low', f'{TICKER}_close', f'{TICKER}_adj_close']
    
    # Combine all data
    data = pd.concat([qqq_data, leveraged_data, inverse_data], axis=1).dropna()
    
    return data

def calculate_indicators(data, target_date):
    """Calculates all required indicators up to a specific target_date's close."""
    
    # Filter data up to and including the target date
    data_upto_target = data[data.index.date <= target_date].copy()
    
    # Check for sufficient data
    if len(data_upto_target) < SMA_PERIOD:
        return None

    # Calculate indicators
    
    # 1. Simple Moving Average (SMA)
    data_upto_target['SMA_200'] = pta.sma(data_upto_target[f'{TICKER}_adj_close'], length=SMA_PERIOD)
    
    # 2. Exponential Moving Average (EMA)
    data_upto_target['EMA_5'] = pta.ema(data_upto_target[f'{TICKER}_adj_close'], length=EMA_PERIOD)

    # 3. Average True Range (ATR)
    # ATR needs High, Low, and Close
    atr = pta.atr(
        data_upto_target[f'{TICKER}_high'],
        data_upto_target[f'{TICKER}_low'],
        data_upto_target[f'{TICKER}_close'],
        length=ATR_PERIOD
    )
    data_upto_target['ATR_14'] = atr
    
    # Calculate Stop-Loss Levels
    data_upto_target['VASL_LEVEL'] = data_upto_target['EMA_5'] - (data_upto_target['ATR_14'] * ATR_MULTIPLIER)
    # Mini-VASL Exit (NEW)
    data_upto_target['MINI_VASL_LEVEL'] = data_upto_target['EMA_5'] - (data_upto_target['ATR_14'] * 0.5)

    # Return the indicators for the target date's last available row
    return data_upto_target.iloc[-1]


# --- Signal Logic ---

def generate_signal(indicators, current_holding_ticker=None):
    """
    Generates the trade signal (TQQQ, SQQQ, or CASH) based on DMA, VASL, and Mini-VASL logic.
    
    Logic Priorities (Highest to Lowest):
    1. VASL Exit (Absolute Stop) -> CASH
    2. Mini-VASL Exit (DMA Bull Exit) -> CASH
    3. DMA Entry/Hold (TQQQ or SQQQ)
    4. DMA Retention/Inverse Exit (TQQQ Hold or SQQQ Exit)
    """
    
    # Check for necessary indicators (SMA_200 requires 200 days of data)
    if indicators is None or pd.isna(indicators['SMA_200']):
        return "N/A"

    # Get the QQQ Close Price (adjusted close is preferred but standard close is fine for intraday analysis)
    qqq_close = indicators[f'{TICKER}_adj_close']
    
    # Get Indicator Values
    sma_200 = indicators['SMA_200']
    ema_5 = indicators['EMA_5']
    vasl_level = indicators['VASL_LEVEL']
    mini_vasl_level = indicators['MINI_VASL_LEVEL']
    
    # --- 1. VASL Exit (Absolute Stop-Loss) ---
    # QQQ Price falls below the extreme volatility-adjusted stop.
    if qqq_close < vasl_level:
        return "CASH (VASL)"

    # --- Trend Status ---
    # DMA Bull Regime: Price is above the long-term trend
    is_dma_bull_regime = qqq_close >= sma_200
    
    # --- 2. Mini-VASL Exit (Stricter Exit in a Bull Market) ---
    # Only applies if we are in a DMA Bull Regime (Price >= SMA_200)
    # QQQ Price falls below the stricter volatility-adjusted stop.
    if is_dma_bull_regime and qqq_close < mini_vasl_level:
        return "CASH (Mini-VASL)"

    # --- 3. DMA Entry/Hold (Primary Logic) ---
    if is_dma_bull_regime:
        # P3: TQQQ Entry/Hold (High Conviction)
        if ema_5 >= sma_200:
            return LEVERAGED_TICKER # TQQQ
        
        # P4: TQQQ Retention
        # If we are in the DMA Bull Regime (Price >= SMA_200) but EMA_5 is lagging (EMA_5 < SMA_200)
        # We hold TQQQ if we are already in TQQQ, preventing whiplash.
        elif current_holding_ticker == LEVERAGED_TICKER:
            return LEVERAGED_TICKER # TQQQ (Hold)
        
        # All other cases in Bull Regime (e.g., in CASH, EMA_5 < SMA_200) lead to CASH
        else:
            return "CASH"

    # --- 4. DMA Bear Regime (Inverse Logic) ---
    else: # qqq_close < sma_200 (Bear Regime)
        # P6: SQQQ Entry
        if qqq_close <= ema_5:
            return INVERSE_TICKER # SQQQ
            
        # P5: Inverse Exit
        # If QQQ Price < SMA_200 AND QQQ Price > EMA_5
        # The bear trend is weakening (Price > short-term EMA), so we exit SQQQ to CASH.
        elif current_holding_ticker == INVERSE_TICKER:
            return "CASH"

        # All other cases in Bear Regime lead to CASH
        else:
            return "CASH"

# --- Risk Calculation Helper ---

def calculate_risk_metrics(value_series):
    if value_series.empty or len(value_series) < 2:
        return 0.0, 0.0 # MDD, Sharpe

    # 1. Maximum Drawdown (MDD)
    cumulative_max = value_series.cummax()
    # Ensure all values in cumulative_max are positive before division
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
        # Sharpe Ratio = (Annualized Return - Risk-Free Rate) / Annualized Volatility
        # Assuming Risk-Free Rate = 0
        sharpe_ratio = annualized_return / annualized_std_dev
    
    return float(mdd), float(sharpe_ratio)


# --- Backtesting Engine ---

class BacktestEngine:
    """Handles the simulation of the trading strategy."""
    
    def __init__(self, data):
        self.data = data
        self.qqq_close_col = f'{TICKER}_adj_close'
        self.tqqq_close_col = f'{LEVERAGED_TICKER}_close'
        self.sqqq_close_col = f'{INVERSE_TICKER}_close'
        
    def _execute_trade(self, current_asset, next_asset, date, qqq_price, open_prices):
        """Calculates portfolio value change and returns new asset/value."""
        
        # Default next values
        next_asset_value = Decimal("0.0")
        next_asset_shares = Decimal("0.0")
        
        # Trade History recording
        trade_record = {
            'Date': date,
            'Action': None,
            'Asset': None,
            'Price': 0.0,
            'Portfolio Value': 0.0
        }
        
        # --- 1. Exit Current Position ---
        exit_value = Decimal("0.0")
        if current_asset != "CASH":
            exit_price = open_prices.get(current_asset, Decimal("0.0"))
            exit_value = self.portfolio_shares * exit_price
            
            # Record exit trade
            trade_record.update({
                'Action': f"SELL {current_asset}",
                'Asset': current_asset,
                'Price': float(exit_price),
                'Portfolio Value': float(exit_value)
            })
            # Clear shares and update cash
            self.portfolio_shares = Decimal("0.0")
            self.portfolio_value = exit_value
            
        else:
            # If current_asset is CASH, exit_value is just the current portfolio value
            exit_value = self.portfolio_value

        # --- 2. Enter New Position ---
        if next_asset != "CASH":
            # Only trade if asset is changing or if we are entering from CASH
            if next_asset != current_asset or current_asset == "CASH":
                
                entry_price = open_prices.get(next_asset, Decimal("0.0"))
                if entry_price > 0:
                    next_asset_shares = exit_value / entry_price
                    next_asset_value = next_asset_shares * entry_price
                    
                    # Record entry trade
                    # Use a new record if an exit already occurred
                    if trade_record['Action'] is not None:
                         # Append a second record for the entry action
                        self.trade_history.append({
                            'Date': date,
                            'Action': f"BUY {next_asset}",
                            'Asset': next_asset,
                            'Price': float(entry_price),
                            'Portfolio Value': float(next_asset_value)
                        })
                    else:
                        # Otherwise, update the existing record
                        trade_record.update({
                            'Action': f"BUY {next_asset}",
                            'Asset': next_asset,
                            'Price': float(entry_price),
                            'Portfolio Value': float(next_asset_value)
                        })

                    # Update portfolio state
                    self.portfolio_shares = next_asset_shares
                    self.portfolio_value = next_asset_value
                    self.current_asset = next_asset
                    
                    # Only append the trade_record if it was an entry or a combined exit/entry (first part)
                    if trade_record['Action'] is not None and trade_record['Action'].startswith('SELL'):
                        # If a SELL occurred, the entry record was appended above, so we don't append the SELL record here.
                        # We use the final Portfolio Value of the day to update the simulation series.
                        pass
                    elif trade_record['Action'] is not None:
                        # Append only the BUY record if it was the only action
                         self.trade_history.append(trade_record)
                        
                else:
                    # Failsafe if price is zero (shouldn't happen with yfinance data)
                    self.portfolio_value = exit_value
                    self.current_asset = "CASH"
        else:
            # If next_asset is CASH, just update the asset state
            self.portfolio_value = exit_value
            self.current_asset = "CASH"
        
        # Append only the record of the exit trade if a trade occurred
        if trade_record['Action'] is not None and trade_record not in self.trade_history:
            self.trade_history.append(trade_record)


    def run_simulation(self, start_date):
        """Runs the backtest simulation from start_date."""
        
        # Filter data to the simulation period
        sim_data = self.data[self.data.index.date >= start_date].copy()
        
        # Check if we have enough data to run
        if sim_data.empty or len(sim_data) < 2:
            return float(INITIAL_INVESTMENT), float(INITIAL_INVESTMENT), pd.DataFrame(), pd.DataFrame({'Portfolio_Value': [float(INITIAL_INVESTMENT)]})
            
        # Initial State
        self.portfolio_value = INITIAL_INVESTMENT
        self.portfolio_shares = Decimal("0.0")
        self.current_asset = "CASH" # Start in CASH until a BUY signal is generated
        self.trade_history = []
        
        # Simulation tracking (for charting and risk metrics)
        sim_df = pd.DataFrame(index=sim_data.index)
        
        # --- 1. Calculate Signals and Execute Trades ---
        
        # We start the loop from the second day to ensure a previous day's close for the signal
        for i in range(1, len(sim_data)):
            current_date = sim_data.index[i].date()
            previous_date = sim_data.index[i-1].date()
            
            # --- Signal Generation (Look-ahead Free) ---
            # Signal is based on the previous day's close indicators
            indicators = calculate_indicators(self.data, previous_date)
            
            # Check for sufficient indicator data
            if indicators is None:
                # Can't generate signal, skip this day
                sim_df.loc[sim_data.index[i], 'Portfolio_Value'] = float(self.portfolio_value)
                continue

            # Generate the target asset for the *current day's open*
            target_asset = generate_signal(indicators, self.current_asset)
            
            # --- Trade Execution (at Current Day's Open) ---
            
            # Open prices for today (to execute trades)
            open_prices = {
                LEVERAGED_TICKER: Decimal(sim_data.iloc[i][f'{LEVERAGED_TICKER}_open']),
                INVERSE_TICKER: Decimal(sim_data.iloc[i][f'{INVERSE_TICKER}_open']),
                "CASH": Decimal("1.0")
            }
            
            current_open_price = open_prices.get(self.current_asset)

            # Check for a transition (or forced exit)
            if target_asset != self.current_asset:
                self._execute_trade(self.current_asset, target_asset, current_date, sim_data.iloc[i][self.qqq_close_col], open_prices)
                
            # --- 3. Update Portfolio Value (at Current Day's Close) ---
            
            # If holding a stock, update value with today's close price
            if self.current_asset != "CASH":
                close_price_col = f'{self.current_asset}_close'
                close_price = Decimal(sim_data.iloc[i][close_price_col])
                self.portfolio_value = self.portfolio_shares * close_price
            
            # Record daily portfolio value for sim_df
            sim_df.loc[sim_data.index[i], 'Portfolio_Value'] = float(self.portfolio_value)
            
        
        # --- 4. Buy & Hold QQQ Benchmark ---
        
        qqq_close_col = f'{TICKER}_adj_close'
        qqq_start_price = sim_data.iloc[0][qqq_close_col]
        qqq_final_price = sim_data.iloc[-1][qqq_close_col]
        
        buy_and_hold_qqq = float(INITIAL_INVESTMENT) * (qqq_final_price / qqq_start_price)
        
        # Final trade record if we are still holding an asset
        if self.current_asset != "CASH":
             # Use the last day's close price for the final (unrealized) trade value
            final_close_price = Decimal(sim_data.iloc[-1][f'{self.current_asset}_close'])
            final_value = self.portfolio_shares * final_close_price
            
            # Update the last trade record with the final market value
            self.trade_history.append({
                'Date': sim_data.index[-1].date(),
                'Action': f"HOLD {self.current_asset}",
                'Asset': self.current_asset,
                'Price': float(final_close_price),
                'Portfolio Value': float(final_value)
            })
            
            # Update the final portfolio value used for the simulation return
            self.portfolio_value = final_value
            
        return float(self.portfolio_value), buy_and_hold_qqq, pd.DataFrame(self.trade_history), sim_df


# --- Backtest Runner ---

def run_backtests(full_data, target_date):
    """Runs the backtest simulation across defined timeframes."""
    
    backtester = BacktestEngine(full_data)
    results = []
    
    # 1. Full History (Since TQQQ inception)
    # The backtester should automatically handle the initial data check
    
    # 2. Timeframes to test
    timeframes = {
        "Full History": TQQQ_INCEPTION_DATE,
        "YTD": datetime(target_date.year, 1, 1).date(),
        "1 Year": target_date - timedelta(days=365),
        "3 Months": target_date - timedelta(days=90),
        "1 Week": target_date - timedelta(days=7)
    }
    
    for label, start_date in timeframes.items():
        # Ensure start date is not after target date
        first_trade_day = max(start_date, TQQQ_INCEPTION_DATE)
        
        # Run simulation
        final_value, buy_and_hold_qqq, trade_history_df, sim_df = backtester.run_simulation(first_trade_day)

        # --- QQQ B&H Value Series Calculation (for risk metrics) ---
        qqq_close_col = f'{TICKER}_close'
        sim_data = full_data[full_data.index.date >= first_trade_day].copy()
        
        if not sim_data.empty:
            qqq_start_price = sim_data.iloc[0][qqq_close_col]
            # Create a B&H QQQ value series
            qqq_value_series = (sim_data[qqq_close_col] / qqq_start_price) * float(INITIAL_INVESTMENT)
        else:
            qqq_value_series = pd.Series([float(INITIAL_INVESTMENT)])
        
        # --- Risk Metrics Calculation ---
        
        # Strategy Metrics
        strategy_mdd, strategy_sharpe = calculate_risk_metrics(sim_df['Portfolio_Value'])

        # B&H Metrics
        bh_mdd, bh_sharpe = calculate_risk_metrics(qqq_value_series)
        
        
        initial_float = float(INITIAL_INVESTMENT)
        profit_loss = final_value - initial_float
        bh_profit_loss = buy_and_hold_qqq - initial_float

        # --- CAGR CALCULATION ---
        
        # Calculate the duration in years
        duration_days = (sim_data.index[-1].date() - first_trade_day).days
        duration_years = duration_days / 365.25
        
        # Calculate CAGR
        if duration_years > 0 and final_value > 0 and buy_and_hold_qqq > 0:
            strategy_cagr = (((final_value / initial_float) ** (1 / duration_years)) - 1) * 100
            bh_qqq_cagr = (((buy_and_hold_qqq / initial_float) ** (1 / duration_years)) - 1) * 100
        else:
            strategy_cagr = 0.0
            bh_qqq_cagr = 0.0

        # Determine the asset held at the start date (for display)
        initial_trade = trade_history_df.iloc[0]['Asset'] if not trade_history_df.empty else "CASH"

        results.append({
            "Timeframe": label,
            "Start Date": first_trade_day.strftime('%Y-%m-%d'),
            "First Trade": initial_trade,
            "Strategy Value": final_value,
            "B&H QQQ Value": buy_and_hold_qqq,
            "P/L": profit_loss,
            "B&H P/L": bh_profit_loss,
            "Strategy CAGR": strategy_cagr, 
            "B&H CAGR": bh_qqq_cagr,
            "Strategy MDD": strategy_mdd,
            "B&H MDD": bh_mdd,
            "Strategy Sharpe": strategy_sharpe,
            "B&H Sharpe": bh_sharpe
        })

    return results, trade_history_df


# --- Charting ---

def create_chart(data, target_date, target_price=None):
    """Creates a Plotly candlestick chart for QQQ with indicators."""
    
    # Filter data up to the signal day (or today if target is today)
    data_filtered = data[data.index.date <= target_date].copy()
    
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
        # Set default date to yesterday's close
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


    # --- 2. Data Fetching ---
    try:
        full_data = fetch_historical_data()
    except Exception as e:
        st.error(f"Error fetching data from yfinance: {e}")
        return

    # Adjust selected date to the last available trading day if it's today and market is not closed
    target_date = get_last_closed_trading_day(selected_date)
    
    # --- 3. Indicator Calculation ---
    indicators = calculate_indicators(full_data, target_date)
    
    # If override is active, create a new indicators object with the overridden price
    final_price = full_data.loc[full_data.index.date == target_date, f'{TICKER}_adj_close'].iloc[-1] if not full_data.empty and target_date in full_data.index.date else None
    
    if st.session_state['override_enabled'] and st.session_state['override_price'] is not None and indicators is not None:
        final_price = st.session_state['override_price']
        
        # Create a mutable copy of the latest day's indicators to test the override price
        indicators_override = indicators.copy()
        
        # 1. Update the price
        indicators_override[f'{TICKER}_adj_close'] = final_price
        
        # 2. Recalculate Stop-Loss Levels based on the *new* price and the *old* indicators
        # Note: EMA_5 and ATR_14 are based on the *previous* day's close and are not affected by today's price.
        # However, the stop levels *relative to EMA_5* are what matters.
        ema_5 = indicators_override['EMA_5']
        atr_14 = indicators_override['ATR_14']
        
        indicators_override['VASL_LEVEL'] = ema_5 - (atr_14 * ATR_MULTIPLIER)
        indicators_override['MINI_VASL_LEVEL'] = ema_5 - (atr_14 * 0.5)
        
        # Use the overridden indicators for the signal
        final_indicators = indicators_override
        
    else:
        final_indicators = indicators
        
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
        elif "CASH" in final_signal:
            st.info(f"**{final_signal}**", icon="💰")
        else:
            st.error(f"**{final_signal}**", icon="❓")
            
        st.markdown(f"**Signal Date (Close):** {target_date.strftime('%Y-%m-%d')}")
        st.markdown(f"**QQQ Price:** **${final_price:,.2f}**")
        if st.session_state['override_enabled']:
            st.markdown(f"*(Price is overridden)*")
            
        st.subheader("Key Indicator Levels")
        if final_indicators is not None and not pd.isna(final_indicators['SMA_200']):
            st.metric(f"{SMA_PERIOD}-Day SMA", f"${final_indicators['SMA_200']:,.2f}")
            st.metric(f"{EMA_PERIOD}-Day EMA", f"${final_indicators['EMA_5']:,.2f}")
            st.metric("VASL Stop (x2.0 ATR)", f"${final_indicators['VASL_LEVEL']:,.2f}", delta=f"{final_price - final_indicators['VASL_LEVEL']:,.2f}", delta_color="inverse")
            st.metric("Mini-VASL Exit (x0.5 ATR)", f"${final_indicators['MINI_VASL_LEVEL']:,.2f}", delta=f"{final_price - final_indicators['MINI_VASL_LEVEL']:,.2f}", delta_color="inverse")
        else:
            st.warning("Not enough data (200 days) to calculate full indicators.")
            
    # --- 6. Display Chart ---
    with col2:
        if final_indicators is not None:
            fig = create_chart(full_data, target_date, target_price=final_price)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # --- 7. Backtesting Results ---
    st.header("🧪 Strategy Backtest Performance (vs. QQQ Buy & Hold)")
    
    # Run all backtests
    backtest_results, trade_history_df = run_backtests(full_data, target_date)
    
    if backtest_results:
        df_results = pd.DataFrame(backtest_results)
        
        # Format Value columns
        df_results['Strategy Value'] = df_results['Strategy Value'].map('${:,.2f}'.format)
        df_results['B&H QQQ Value'] = df_results['B&H QQQ Value'].map('${:,.2f}'.format)

        # Format the new CAGR columns
        df_results['Strategy CAGR'] = df_results['Strategy CAGR'].map(lambda x: f"{'+' if x >= 0 else ''}{x:,.2f}%")
        df_results['B&H CAGR'] = df_results['B&H CAGR'].map(lambda x: f"{'+' if x >= 0 else ''}{x:,.2f}%")
        
        # New Formatting for Risk Metrics
        df_results['Strategy MDD'] = df_results['Strategy MDD'].map('{:,.2%}'.format)
        df_results['B&H MDD'] = df_results['B&H MDD'].map('{:,.2%}'.format)
        df_results['Strategy Sharpe'] = df_results['Strategy Sharpe'].map('{:,.2f}'.format)
        df_results['B&H Sharpe'] = df_results['B&H Sharpe'].map('{:,.2f}'.format)
        
        df_results = df_results.rename(columns={
            "B&H QQQ Value": "B&H Value", 
            "B&H P/L": "B&H P/L", 
            "Initial Trade": "First Trade",
            "Strategy MDD": "Strategy Max Drawdown",
            "B&H MDD": "B&H Max Drawdown"
        })
        
        # Updated column order to include all new risk metrics
        column_order = [
            "Timeframe", 
            "Start Date", 
            "First Trade", 
            "Strategy Value", 
            "Strategy CAGR", 
            "Strategy Sharpe", 
            "Strategy Max Drawdown",
            "B&H Value", 
            "B&H CAGR",
            "B&H Sharpe",
            "B&H Max Drawdown"
        ]
        df_results = df_results[column_order]
        st.dataframe(df_results, hide_index=True)

    st.markdown("---")
    
    # --- 8. Display Detailed Trade History ---
    st.header(f"📜 Detailed Trade History (From {target_date.strftime('%Y-%m-%d')} to Today)")
    st.caption("Trades listed here are executed at the Open price of the Date shown.")
    
    if not trade_history_df.empty:
        
        # Apply formatting
        trade_history_df['Price'] = trade_history_df['Price'].map('${:,.2f}'.format)
        trade_history_df['Portfolio Value'] = trade_history_df['Portfolio Value'].map('${:,.2f}'.format)
        
        # Display the dataframe without custom row styling
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
