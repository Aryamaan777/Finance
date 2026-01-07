import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Configuration ---
st.set_page_config(page_title="Stock EDA Dashboard", layout="wide")
st.title("📈 Stock Exploratory Data Analysis (EDA) Dashboard")

# --- Sidebar Inputs ---
st.sidebar.header("Configuration")

# Default tickers from your script
default_tickers = ['^GSPC', 'AAPL', 'TSLA', 'SPY']
tickers_input = st.sidebar.text_input("Enter Tickers (comma separated)", value=", ".join(default_tickers))
tickers = [x.strip() for x in tickers_input.split(',')]

# Date Range Selection
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2019-01-03"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))

# --- Data Loading ---
@st.cache_data
def load_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end, auto_adjust=False)
        # Handle yfinance MultiIndex issue if multiple tickers are present
        if isinstance(data.columns, pd.MultiIndex):
            # Ensure we can access via levels easily
            pass 
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data(tickers, start_date, end_date)

if data is None or data.empty:
    st.warning("No data found. Please check your ticker symbols.")
    st.stop()

# Prepare Adj Close DataFrame for general analysis
if isinstance(data.columns, pd.MultiIndex):
    adj_close = data['Adj Close']
    volume_data = data['Volume']
else:
    # If only one ticker is downloaded, yfinance structure is flat
    adj_close = pd.DataFrame(data['Adj Close']).rename(columns={'Adj Close': tickers[0]})
    volume_data = pd.DataFrame(data['Volume']).rename(columns={'Volume': tickers[0]})

# Calculate Returns
returns = adj_close.pct_change().dropna()

# --- Layout: Tabs for Organization ---
tab1, tab2, tab3, tab4 = st.tabs(["Price Analysis", "Portfolio Performance", "Risk Metrics", "Distributions & Volume"])

# ==========================================
# TAB 1: PRICE ANALYSIS (SMA & Candlestick)
# ==========================================
with tab1:
    st.header("Price Analysis & Moving Averages")
    
    # User selects which asset to view in detail
    selected_ticker = st.selectbox("Select Asset for Detailed View", tickers)

    # 1. Moving Averages Chart
    # Extract series for the selected ticker
    if isinstance(data.columns, pd.MultiIndex):
        ticker_series = adj_close[selected_ticker]
    else:
        ticker_series = adj_close[tickers[0]] # Fallback for single ticker

    sma50 = ticker_series.rolling(window=50).mean()
    sma200 = ticker_series.rolling(window=200).mean()

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=ticker_series.index, y=ticker_series, mode='lines', name=f'{selected_ticker} Adj Close', line=dict(width=2, color='royalblue')))
    fig_ma.add_trace(go.Scatter(x=sma50.index, y=sma50, mode='lines', name='50-day SMA', line=dict(width=1.5, dash='dash', color='orange')))
    fig_ma.add_trace(go.Scatter(x=sma200.index, y=sma200, mode='lines', name='200-day SMA', line=dict(width=1.5, dash='dot', color='red')))
    
    fig_ma.update_layout(title=f'{selected_ticker} Price Trend & Moving Averages', xaxis_title='Date', yaxis_title='Price (USD)', hovermode='x unified')
    st.plotly_chart(fig_ma, use_container_width=True)

    # 2. Candlestick & Volume Chart
    st.subheader(f"{selected_ticker} Candlestick & Volume")
    
    # Extract OHLC data specifically for the selected ticker
    if isinstance(data.columns, pd.MultiIndex):
        # Using xs to slice the MultiIndex to get OHLC for specific ticker
        # Note: yfinance structure is usually columns=[(PriceType, Ticker), ...] or [(Ticker, PriceType)]
        # We try to locate the ticker in the level 1 (Ticker level) if formatted that way
        try:
            # Recent yfinance versions put Ticker in columns.
            # We reconstruct a single DF for the ticker.
            ohlc_dict = {
                'Open': data['Open'][selected_ticker],
                'High': data['High'][selected_ticker],
                'Low': data['Low'][selected_ticker],
                'Close': data['Close'][selected_ticker],
                'Volume': data['Volume'][selected_ticker]
            }
            ticker_ohlc = pd.DataFrame(ohlc_dict)
        except KeyError:
            st.error("Could not parse OHLC data structure.")
            ticker_ohlc = pd.DataFrame()
    else:
        ticker_ohlc = data

    if not ticker_ohlc.empty:
        fig_candle = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=('OHLC', 'Volume'), row_width=[0.2, 0.7])
        fig_candle.add_trace(go.Candlestick(x=ticker_ohlc.index, open=ticker_ohlc['Open'], high=ticker_ohlc['High'], low=ticker_ohlc['Low'], close=ticker_ohlc['Close'], name='OHLC'), row=1, col=1)
        fig_candle.add_trace(go.Bar(x=ticker_ohlc.index, y=ticker_ohlc['Volume'], name='Volume'), row=2, col=1)
        fig_candle.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_candle, use_container_width=True)

# ==========================================
# TAB 2: PORTFOLIO PERFORMANCE
# ==========================================
with tab2:
    st.header("Portfolio Performance Comparison")

    # 1. Cumulative Returns
    cumulative_returns = (1 + returns).cumprod()
    fig_cum = px.line(cumulative_returns, title='Growth of $1 Investment (Cumulative Returns)', labels={'value': 'Portfolio Value', 'index': 'Date'})
    st.plotly_chart(fig_cum, use_container_width=True)

    col1, col2 = st.columns(2)
    
    # 2. Correlation Matrix
    with col1:
        st.subheader("Asset Correlation Matrix")
        corr_matrix = returns.corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig_corr, use_container_width=True)

    # 3. Risk vs Return
    with col2:
        st.subheader("Risk vs. Return Profile")
        risk_return = pd.DataFrame({
            'Risk (Volatility)': returns.std() * np.sqrt(252),
            'Return': returns.mean() * 252
        })
        fig_rr = px.scatter(risk_return, x='Risk (Volatility)', y='Return', text=risk_return.index, size_max=60)
        fig_rr.update_traces(textposition='top center')
        st.plotly_chart(fig_rr, use_container_width=True)

    # 4. Rolling Correlation (Benchmark)
    st.subheader("Rolling Correlation to Benchmark")
    benchmark_options = tickers
    benchmark = st.selectbox("Select Benchmark for Correlation", benchmark_options, index=0)
    
    window_days = st.slider("Rolling Window (Days)", 30, 365, 90)
    
    rolling_corr_df = pd.DataFrame(index=returns.index)
    for ticker in tickers:
        if ticker != benchmark:
            rolling_corr_df[ticker] = returns[ticker].rolling(window=window_days).corr(returns[benchmark])
            
    fig_roll_corr = px.line(rolling_corr_df, title=f"{window_days}-Day Rolling Correlation to {benchmark}")
    fig_roll_corr.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Perfect Correlation")
    fig_roll_corr.add_hline(y=0.0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_roll_corr, use_container_width=True)

# ==========================================
# TAB 3: RISK METRICS
# ==========================================
with tab3:
    st.header("Risk Analysis")

    # 1. Drawdowns
    st.subheader("Drawdown Analysis")
    roll_max = adj_close.expanding().max()
    drawdown = (adj_close - roll_max) / roll_max
    fig_dd = px.line(drawdown, title="Drawdown (Percentage from Peak)", labels={'value': 'Drawdown %', 'Date': 'Date'})
    fig_dd.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig_dd, use_container_width=True)

    col1, col2 = st.columns(2)

    # 2. Rolling Volatility
    with col1:
        st.subheader("Rolling Volatility")
        vol_window = st.slider("Volatility Window", 10, 100, 30)
        rolling_vol = returns.rolling(window=vol_window).std() * np.sqrt(252)
        fig_vol = px.line(rolling_vol, title=f"{vol_window}-Day Rolling Annualized Volatility")
        st.plotly_chart(fig_vol, use_container_width=True)

    # 3. Sharpe Ratio
    with col2:
        st.subheader("Sharpe Ratio")
        risk_free_rate = st.number_input("Risk Free Rate", value=0.03, step=0.01)
        annualized_return = returns.mean() * 252
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe_ratios = (annualized_return - risk_free_rate) / annualized_vol
        
        sharpe_table = pd.DataFrame({
            'Sharpe Ratio': sharpe_ratios
        }).sort_values(by='Sharpe Ratio', ascending=False)
        
        fig_sharpe = px.bar(sharpe_table, x=sharpe_table.index, y='Sharpe Ratio', color='Sharpe Ratio', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_sharpe, use_container_width=True)

# ==========================================
# TAB 4: DISTRIBUTIONS & VOLUME
# ==========================================
with tab4:
    st.header("Distributions & Volume Analysis")
    
    target_ticker_dist = st.selectbox("Select Asset for Analysis", tickers, key="dist_select")
    
    col1, col2 = st.columns(2)
    
    # 1. Histogram of Returns
    with col1:
        fig_hist = px.histogram(returns, x=target_ticker_dist, nbins=100, marginal="box", title=f"Distribution of {target_ticker_dist} Daily Returns")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    # 2. Volume vs Return
    with col2:
        # Prepare data for scatter
        if isinstance(data.columns, pd.MultiIndex):
             vol_col = volume_data[target_ticker_dist]
             ret_col = returns[target_ticker_dist]
        else:
             vol_col = volume_data[tickers[0]]
             ret_col = returns[tickers[0]]

        vol_price_df = pd.DataFrame({
            'Daily Return': ret_col,
            'Volume': vol_col
        }).dropna()
        
        fig_vol_scat = px.scatter(vol_price_df, x='Volume', y='Daily Return', trendline="ols", title=f'Volume vs. Daily Return: {target_ticker_dist}', opacity=0.5)
        st.plotly_chart(fig_vol_scat, use_container_width=True)
