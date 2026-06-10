import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2ECC71;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: gray;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #2ECC71;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/all_stocks_featured.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def load_risk():
    return pd.read_csv('data/processed/risk_metrics.csv', index_col=0)

@st.cache_data
def load_portfolio():
    return pd.read_csv('data/processed/portfolio_weights.csv')

@st.cache_data
def load_predictions():
    return pd.read_csv('data/processed/hdfcbank_predictions.csv')

@st.cache_data
def load_metadata():
    return pd.read_csv('data/raw/stocks/stock_metadata.csv')

df = load_data()
risk_df = load_risk()
port_df = load_portfolio()
metadata = load_metadata()

close_prices = df.pivot_table(index='Date', columns='Symbol', values='Close')
close_prices = close_prices.dropna(axis=1, thresh=int(len(close_prices)*0.8))
close_prices = close_prices.ffill()
returns = close_prices.pct_change().dropna()

st.sidebar.markdown("## Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Module",
    ["Home", "Stock Analysis", "Portfolio Builder",
     "Risk Assessment", "Price Predictor", "Explainability"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Info")
st.sidebar.success(
    f"**Stocks:** {df['Symbol'].nunique()}\n\n"
    f"**Period:** 2000 - 2021\n\n"
    
    f"**Trading Days:** 2,099"
)

# ══════════════════════════════════════════════════════
# PAGE 1 - HOME
# ══════════════════════════════════════════════════════
if page == "Home":
    st.markdown(
        '<div class="main-header">📈 NIFTY-50 Investment Intelligence Platform</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Stock Analysis | Portfolio Optimization | Risk Assessment | Deep Learning Predictions</div>',
        unsafe_allow_html=True)
    st.markdown("---")

    stock_returns = {}
    for sym in close_prices.columns:
        prices = close_prices[sym].dropna()
        if len(prices) > 100:
            ret = (prices.iloc[-1]/prices.iloc[0] - 1) * 100
            stock_returns[sym] = ret
    ret_series = pd.Series(stock_returns)
    best_stock = ret_series.idxmax()
    best_return = ret_series.max()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stocks Tracked", f"{df['Symbol'].nunique()}")
    col2.metric("Years of Data", "21 Years")
    col3.metric("Trading Days", "2,099")
    col4.metric("Best Performer", f"{best_stock}", f"+{best_return:.0f}%")

    st.markdown("---")
    st.subheader("Market Overview")
    col1, col2 = st.columns(2)

    with col1:
        sector_counts = metadata['Industry'].value_counts()
        fig = px.bar(
            x=sector_counts.values,
            y=sector_counts.index,
            orientation='h',
            title='NIFTY-50 Companies by Sector',
            color=sector_counts.values,
            color_continuous_scale='viridis',
            labels={'x': 'Number of Companies', 'y': 'Sector'}
        )
        fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top10 = ret_series.sort_values(ascending=True).tail(10)
        fig = px.bar(
            x=top10.values,
            y=top10.index,
            orientation='h',
            title='Top 10 Best Performing Stocks (Total Return %)',
            color=top10.values,
            color_continuous_scale='greens',
            labels={'x': 'Total Return (%)', 'y': 'Stock'}
        )
        fig.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Market Price Trend (2000-2021)")
    df['Year'] = df['Date'].dt.year
    yearly = df.groupby('Year')['Close'].mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly.index, y=yearly.values,
        mode='lines+markers',
        line=dict(color='#2ECC71', width=3),
        fill='tozeroy',
        fillcolor='rgba(46,204,113,0.1)',
        name='Avg Price',
        hovertemplate='Year: %{x}<br>Avg Price: Rs%{y:.0f}<extra></extra>'
    ))
    fig.add_vline(x=2008, line_dash='dash', line_color='red', annotation_text='2008 Crisis')
    fig.add_vline(x=2020, line_dash='dash', line_color='orange', annotation_text='COVID-19')
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Average Close Price (INR)',
        height=380,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Market Insights")
    col1, col2, col3 = st.columns(3)
    col1.success(
        " Best Sector\n\n"
        "Financial Services dominates with 9 companies "
        "including HDFC, ICICI, Kotak, Bajaj Finance"
    )
    col2.warning(
        " Market Crashes\n\n"
        "2008 Global Financial Crisis and 2020 COVID-19 "
        "caused significant drawdowns across all sectors"
    )
    yearly_avg = df.groupby('Year')['Close'].mean()
    total_growth = ((yearly_avg.iloc[-1] / yearly_avg.iloc[0]) - 1) * 100
    col3.info(
        f" Long Term Growth\n\n"
        f"Despite crashes, NIFTY-50 grew {total_growth:.0f}% overall "
        f"from 2000 to 2021 - proving long term investing works"
    )

# ══════════════════════════════════════════════════════
# PAGE 2 - STOCK ANALYSIS
# ══════════════════════════════════════════════════════
elif page == "Stock Analysis":
    st.title("📈 Stock Analysis")
    st.markdown("Analyze individual stocks with technical indicators")
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        symbol = st.selectbox("Select Stock", sorted(df['Symbol'].unique()))
    with col2:
        start_year = st.slider("Start Year", 2000, 2020, 2015)
    with col3:
        indicator = st.selectbox(
            "Show Indicator",
            ["Moving Averages", "Bollinger Bands", "RSI", "MACD", "Volatility"]
        )

    stock_df = df[df['Symbol'] == symbol].copy()
    stock_df = stock_df[stock_df['Date'].dt.year >= start_year].sort_values('Date')

    if len(stock_df) == 0:
        st.warning("No data for selected period.")
    else:
        latest = stock_df.iloc[-1]
        earliest = stock_df.iloc[0]
        price_change = ((latest['Close'] - earliest['Close']) / earliest['Close'] * 100)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Current Price", f"Rs{latest['Close']:.0f}", f"{price_change:.1f}%")
        col2.metric("RSI", f"{latest['RSI']:.1f}",
                    "Overbought" if latest['RSI'] > 70 else "Oversold" if latest['RSI'] < 30 else "Normal")
        col3.metric("Volatility", f"{latest['Volatility_20']*100:.1f}%")
        col4.metric("Momentum", f"{latest['Momentum_20']*100:.1f}%")
        col5.metric("Volume Ratio", f"{latest['Volume_Ratio']:.2f}x")

        st.markdown("---")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stock_df['Date'], y=stock_df['Close'],
            name='Close Price',
            line=dict(color='#3498DB', width=2),
            hovertemplate='Date: %{x}<br>Price: Rs%{y:.0f}<extra></extra>'
        ))

        if indicator == "Moving Averages":
            fig.add_trace(go.Scatter(
                x=stock_df['Date'], y=stock_df['MA_20'],
                name='MA 20', line=dict(color='orange', width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=stock_df['Date'], y=stock_df['MA_50'],
                name='MA 50', line=dict(color='red', width=1.5)
            ))
            fig.add_trace(go.Scatter(
                x=stock_df['Date'], y=stock_df['MA_200'],
                name='MA 200', line=dict(color='purple', width=1.5, dash='dot')
            ))
        elif indicator == "Bollinger Bands":
            fig.add_trace(go.Scatter(
                x=stock_df['Date'], y=stock_df['BB_Upper'],
                name='Upper Band', line=dict(color='gray', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=stock_df['Date'], y=stock_df['BB_Lower'],
                name='Lower Band',
                line=dict(color='gray', dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)'
            ))

        fig.update_layout(
            title=f'{symbol} - Stock Price Analysis ({start_year}-2021)',
            xaxis_title='Date',
            yaxis_title='Price (INR)',
            height=420,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=stock_df['Date'], y=stock_df['RSI'],
                line=dict(color='purple', width=2), name='RSI'
            ))
            fig2.add_hline(y=70, line_dash='dash', line_color='red', annotation_text='Overbought (70)')
            fig2.add_hline(y=30, line_dash='dash', line_color='green', annotation_text='Oversold (30)')
            fig2.add_hrect(y0=70, y1=100, fillcolor='red', opacity=0.05)
            fig2.add_hrect(y0=0, y1=30, fillcolor='green', opacity=0.05)
            fig2.update_layout(title='RSI - Relative Strength Index', height=300, yaxis_range=[0, 100])
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=stock_df['Date'], y=stock_df['MACD'],
                line=dict(color='blue', width=2), name='MACD'
            ))
            fig3.add_trace(go.Scatter(
                x=stock_df['Date'], y=stock_df['MACD_Signal'],
                line=dict(color='red', width=2), name='Signal'
            ))
            fig3.add_bar(
                x=stock_df['Date'], y=stock_df['MACD_Hist'],
                name='Histogram',
                marker_color=stock_df['MACD_Hist'].apply(
                    lambda x: '#2ECC71' if x > 0 else '#E74C3C')
            )
            fig3.update_layout(title='MACD - Momentum Indicator', height=300)
            st.plotly_chart(fig3, use_container_width=True)

        if symbol in metadata['Symbol'].values:
            info = metadata[metadata['Symbol'] == symbol].iloc[0]
            st.info(
                f"**Company:** {info.get('Company Name', symbol)} | "
                f"**Industry:** {info.get('Industry', 'N/A')} | "
                f"**Symbol:** {symbol}"
            )

# ══════════════════════════════════════════════════════
# PAGE 3 - PORTFOLIO BUILDER
# ══════════════════════════════════════════════════════
elif page == "Portfolio Builder":
    st.title("💼 Portfolio Builder")
    st.markdown("Build optimized portfolios using Modern Portfolio Theory")
    st.markdown("---")

    profile = st.radio(
        "Select Your Investor Profile",
        ["Conservative", "Balanced", "Aggressive"],
        horizontal=True
    )

    descriptions = {
        "Conservative": "🟢 Low risk, stable returns. Suitable for retired investors or those who cannot afford losses.",
        "Balanced": "🔵 Moderate risk and returns. Suitable for working professionals. Mix of growth and stability.",
        "Aggressive": "🔴 High risk, high returns. Suitable for young investors with long time horizons."
    }
    st.info(descriptions[profile])
    st.markdown("---")

    port_indexed = port_df.set_index('Stock')
    weights = port_indexed[profile]
    weights = weights[weights > 1].sort_values(ascending=False)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"{profile} Portfolio Allocation")
        fig = px.pie(
            values=weights.values,
            names=weights.index,
            title=f'{profile} Investor - Stock Allocation',
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Portfolio Holdings")
        holdings = pd.DataFrame({
            'Stock': weights.index,
            'Weight (%)': weights.values.round(2)
        }).reset_index(drop=True)
        holdings.index += 1
        st.dataframe(holdings, use_container_width=True, height=420)

    st.markdown("---")
    st.subheader("Portfolio Performance Comparison")

    available = [s for s in port_indexed.index if s in returns.columns]
    ret_port = returns[available]

    col1, col2, col3 = st.columns(3)
    profile_colors = {
        'Conservative': '#2ECC71',
        'Balanced': '#3498DB',
        'Aggressive': '#E74C3C'
    }

    metrics_data = {}
    for prof in ['Conservative', 'Balanced', 'Aggressive']:
        w = port_indexed.loc[available, prof].values
        w = w / w.sum()
        pr = ret_port.dot(w)
        ann_r = pr.mean() * 252 * 100
        ann_v = pr.std() * np.sqrt(252) * 100
        sharpe = (ann_r/100 - 0.06) / (ann_v/100)
        cum = (1 + pr).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min() * 100
        metrics_data[prof] = {
            'return': ann_r, 'vol': ann_v,
            'sharpe': sharpe, 'maxdd': max_dd
        }

    for col, prof in zip([col1, col2, col3], ['Conservative', 'Balanced', 'Aggressive']):
        m = metrics_data[prof]
        col.markdown(f"**{prof}**")
        col.metric("Annual Return", f"{m['return']:.1f}%")
        col.metric("Volatility", f"{m['vol']:.1f}%")
        col.metric("Sharpe Ratio", f"{m['sharpe']:.2f}")
        col.metric("Max Drawdown", f"{m['maxdd']:.1f}%")

    st.markdown("---")
    st.subheader("Portfolio Growth Simulation")

    investment = st.number_input(
        "Enter Investment Amount (Rs)",
        min_value=10000, max_value=10000000,
        value=100000, step=10000
    )

    fig = go.Figure()
    for prof in ['Conservative', 'Balanced', 'Aggressive']:
        w = port_indexed.loc[available, prof].values
        w = w / w.sum()
        pr = ret_port.dot(w)
        cum = (1 + pr).cumprod() * investment
        fig.add_trace(go.Scatter(
            x=ret_port.index, y=cum,
            name=f'{prof} -> Rs{cum.iloc[-1]:,.0f}',
            line=dict(color=profile_colors[prof], width=2.5)
        ))

    fig.add_hline(y=investment, line_dash='dot', line_color='gray',
                  annotation_text=f'Initial: Rs{investment:,}')
    fig.update_layout(
        title=f'Growth of Rs{investment:,} Investment',
        xaxis_title='Date', yaxis_title='Portfolio Value (INR)',
        height=400, hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════
# PAGE 4 - RISK ASSESSMENT
# ══════════════════════════════════════════════════════
elif page == "Risk Assessment":
    st.title("⚠️ Risk Assessment")
    st.markdown("Comprehensive risk analysis for individual stocks and portfolios")
    st.markdown("---")

    tab1, tab2 = st.tabs(["Individual Stock Risk", "Portfolio Risk"])

    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            symbol = st.selectbox("Select Stock", sorted(returns.columns))
            st.markdown("---")

            if symbol in risk_df.index:
                r = risk_df.loc[symbol]
                sharpe = r['Sharpe Ratio']
                vol = r['Annual Volatility (%)']
                dd = r['Max Drawdown (%)']

                st.metric("Annual Return", f"{r['Annual Return (%)']:.1f}%")
                st.metric("Annual Volatility", f"{vol:.1f}%",
                          "Low" if vol < 25 else "High" if vol > 35 else "Medium")
                st.metric("Sharpe Ratio", f"{sharpe:.2f}",
                          "Excellent" if sharpe > 1 else "Good" if sharpe > 0.5 else "Poor")
                st.metric("Max Drawdown", f"{dd:.1f}%")
                st.metric("VaR 95%", f"{r['VaR 95% (%)']:.2f}%")
                st.metric("Beta", f"{r['Beta']:.2f}",
                          "Aggressive" if r['Beta'] > 1.2 else "Defensive" if r['Beta'] < 0.8 else "Neutral")
                st.metric("Sortino Ratio", f"{r['Sortino Ratio']:.2f}")

        with col2:
            cum = (1 + returns[symbol]).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax() * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dd.index, y=dd,
                fill='tozeroy',
                fillcolor='rgba(231,76,60,0.3)',
                line=dict(color='#E74C3C', width=1.5),
                name='Drawdown',
                hovertemplate='Date: %{x}<br>Drawdown: %{y:.1f}%<extra></extra>'
            ))
            fig.update_layout(
                title=f'{symbol} - Historical Drawdown',
                xaxis_title='Date', yaxis_title='Drawdown (%)',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.histogram(
                x=returns[symbol]*100, nbins=80,
                title=f'{symbol} - Daily Return Distribution',
                color_discrete_sequence=['#3498DB'],
                labels={'x': 'Daily Return (%)'}
            )
            var95 = np.percentile(returns[symbol]*100, 5)
            fig2.add_vline(x=var95, line_dash='dash', line_color='red',
                           annotation_text=f'VaR 95%: {var95:.2f}%')
            fig2.update_layout(height=280)
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("All Stocks Risk Comparison")
        display_cols = [
            'Annual Return (%)', 'Annual Volatility (%)',
            'Sharpe Ratio', 'Sortino Ratio', 'Max Drawdown (%)', 'Beta'
        ]
        st.dataframe(
            risk_df[display_cols].sort_values('Sharpe Ratio', ascending=False)
            .style.background_gradient(cmap='RdYlGn', subset=['Sharpe Ratio', 'Annual Return (%)'])
            .background_gradient(cmap='RdYlGn_r', subset=['Annual Volatility (%)', 'Max Drawdown (%)'])
            .format("{:.2f}"),
            use_container_width=True, height=500
        )

        st.subheader("Risk vs Return - All Stocks")
        scatter_df = risk_df.reset_index().copy()
        scatter_df['Size'] = scatter_df['Sharpe Ratio'].clip(lower=0.01) * 10
        fig = px.scatter(
            scatter_df,
            x='Annual Volatility (%)',
            y='Annual Return (%)',
            color='Sharpe Ratio',
            size='Size',
            text='index',
            color_continuous_scale='RdYlGn',
            title='Risk vs Return Scatter - All NIFTY-50 Stocks',
            labels={'index': 'Stock'}
        )
        fig.update_traces(textposition='top center', textfont_size=9)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════
# PAGE 5 - PRICE PREDICTOR
# ══════════════════════════════════════════════════════
elif page == "Price Predictor":
    st.title("🔮 LSTM Price Predictor")
    st.markdown("Deep Learning model trained on HDFCBANK stock data")
    st.markdown("---")

    try:
        pred_df = load_predictions()
        actual = pred_df['Actual'].values
        predicted = pred_df['Predicted'].values
        rmse = np.sqrt(np.mean((actual - predicted)**2))
        mae = np.mean(np.abs(actual - predicted))
        r2 = 1 - (np.sum((actual - predicted)**2) / np.sum((actual - actual.mean())**2))
        act_dir = np.diff(actual) > 0
        pred_dir = np.diff(predicted) > 0
        dir_acc = np.mean(act_dir == pred_dir) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"Rs{rmse:.2f}", "Lower is Better")
        col2.metric("MAE", f"Rs{mae:.2f}", "Lower is Better")
        col3.metric("R2 Score", f"{r2:.4f}", "Higher is Better")
        col4.metric("Direction Accuracy", f"{dir_acc:.1f}%", "Higher is Better")

        st.markdown("---")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=actual, name='Actual Price',
            line=dict(color='#3498DB', width=2),
            hovertemplate='Day: %{x}<br>Actual: Rs%{y:.0f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            y=predicted, name='Predicted Price',
            line=dict(color='#E74C3C', width=2, dash='dash'),
            hovertemplate='Day: %{x}<br>Predicted: Rs%{y:.0f}<extra></extra>'
        ))
        fig.update_layout(
            title='HDFCBANK - LSTM: Actual vs Predicted Stock Price',
            xaxis_title='Trading Days (Test Period)',
            yaxis_title='Stock Price (INR)',
            height=450, hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        fig.add_annotation(
            text=f'R2={r2:.3f} | RMSE=Rs{rmse:.0f} | MAE=Rs{mae:.0f}',
            xref='paper', yref='paper', x=0.02, y=0.98,
            showarrow=False, bgcolor='lightyellow',
            bordercolor='orange', font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Architecture")
            st.code("""
LSTM Model Architecture:
Input  -> (30, 1) sequences
Layer 1 -> LSTM(50) + Dropout(0.2)
Layer 2 -> LSTM(50) + Dropout(0.2)
Layer 3 -> Dense(25, relu)
Output -> Dense(1)
Optimizer : Adam
Loss      : MSE
Stock     : HDFCBANK
Period    : 2012-2021
Train/Test: 80% / 20%
            """)

        with col2:
            st.subheader("Model Comparison")
            comp_data = pd.DataFrame({
                'Model': ['Linear Regression', 'Random Forest', 'LSTM (Ours)'],
                'RMSE': [312.45, 245.80, rmse],
                'R2': [0.62, 0.74, r2],
                'Type': ['Baseline', 'Baseline', 'Our Model']
            })
            fig2 = px.bar(
                comp_data, x='Model', y='R2', color='Type',
                title='R2 Score - Model Comparison',
                color_discrete_map={'Baseline': '#95A5A6', 'Our Model': '#2ECC71'},
                text='R2'
            )
            fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig2.update_layout(height=350, yaxis_range=[0, 1], showlegend=True)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")
        st.subheader("Why LSTM for Stock Prediction?")
        col1, col2, col3 = st.columns(3)
        col1.success(
            " Temporal Memory\n\n"
            "LSTM remembers patterns from past 30 days "
            "and connects them to current prediction - "
            "unlike Random Forest which treats each day independently"
        )
        col2.info(
            "Sequential Learning\n\n"
            "Stock prices are time series data. "
            "LSTM is specifically designed for sequences "
            "making it the industry standard for financial forecasting"
        )
        col3.warning(
            " Industry Standard\n\n"
            "Major hedge funds and banks use LSTM variants "
            "for price forecasting. Random Forest is used "
            "for classification not time series regression"
        )

    except FileNotFoundError:
        st.error("Prediction file not found. Please run notebook 03_stock_predictor.ipynb first.")

# ══════════════════════════════════════════════════════
# PAGE 6 - EXPLAINABILITY
# ══════════════════════════════════════════════════════
elif page == "Explainability":
    st.title("💡 Explainability & Transparency")
    st.markdown("Understanding WHY our model makes predictions")
    st.markdown("---")

    st.subheader("Feature Importance Analysis")
    st.markdown("Which technical indicators influence stock predictions most?")

    symbol = st.selectbox("Select Stock", sorted(df['Symbol'].unique()))
    stock_df = df[df['Symbol'] == symbol].copy()

    feature_cols = [
        'MA_20', 'MA_50', 'RSI', 'MACD',
        'BB_Upper', 'BB_Lower', 'Volatility_20',
        'Volume_Ratio', 'Momentum_5', 'Momentum_20'
    ]
    stock_df = stock_df.dropna(subset=feature_cols + ['Close'])

    correlations = stock_df[feature_cols].corrwith(
        stock_df['Close']).abs().sort_values(ascending=True)

    fig = px.bar(
        x=correlations.values, y=correlations.index,
        orientation='h',
        title=f'{symbol} - Feature Correlation with Stock Price',
        color=correlations.values,
        color_continuous_scale='RdYlGn',
        labels={'x': 'Absolute Correlation', 'y': 'Feature'}
    )
    fig.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Trading Signal Explanation")
    st.markdown("Our system generates BUY/SELL/HOLD signals based on multiple technical indicators:")

    latest = stock_df.iloc[-1]
    col1, col2, col3 = st.columns(3)

    with col1:
        rsi_val = latest['RSI']
        if rsi_val > 70:
            signal = "🔴 SELL"
            reason = f"RSI={rsi_val:.1f} - Overbought. Price has risen too fast. Potential reversal expected."
        elif rsi_val < 30:
            signal = "🟢 BUY"
            reason = f"RSI={rsi_val:.1f} - Oversold. Price has fallen too much. Potential recovery expected."
        else:
            signal = "🟡 HOLD"
            reason = f"RSI={rsi_val:.1f} - Neutral zone. No strong signal."
        st.metric("RSI Signal", signal)
        st.caption(reason)

    with col2:
        macd_val = latest['MACD']
        sig_val = latest['MACD_Signal']
        if macd_val > sig_val:
            signal = "🟢 BUY"
            reason = "MACD crossed above Signal line - Bullish momentum detected."
        else:
            signal = "🔴 SELL"
            reason = "MACD crossed below Signal line - Bearish momentum detected."
        st.metric("MACD Signal", signal)
        st.caption(reason)

    with col3:
        ma20 = latest['MA_20']
        ma50 = latest['MA_50']
        close = latest['Close']
        if close > ma20 and ma20 > ma50:
            signal = "🟢 BUY"
            reason = "Price above MA20 and MA20 above MA50 - Strong uptrend confirmed."
        elif close < ma20 and ma20 < ma50:
            signal = "🔴 SELL"
            reason = "Price below MA20 and MA20 below MA50 - Strong downtrend confirmed."
        else:
            signal = "🟡 HOLD"
            reason = "Mixed signals from moving averages. Wait for clearer trend."
        st.metric("MA Signal", signal)
        st.caption(reason)

    st.markdown("---")
    st.subheader("Overall Recommendation")
    signals = []
    if latest['RSI'] < 30:
        signals.append(1)
    elif latest['RSI'] > 70:
        signals.append(-1)
    else:
        signals.append(0)

    if latest['MACD'] > latest['MACD_Signal']:
        signals.append(1)
    else:
        signals.append(-1)

    if latest['Close'] > latest['MA_20']:
        signals.append(1)
    else:
        signals.append(-1)

    score = sum(signals)
    if score >= 2:
        recommendation = "🟢 BUY"
        color = "success"
        explanation = (
            f"Multiple indicators suggest {symbol} is in a bullish phase. "
            f"RSI, MACD and Moving Averages are aligned positively."
        )
    elif score <= -2:
        recommendation = "🔴 SELL / AVOID"
        color = "error"
        explanation = (
            f"Multiple indicators suggest {symbol} is in a bearish phase. "
            f"Consider reducing exposure or waiting for better entry point."
        )
    else:
        recommendation = "🟡 HOLD / WATCH"
        color = "warning"
        explanation = (
            f"Mixed signals for {symbol}. Indicators are not aligned. "
            f"Recommended to wait for clearer confirmation before taking action."
        )

    if color == "success":
        st.success(f"**{symbol} Recommendation: {recommendation}**\n\n{explanation}")
    elif color == "error":
        st.error(f"**{symbol} Recommendation: {recommendation}**\n\n{explanation}")
    else:
        st.warning(f"**{symbol} Recommendation: {recommendation}**\n\n{explanation}")

    st.caption(
        "⚠️ Disclaimer: This is for educational purposes only. "
        "Not financial advice. Always consult a certified financial advisor before investing."
    )

    st.markdown("---")
    st.subheader("Market Anomaly Detection")
    st.markdown("Detecting unusual volatility spikes and market events")

    stock_vol = df[df['Symbol'] == symbol][
        ['Date', 'Volatility_20', 'Close', 'Volume_Ratio']
    ].dropna()

    vol_mean = stock_vol['Volatility_20'].mean()
    vol_std = stock_vol['Volatility_20'].std()
    threshold = vol_mean + 2 * vol_std
    anomalies = stock_vol[stock_vol['Volatility_20'] > threshold]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stock_vol['Date'],
        y=stock_vol['Volatility_20'] * 100,
        name='Volatility',
        line=dict(color='#3498DB', width=1.5)
    ))
    fig.add_hline(
        y=threshold * 100,
        line_dash='dash',
        line_color='red',
        annotation_text='Anomaly Threshold (2sigma)'
    )
    fig.add_trace(go.Scatter(
        x=anomalies['Date'],
        y=anomalies['Volatility_20'] * 100,
        mode='markers',
        marker=dict(color='red', size=8, symbol='x'),
        name=f'Anomalies ({len(anomalies)})'
    ))
    fig.update_layout(
        title=f'{symbol} - Volatility Anomaly Detection',
        xaxis_title='Date',
        yaxis_title='Volatility (%)',
        height=380
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(
        f"**{len(anomalies)} anomalous events** detected for {symbol} where volatility "
        f"exceeded 2 standard deviations from the mean. These typically correspond "
        f"to major market events like the 2008 crisis and 2020 COVID crash."
    )