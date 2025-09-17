import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import json
import os
import sys
import logging
import base64

sys.path.append('src')
from backtester import QuantBacktester
from data_loader import DataLoader
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.black_scholes import BlackScholesStrategy
from strategies.volatility_scaling_trend import VolatilityScalingTrend

logging.getLogger("watchdog").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("src.visualizer").setLevel(logging.WARNING)
logging.getLogger("backtester").setLevel(logging.WARNING)


def _fmt_pct(val, na="N/A"):
    try:
        if val is None or val == 0:
            return "0.00%"
        if isinstance(val, str):
            val = float(val)
        if isinstance(val, (float, int, np.floating, np.integer)):
            if np.isnan(val) or np.isinf(val):
                return "0.00%"
        return f"{float(val):.2%}"
    except Exception:
        return "0.00%"


def _fmt_float(val, decimals=2, na="N/A"):
    try:
        if val is None or val == 0:
            return "0.00"
        if isinstance(val, str):
            val = float(val)
        if isinstance(val, (float, int, np.floating, np.integer)):
            if np.isnan(val) or np.isinf(val):
                return "0.00"
            if np.isnan(val) or np.isinf(val):
                return na
        return f"{float(val):,.{decimals}f}"
    except Exception:
        return na


def _fmt_int(val, na="N/A"):
    try:
        if val is None:
            return na
        if isinstance(val, str) and val.strip() == "":
            return na
        return f"{int(float(val)):,}"
    except Exception:
        return na


LINKEDIN_URL = "https://www.linkedin.com/in/yash-g-17663b166/"

st.set_page_config(
    page_title="StratForge - Advanced Backtesting Platform",
    page_icon="🗿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .main-header { text-align:center; padding:1rem 0; background:#111827; color:#fff; border-radius:8px; margin-bottom:1rem;}
  .strategy-section { border-radius:8px; padding:0.6rem; margin-bottom:1rem; background:#fff;}
  .button-container { display:flex; justify-content:center; margin:1.25rem 0;}
  .info-section { font-size:0.9rem; color:#374151; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>💱 StratForge</h1>
    <p>Advanced Quantitative Backtesting Platform</p>
</div>
""", unsafe_allow_html=True)

def image_to_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

with st.sidebar:
    st.markdown("### Profile")

    profile_img_path = os.path.join('assets', 'phto.png')

    if os.path.exists(profile_img_path):
        img_base64 = image_to_base64(profile_img_path)
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:12px; margin-bottom:16px; padding:12px; background:#e5e7eb; border-radius:8px;'>
            <img src='data:image/png;base64,{img_base64}' style='width:60px; height:60px; object-fit:cover; border-radius:50%;'/>
            <div>
                <div style='font-weight:bold; font-size:16px; color:#1f2937; margin-bottom:4px;'>Yash Gupta</div>
                <div style='font-size:12px; color:#6b7280;'>Quantitative Enthusiast</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='display:flex; align-items:center; gap:12px; margin-bottom:16px; padding:12px; background:#e5e7eb; border-radius:8px;'>
            <div style='width:60px; height:60px; background:linear-gradient(135deg, #5b8cff, #3b82f6); border-radius:50%; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold; font-size:24px;'>YG</div>
            <div>
                <div style='font-weight:bold; font-size:16px; color:#1f2937; margin-bottom:4px;'>Yash Gupta</div>
                <div style='font-size:12px; color:#6b7280;'>Quantitative Enthusiast</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
   
    st.markdown(
        f'<div style="text-align: center; margin-bottom: 20px;"><a href="{LINKEDIN_URL}" target="_blank" style="text-decoration:none; display: inline-flex; align-items: center; gap: 8px; background: #0a66c2; color: white; padding: 8px 16px; border-radius: 6px; font-weight: 600; transition: all 0.3s;"><span>Connect on</span><span style="background: white; color: #0a66c2; border-radius: 3px; padding: 2px 6px; font-weight: 700;">in</span></a></div>',
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.markdown("### Data Source")
    data_source = st.radio(
        "Choose Data:",
        ["Sample Data", "Upload CSV"],
        key="data_source"
    )
    # Preloaded dataset selection when using sample data
    if data_source == "Sample Data":
        dataset_choice = st.selectbox(
            "Preloaded Dataset:",
            ["S&P 500 (SPY)", "Crypto (BTC-USD - simulated)", "Forex (EURUSD - simulated)", "Multi-Asset Demo"],
            key="dataset_choice"
        )
        if dataset_choice == "Multi-Asset Demo":
            num_assets = st.slider("Number of simulated assets", min_value=2, max_value=20, value=5, step=1, key="num_assets")
        else:
            num_assets = 1
    else:
        dataset_choice = None
        num_assets = 1
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=['csv'],
            help="Columns: date, open, high, low, close, volume"
        )
    else:
        uploaded_file = None
    st.markdown("---")
    st.markdown("### Period")
    today = datetime.now()
    start_date = st.date_input(
        "Start Date",
        value=datetime(2015, 1, 1),
        min_value=datetime(2015, 1, 1),
        max_value=today,
        key="start_date"
    )
    end_date = st.date_input(
        "End Date", 
        value=today,
        min_value=datetime(2015, 1, 1),
        max_value=today,
        key="end_date"
    )
    st.markdown("---")
    st.markdown("### Trading Parameters")
    initial_capital = st.slider(
        "Initial Capital ($)",
        min_value=1000,
        max_value=1000000,
        value=100000,
        step=1000,
        key="initial_capital"
    )
    slippage = st.slider(
        "Slippage (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        key="slippage"
    )
    transaction_cost = st.slider(
        "Transaction Cost (%)",
        min_value=0.0,
        max_value=5.0,
        value=0.1,
        step=0.01,
        key="transaction_cost",
        help="Transaction cost as percentage of trade value (e.g. 0.1 = 0.1%)"
    )
    st.markdown("---")
    st.markdown("### ⚙️ Execution Options")
    force_cpp = st.checkbox("Force use of C++ core if available (falls back to Python on error)", value=False)
    st.markdown("---")
st.markdown("""
<div class="strategy-section">
    <h2 style="color: #1f2937;">Strategy Configuration</h2>
</div>
""", unsafe_allow_html=True)
strategy_type = st.radio(
    "**Strategy Type:**",
    ["Built-in Strategies", "Custom Strategy"],
    key="strategy_type"
)
if strategy_type == "Built-in Strategies":
    strategy_name = st.selectbox(
        "Select Strategy:",
        ["Momentum Strategy", "Mean Reversion Strategy", "Black Scholes Strategy", "Volatility Scaling Trend"],
        key="strategy_name"
    )
    if strategy_name == "Momentum Strategy":
        st.info("📊 Momentum: Buys when price breaks above moving average")
    elif strategy_name == "Mean Reversion Strategy":
        st.info("📊 Mean Reversion: Buys when price is below lower Bollinger Band")
    elif strategy_name == "Black Scholes Strategy":
        st.info("⚖️ Black-Scholes Hybrid: switches between mean-reversion and trend-following based on realized volatility")
    elif strategy_name == "Volatility Scaling Trend":
        st.info("📈 Volatility-Scaled Trend: momentum with position sizing targeting a fixed volatility")
    st.session_state.use_custom_strategy = False
elif strategy_type == "Custom Strategy":
    st.markdown("###  Custom Strategy")
    with st.expander("📋 Template", expanded=False):
        st.markdown("""
```python
from strategies.base_strategy import BaseStrategy
import pandas as pd

class MyCustomStrategy(BaseStrategy):
    def __init__(self, name=None):
        super().__init__(name)
        # add your parameters here, e.g. self.lookback = 20

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Return a pandas Series of position weights (-1, 0, 1) indexed by data.index
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        signals = pd.Series(0, index=data.index)
        # Example: flat strategy (replace with your logic)
        return signals
```
""")
    custom_strategy_code = st.text_area(
        "Strategy Code:",
        value="",
        height=300,
        help="Paste your strategy code here"
    )
    if st.button("🔧 Load Strategy", type="secondary", width="stretch"):
        try:
            with open("temp_custom_strategy.py", "w") as f:
                f.write(custom_strategy_code)
            import importlib
            import temp_custom_strategy
            importlib.reload(temp_custom_strategy)
            strategy_class = None
            for name, obj in vars(temp_custom_strategy).items():
                if (isinstance(obj, type) and
                    hasattr(obj, 'generate_signals') and
                    name != 'BaseStrategy'):
                    strategy_class = obj
                    break
            if strategy_class is None:
                st.error(" No valid strategy class found")
                st.session_state.use_custom_strategy = False
            else:
                st.success(f" {strategy_class.__name__} loaded!")
                st.session_state.custom_strategy_class = strategy_class
                st.session_state.use_custom_strategy = True
        except Exception as e:
            st.error(f" Error: {str(e)}")
            st.session_state.use_custom_strategy = False
with st.expander("💡 Tips", expanded=False):
    st.markdown("""
    - tweak parameters with sliders on the sidebar    
    - use C++ core for faster execution and you can force it in 'Execution Options' on the sidebar                   
    """)
st.markdown('<div class="button-container">', unsafe_allow_html=True)
if st.button("Run Backtest", type="primary"):
    try:
        data_loader = DataLoader()
        if data_source == "Sample Data" or uploaded_file is None:
            data = data_loader.load_sample_data(start_date=str(start_date), end_date=str(end_date), dataset_choice=dataset_choice, num_assets=num_assets)
        else:
            data = data_loader.load_csv_data(uploaded_file, start_date=str(start_date), end_date=str(end_date))
        if len(data) == 0:
            st.error(" No data in selected period")
            st.stop()
        if strategy_type == "Custom Strategy" and st.session_state.get('use_custom_strategy', False) and 'custom_strategy_class' in st.session_state:
            strategy_class = st.session_state.custom_strategy_class
            strategy_instance = strategy_class()
        else:
            if strategy_name == "Momentum Strategy":
                strategy_instance = MomentumStrategy()
            elif strategy_name == "Mean Reversion Strategy":
                strategy_instance = MeanReversionStrategy()
            elif strategy_name == "Black Scholes Strategy":
                strategy_instance = BlackScholesStrategy()
            elif strategy_name == "Volatility Scaling Trend":
                strategy_instance = VolatilityScalingTrend()
        backtester = QuantBacktester(
            initial_capital=initial_capital,
            slippage=slippage/100.0,
            transaction_costs=transaction_cost/100.0,
            verbose=True,
            max_pct_per_trade=0.15,
            stop_loss_pct=0.08,
            take_profit_pct=0.25,
            force_cpp=force_cpp
        )
        with st.spinner("🔄 Running backtest..."):
            results = backtester.run_backtest(
                strategy=strategy_instance,
                data=data,
                output_dir="results"
            )
        st.success("✅ Backtest completed!")
        st.session_state.backtest_results = results
        st.session_state.backtest_completed = True

        try:
            backend = results.get('backend', 'Python')
            exec_time = results.get('execution_time', None)
            perf = results.get('performance_metrics', {})
            logger = logging.getLogger('streamlit_app')
            if exec_time is not None:
                logger.info(f"Backtest completed. Backend={backend}, ExecutionTime={exec_time:.3f}s")
            else:
                logger.info(f"Backtest completed. Backend={backend}")
            try:
                summary_parts = [f"[Backtest] Backend: {backend}"]
                if exec_time is not None:
                    summary_parts.append(f"Execution Time: {exec_time:.3f}s")
                summary_parts.append(f"Total Return: {_fmt_pct(perf.get('total_return'))}")
                summary_parts.append(f"Sharpe: {_fmt_float(perf.get('sharpe_ratio'), 2)}")
                summary_parts.append(f"Max DD: {_fmt_pct(perf.get('max_drawdown'))}")
                summary_parts.append(f"Trades: {_fmt_int(perf.get('total_trades'))}")
                print(' | '.join(summary_parts))
                cpp_metrics = results.get('cpp_metrics')
                if cpp_metrics:
                    print(f"[Backtest][C++] Extra metrics: {cpp_metrics}")
                cpp_err = results.get('cpp_error')
                if cpp_err:
                    print(f"[Backtest][C++] Error captured: {cpp_err}")
            except Exception as e:
                logger.warning(f"Could not print backtest summary to terminal: {e}")
        except Exception:
            pass
    except Exception as e:
        st.error(f" Error: {str(e)}")
        st.session_state.backtest_completed = False
st.markdown('</div>', unsafe_allow_html=True)
if st.session_state.get('backtest_completed', False):
    st.markdown("---")
    results = st.session_state.backtest_results
    backend = results.get('backend', 'Python')
    exec_time = results.get('execution_time', None)
    if exec_time is not None:
        st.info(f"Backend: {backend}  |  Execution time: {exec_time:.3f}s")
    else:
        st.info(f"Backend: {backend}")
    if results.get('cpp_available'):
        if results.get('backend') == 'Python' and results.get('cpp_error'):
            st.warning(f"C++ core available but failed at runtime. Error: {results.get('cpp_error')}")
        else:
            st.success("C++ core is available and was used." if results.get('backend') == 'C++' else "C++ core is available but Python backend was used")

    tab1, tab2, tab3, tab4 = st.tabs([" Performance", " Risk", " Charts", " Download"])
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_return = results['performance_metrics'].get('total_return')
            if total_return is not None:
                delta_val = None
                try:
                    dollar_change = float(total_return) * float(initial_capital)
                    if dollar_change >= 0:
                        delta_val = f"+${abs(dollar_change):,.0f}"
                    else:
                        delta_val = f"-${abs(dollar_change):,.0f}"
                except Exception:
                    delta_val = "N/A"
                st.metric(
                    "Total Return",
                    _fmt_pct(total_return),
                    delta=delta_val
                )
            else:
                st.metric("Total Return", "N/A")
        with col2:
            annualized_return = results['performance_metrics'].get('annualized_return')
            if annualized_return is not None:
                st.metric("Annualized Return", _fmt_pct(annualized_return))
            else:
                st.metric("Annualized Return", "N/A")
        with col3:
            sharpe_ratio = results['performance_metrics'].get('sharpe_ratio')
            if sharpe_ratio is not None:
                st.metric("Sharpe Ratio", _fmt_float(sharpe_ratio, 2))
            else:
                st.metric("Sharpe Ratio", "N/A")
        with col4:
            max_drawdown = results['performance_metrics'].get('max_drawdown')
            if max_drawdown is not None:
                st.metric("Max Drawdown", _fmt_pct(max_drawdown))
            else:
                st.metric("Max Drawdown", "N/A")
        st.markdown("#### Detailed Performance")
        col1, col2 = st.columns(2)
        with col1:
            final_value = results['performance_metrics'].get('final_value')
            st.write(f"• **Final Portfolio Value**: ${_fmt_float(final_value, 2)}")
            total_trades = results['performance_metrics'].get('total_trades')
            st.write(f"• **Total Trades**: {_fmt_int(total_trades)}")
            win_rate = results['performance_metrics'].get('win_rate')
            st.write(f"• **Win Rate**: {_fmt_pct(win_rate)}")
        with col2:
            avg_win = results['performance_metrics'].get('average_win')
            # prefer C++ computed metric when available
            if avg_win in (None, 0.0) and results.get('cpp_metrics_dict'):
                avg_win = results['cpp_metrics_dict'].get('average_win', avg_win)
            st.write(f"• **Average Win**: ${_fmt_float(avg_win, 2)}")
            avg_loss = results['performance_metrics'].get('average_loss')
            if avg_loss in (None, 0.0) and results.get('cpp_metrics_dict'):
                avg_loss = results['cpp_metrics_dict'].get('average_loss', avg_loss)
            st.write(f"• **Average Loss**: ${_fmt_float(avg_loss, 2)}")
            profit_factor = results['performance_metrics'].get('profit_factor')
            if profit_factor in (None, 0.0) and results.get('cpp_metrics_dict'):
                profit_factor = results['cpp_metrics_dict'].get('profit_factor', profit_factor)
            st.write(f"• **Profit Factor**: {_fmt_float(profit_factor, 2)}")
    with tab2:
        st.markdown("### Risk Analysis")
        risk_metrics = results.get('risk_analysis', {})
        perf_metrics = results.get('performance_metrics', {})
        def _pick(*keys):
            for k in keys:
                v = risk_metrics.get(k)
                if v is not None:
                    return v
                v = perf_metrics.get(k)
                if v is not None:
                    return v
            return None
        var95_val = _pick('var_95', 'var_95_historical', 'var_95_parametric', 'var_95_modified')
        cvar95_val = _pick('cvar_95', 'es_95_historical')
        vol_val = _pick('volatility', 'annualized_volatility')
        if vol_val is None:
            dv = _pick('daily_volatility')
            vol_val = dv * np.sqrt(252) if dv is not None else None
        skew_val = _pick('skewness')
        kurt_val = _pick('excess_kurtosis', 'kurtosis')
        max_dd_dur_val = _pick('max_drawdown_duration')
        col1, col2, col3 = st.columns(3)
        with col1:
            #prefer C++ computed VaR/CVaR when available; also normalize to positive loss magnitudes
            cpp_md = results.get('cpp_metrics_dict') or {}
            cpp_var = cpp_md.get('var_95')
            cpp_cvar = cpp_md.get('cvar_95')
            def _norm_var_val(v):
                if v is None:
                    return None
                try:
                    return abs(float(v))
                except Exception:
                    return None

            display_var = _norm_var_val(cpp_var) if cpp_var is not None else var95_val
            display_cvar = _norm_var_val(cpp_cvar) if cpp_cvar is not None else cvar95_val

            st.metric("VaR (95%)", _fmt_pct(display_var))
            st.metric("CVaR (95%)", _fmt_pct(display_cvar))
        with col2:
            st.metric("Volatility", _fmt_pct(vol_val))
            st.metric("Skewness", _fmt_float(skew_val, 2))
        with col3:
            st.metric("Kurtosis", _fmt_float(kurt_val, 2))
            st.metric("Max DD Duration", f"{_fmt_int(max_dd_dur_val)} days")
    with tab3:
        st.markdown("###  Visualizations")
        equity_file = "results/equity_curve.csv"
        if os.path.exists(equity_file):
            equity_data = pd.read_csv(equity_file)
            equity_data['date'] = pd.to_datetime(equity_data['date'])
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=equity_data['date'],
                y=equity_data['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#5b8cff', width=2)
            ))
            fig_equity.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig_equity, width="stretch")
            if 'drawdown' in equity_data.columns:
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=equity_data['date'],
                    y=equity_data['drawdown'] * 100,
                    mode='lines',
                    name='Drawdown',
                    fill='tonexty',
                    line=dict(color='#ef4444', width=1)
                ))
                fig_dd.update_layout(
                    title="Drawdown Over Time",
                    xaxis_title="Date",
                    yaxis_title="Drawdown (%)",
                    template="plotly_dark",
                    height=300
                )
                st.plotly_chart(fig_dd, width="stretch")
        chart_files = [
            "results/equity_curve.png",
            "results/drawdown.png",
            "results/performance_dashboard.png",
            "results/return_distribution.png",
            "results/risk_analysis.png",
            "results/rolling_performance.png",
            "results/trade_analysis.png"
        ]
        for chart_file in chart_files:
            if os.path.exists(chart_file):
                st.image(chart_file, caption=chart_file.split('/')[-1].replace('.png', '').replace('_', ' ').title(), width="stretch")
    with tab4:
        st.markdown("###  Download Results")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("results/equity_curve.csv"):
                with open("results/equity_curve.csv", "rb") as file:
                    st.download_button(
                        label="📊 Download Equity Curve (CSV)",
                        data=file.read(),
                        file_name="equity_curve.csv",
                        mime="text/csv"
                    )
            if os.path.exists("results/trades.csv"):
                with open("results/trades.csv", "rb") as file:
                    st.download_button(
                        label="📈 Download Trades (CSV)",
                        data=file.read(),
                        file_name="trades.csv",
                        mime="text/csv"
                    )
        with col2:
            if os.path.exists("results/performance.json"):
                with open("results/performance.json", "rb") as file:
                    st.download_button(
                        label="📋 Download Performance (JSON)",
                        data=file.read(),
                        file_name="performance.json",
                        mime="application/json"
                    )
            if os.path.exists("results/performance_dashboard.png"):
                with open("results/performance_dashboard.png", "rb") as file:
                    st.download_button(
                        label="📊 Download Dashboard (PNG)",
                        data=file.read(),
                        file_name="performance_dashboard.png",
                        mime="image/png"
                    )
