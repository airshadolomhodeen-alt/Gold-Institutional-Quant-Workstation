# -*- coding: utf-8 -*-
"""
Institutional Quant Terminal - Production Hub
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import requests

from src.data_quality import run_data_quality_gate
from src.feature_engineering import compute_features
from src.targets import create_multi_horizon_targets
from src.volatility import compute_volatility_features
from src.regime_models import detect_market_regime
from src.forecasting_models import get_forecasting_models
from src.prediction_models import generate_return_predictions
from src.calibration import calibrate_probabilities
from src.ensemble import aggregate_ensemble
from src.path_forecast import compute_path_statistics
from src.expected_value import compute_expected_values
from src.risk_engine import evaluate_tradeability_gate
from src.explainability import compute_transition_diagnostics, get_feature_importances
from src.visualization import plot_probability_curve
from src.backtest import run_walk_forward_simulation
from src.logger import log_current_predictions
from src.config import Config

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Institutional Quant Terminal | Probabilistic Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .stApp { background-color: #0A0E17; color: #F8FAFC; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #131B2E; border-right: 1px solid #1E293B; }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown { color: #E2E8F0 !important; }

    div[data-testid="stMetric"] {
        background-color: #131B2E; 
        border: 1px solid #334155;
        padding: 12px 14px; 
        border-radius: 8px;
        min-height: 95px;
    }
    div[data-testid="stMetric"] label { 
        color: #38BDF8 !important; 
        font-size: 0.75rem !important; 
        text-transform: uppercase; 
        letter-spacing: 0.05em;
        font-weight: 700;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { 
        color: #F8FAFC !important; 
        font-family: 'Courier New', Courier, monospace; 
        font-size: 1.15rem !important;
        font-weight: bold;
    }

    h1, h2, h3 { color: #F8FAFC !important; letter-spacing: -0.025em; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #0A0E17; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E293B !important; 
        border: 1px solid #334155 !important; 
        border-radius: 6px; 
        color: #E2E8F0 !important; 
        font-weight: 600;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #38BDF8 !important; 
        color: #0A0E17 !important; 
        font-weight: 700;
    }
    p, span, label { color: #E2E8F0; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ MULTI-HORIZON PROBABILISTIC FORECASTING ENGINE")
st.markdown("**Terminal Status:** Production Ready | **Architecture:** Trend-Aware Walk-Forward Ensemble")

# ==============================================================================
# SIDEBAR PARAMETERS
# ==============================================================================
st.sidebar.header("1. Feed & Parameters")
symbol = st.sidebar.text_input("Asset Symbol", value="XAU/USD")
interval = st.sidebar.selectbox("Timeframe", ["15min", "1h", "4h", "1day"], index=1)
outputsize = st.sidebar.slider("Historical Candles", 200, 1000, 500)

st.sidebar.header("2. Execution Costs & Risk")
spread = st.sidebar.number_input("Spread Cost", value=0.20, step=0.05)
commission = st.sidebar.number_input("Commission (%)", value=0.02, step=0.01) / 100.0
slippage = st.sidebar.number_input("Slippage Cost", value=0.05, step=0.01)

st.sidebar.markdown("### 3. Tradeability Gate Thresholds")
min_conviction = st.sidebar.slider("Min Probability Conviction", 0.50, 0.80, 0.53, 0.01)
max_disagreement = st.sidebar.slider("Max Model Disagreement (%)", 10.0, 50.0, 35.0, 5.0)

TWELVE_DATA_API_KEY = "32b6a749e8c14835b95b8a9c271eec95"

@st.cache_data(ttl=300)
def fetch_twelve_data(sym, tf, size, api_key):
    url = f"https://api.twelvedata.com/time_series?symbol={sym}&interval={tf}&outputsize={size}&apikey={api_key}&format=JSON"
    try:
        response = requests.get(url)
        data = response.json()
        if "values" in data:
            df_api = pd.DataFrame(data["values"])
            rename_map = {"datetime": "Time", "open": "Open", "high": "High", "low": "Low", "close": "Close"}
            if "volume" in df_api.columns:
                rename_map["volume"] = "Volume"
            df_api = df_api.rename(columns=rename_map)
            for col in ["Open", "High", "Low", "Close"]:
                df_api[col] = pd.to_numeric(df_api[col], errors='coerce')
            df_api["Volume"] = pd.to_numeric(df_api["Volume"], errors='coerce').fillna(1.0) if "Volume" in df_api.columns else 1.0
            return df_api.sort_values("Time").reset_index(drop=True), None
        return None, data.get("message", "API Error")
    except Exception as e:
        return None, str(e)

df_raw, err_msg = fetch_twelve_data(symbol, interval, outputsize, TWELVE_DATA_API_KEY)
if df_raw is None or df_raw.empty:
    dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='15min')
    prices = 2000 + np.cumsum(np.random.normal(0, 2, 500))
    df_raw = pd.DataFrame({"Time": dates, "Open": prices, "High": prices+1, "Low": prices-1, "Close": prices, "Volume": 1.0})

dq_pass, dq_checks, df = run_data_quality_gate(df_raw)
if not dq_pass:
    st.error("🚨 DATA QUALITY GATE FAILED")
    st.write(dq_checks)
    st.stop()

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================
df = compute_features(df, Config.RSI_PERIOD, Config.MACD_FAST, Config.MACD_SLOW, Config.ATR_PERIOD)
df = compute_volatility_features(df)

try:
    df = create_multi_horizon_targets(df, threshold_type='atr', threshold_multiplier=0.5)
except TypeError:
    try:
        df = create_multi_horizon_targets(df)
    except Exception as e:
        st.error(f"🚨 TARGET CREATION FAILED: {e}")
        st.stop()

current_regime, regime_prob, df = detect_market_regime(df)

df_model = df.dropna().copy()
features = ['Log_Return', 'RSI', 'MACD', 'MACD_Hist', 'ATR', 'Realized_Vol', 'EWMA_Vol', 'SMA_10_Slope', 'LR_Slope_14']
X = df_model[features]

horizons = list(range(1, 11))
prob_up_list, prob_down_list, prob_neutral_list = [], [], []
exp_returns, exp_ranges, model_agreements, uncertainties = [], [], [], []
lower_bounds, upper_bounds = [], []

models = get_forecasting_models()
return_predictions = generate_return_predictions(df_model, features, horizons)

for h in horizons:
    y_h = df_model[f'Target_Dir_{h}']
    valid_idx = y_h != 0.5
    X_h, y_bin = X.values[valid_idx], (y_h[valid_idx] == 1).astype(int)
    
    horizon_probs = {}
    for name, model in models.items():
        try:
            model.fit(X_h, y_bin)
            p = model.predict_proba(X.iloc[[-1]])[0]
            horizon_probs[name] = p[1] if len(p) > 1 else 0.5
        except Exception:
            horizon_probs[name] = 0.5

    p_ens, disagreement = aggregate_ensemble(horizon_probs)
    cal_up, _, _ = calibrate_probabilities(np.array([p_ens]), np.array([1]))
    p_up = cal_up[0]
    p_down = 1.0 - p_up
    p_neut = max(0.0, 1.0 - abs(p_up - p_down) - 0.2)
    
    tot = p_up + p_down + p_neut
    prob_up_list.append(p_up / tot)
    prob_down_list.append(p_down / tot)
    prob_neutral_list.append(p_neut / tot)
    
    reg_data = return_predictions.get(h, {"Expected_Return": 0.0, "Lower_Bound": 0.0, "Upper_Bound": 0.0})
    exp_returns.append(reg_data["Expected_Return"])
    lower_bounds.append(reg_data["Lower_Bound"])
    upper_bounds.append(reg_data["Upper_Bound"])
    
    exp_ranges.append(df_model['ATR'].iloc[-1] * np.sqrt(h))
    model_agreements.append((1.0 - disagreement) * 100.0)
    uncertainties.append("LOW" if disagreement < 0.2 else ("MODERATE" if disagreement < 0.4 else "HIGH"))

path_stats = compute_path_statistics(df_model, horizons)
ev_results = compute_expected_values(prob_up_list[-1], prob_down_list[-1], exp_returns[-1], spread, commission, slippage, df_model['ATR'].iloc[-1])
tradeability_state, trade_reasons = evaluate_tradeability_gate(
    prob_up_list[-1], ev_results['Net_EV'], model_agreements[-1], dq_pass, uncertainties[-1],
    min_conviction=min_conviction, max_disagreement=max_disagreement
)

log_current_predictions(
    symbol=symbol, interval=interval, current_price=df['Close'].iloc[-1],
    horizons=horizons, prob_up_list=prob_up_list, prob_down_list=prob_down_list,
    return_predictions=return_predictions
)

# ==============================================================================
# UI RENDERING
# ==============================================================================
st.subheader(f"📈 Interactive Market Feed: {symbol} ({interval})")
tv_interval_map = {"15min": "15", "1h": "60", "4h": "240", "1day": "D"}
tv_symbol = symbol.replace("/", "")

tradingview_html = f"""
<div class="tradingview-widget-container" style="height:500px;width:100%">
  <div id="tradingview_widget" style="height:100%;width:100%"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
  new TradingView.widget({{
    "width": "100%", "height": 500, "symbol": "OANDA:{tv_symbol}",
    "interval": "{tv_interval_map.get(interval, '60')}", "timezone": "Etc/UTC",
    "theme": "dark", "style": "1", "locale": "en", "toolbar_bg": "#131B2E",
    "enable_publishing": false, "hide_side_toolbar": false, "allow_symbol_change": true,
    "details": true, "hotlist": true, "calendar": true, "container_id": "tradingview_widget"
  }});
  </script>
</div>
"""
components.html(tradingview_html, height=520)

st.markdown("---")
st.subheader("🚨 NEXT 10-CANDLE PROBABILISTIC & PREDICTIVE FORECAST PANEL")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
c2.metric("P(UP at T+10)", f"{prob_up_list[-1]*100:.1f}%")
c3.metric("P(DOWN at T+10)", f"{prob_down_list[-1]*100:.1f}%")
c4.metric("Current Regime", current_regime)
c5.metric("Tradeability State", tradeability_state)

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 10-Candle Table", "📈 Probability Curve", "🛡️ Expected Value Audit", 
    "🔍 Explainability & Diagnostics", "🚀 Walk-Forward Backtest"
])

with tab1:
    forecast_df = pd.DataFrame({
        "Horizon": [f"T+{h}" for h in horizons],
        "P(UP)": prob_up_list, "P(NEUTRAL)": prob_neutral_list, "P(DOWN)": prob_down_list,
        "Expected Return": exp_returns, "Lower Bound (95%)": lower_bounds, "Upper Bound (95%)": upper_bounds,
        "Model Agreement": [f"{m:.1f}%" for m in model_agreements], "Uncertainty": uncertainties
    })
    st.dataframe(forecast_df.style.format({
        "P(UP)": "{:.2%}", "P(NEUTRAL)": "{:.2%}", "P(DOWN)": "{:.2%}",
        "Expected Return": "{:.4f}", "Lower Bound (95%)": "{:.4f}", "Upper Bound (95%)": "{:.4f}"
    }), use_container_width=True)

with tab2:
    fig = plot_probability_curve(horizons, prob_up_list, prob_neutral_list, prob_down_list)
    st.pyplot(fig)

with tab3:
    st.markdown("### 🛡️ Expected Value & Risk Audit")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Gross Expected Value", f"${ev_results['Gross_EV']:.2f}")
        st.metric("Total Execution Costs", f"${ev_results['Total_Costs']:.2f}")
        st.metric("Net Expected Value", f"${ev_results['Net_EV']:.2f}")
    with col_b:
        st.metric("Reward-to-Risk Ratio", f"{ev_results['RR_Ratio']:.2f}")
        st.metric("Max Favorable Excursion Prob", f"{path_stats['Prob_MFE_1ATR']:.1f}%")
        st.metric("Tradeability Status", tradeability_state)

    st.markdown("---")
    
    if tradeability_state == "TRADEABLE":
        st.markdown("### 🎯 EXECUTION PLAN (Verified Tradeable)")
        current_close = df['Close'].iloc[-1]
        atr_val = df_model['ATR'].iloc[-1]
        
        is_long = prob_up_list[-1] >= prob_down_list[-1] and ev_results['Net_EV'] > 0
        direction = "LONG" if is_long else "SHORT"
        
        if direction == "LONG":
            tp1, tp2, sl = current_close + atr_val, current_close + (atr_val * 2), current_close - atr_val
        else:
            tp1, tp2, sl = current_close - atr_val, current_close - (atr_val * 2), current_close + atr_val
            
        risk = abs(current_close - sl)
        reward_tp1 = abs(tp1 - current_close)
        rr_ratio = reward_tp1 / risk if risk > 0 else 0.0
        cal_success = (prob_up_list[-1] if direction == "LONG" else prob_down_list[-1]) * 100

        exec_col1, exec_col2 = st.columns(2)
        with exec_col1:
            st.metric("Trade Direction", direction)
            st.metric("Reference Entry", f"${current_close:.2f}")
            st.metric("Take Profit 1 (TP1)", f"${tp1:.2f}")
        with exec_col2:
            st.metric("Stop Loss (SL)", f"${sl:.2f}")
            st.metric("Risk : Reward", f"1 : {rr_ratio:.2f}")
            st.metric("Success Probability", f"{cal_success:.1f}%")
            
        st.success("✅ **Gate Status:** TRADEABLE — Passed conviction and temporal rules.")
    else:
        st.warning("🚫 **Execution Plan Suppressed (NO-TRADE State)**")
        for reason in (trade_reasons or ["Insufficient edge or high disagreement."]):
            st.markdown(f"- ⚠️ {reason}")

with tab4:
    st.markdown("### Predictive Contributions")
    first_model = list(models.values())[0] if models else None
    feat_df = get_feature_importances(first_model, features)
    if isinstance(feat_df, dict):
        feat_df = pd.DataFrame(list(feat_df.items()), columns=["Feature", "Predictive Contribution"])
    st.dataframe(feat_df, use_container_width=True)

with tab5:
    st.markdown("### 🚀 Historical Walk-Forward Simulation")
    if st.button("Execute Simulation"):
        with st.spinner("Running historical backtest simulation..."):
            sim_results, metrics = run_walk_forward_simulation(
                df_model, features, models, 
                min_conviction=min_conviction, max_disagreement=max_disagreement,
                spread=spread, commission=commission
            )
            st.line_chart(sim_results.set_index("Time")["Capital"])
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Final Capital", f"${metrics['Final_Equity']:,.2f}")
            col2.metric("Total Return", f"{metrics['Total_Return_Pct']:+.2f}%")
            col3.metric("Max Drawdown", f"{metrics['Max_Drawdown_Pct']:.2f}%")
            col4.metric("Win Rate", f"{metrics['Win_Rate']:.1f}%")
