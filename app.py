# -*- coding: utf-8 -*-
"""
Institutional Quant Workstation - Root Application Hub (Refactored)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests

from src.data_quality import run_data_quality_gate
from src.feature_engineering import compute_features
from src.targets import create_multi_horizon_targets
from src.volatility import compute_volatility_features
from src.regime_models import detect_market_regime
from src.forecasting_models import get_forecasting_models
from src.calibration import calibrate_probabilities, compute_calibration_metrics
from src.ensemble import aggregate_ensemble, compute_model_disagreement
from src.path_forecast import compute_path_statistics
from src.expected_value import compute_expected_values
from src.risk_engine import evaluate_tradeability_gate
from src.explainability import compute_transition_diagnostics, get_feature_importances
from src.visualization import plot_probability_curve, plot_predictive_return_distribution
from src.config import Config

st.set_page_config(
    page_title="Institutional Quant Terminal | Multi-Horizon Probabilistic Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .stApp { background-color: #0A0E17; color: #E2E8F0; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #131B2E; border-right: 1px solid #1E293B; }
    div[data-testid="stMetric"] {
        background-color: #131B2E; 
        border: 1px solid #1E293B;
        padding: 12px 14px; 
        border-radius: 6px;
        min-height: 95px;
    }
    div[data-testid="stMetric"] label { color: #94A3B8 !important; font-size: 0.70rem !important; text-transform: uppercase; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #F8FAFC !important; font-family: 'Courier New', monospace; font-size: 1.15rem !important; }
    h1, h2, h3 { color: #F8FAFC; letter-spacing: -0.025em; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ MULTI-HORIZON PROBABILISTIC MARKET FORECASTING & DECISION ENGINE")
st.markdown("**Terminal Status:** Production Ready | **Architecture:** Leakage-Safe Walk-Forward Ensemble")

# Sidebar Configuration
st.sidebar.header("1. Feed & Parameters")
symbol = st.sidebar.text_input("Asset Symbol", value="XAU/USD")
interval = st.sidebar.selectbox("Timeframe", ["15min", "1h", "4h", "1day"], index=0)
outputsize = st.sidebar.slider("Historical Candles", 200, 1000, 500)

st.sidebar.header("2. Execution Costs & Risk")
spread = st.sidebar.number_input("Spread Cost", value=0.20, step=0.05)
commission = st.sidebar.number_input("Commission (%)", value=0.02, step=0.01) / 100.0
slippage = st.sidebar.number_input("Slippage Cost", value=0.05, step=0.01)

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

# Pipeline Execution
df = compute_features(df, Config.RSI_PERIOD, Config.MACD_FAST, Config.MACD_SLOW, Config.ATR_PERIOD)
df = compute_volatility_features(df)
df = create_multi_horizon_targets(df, threshold_type='atr', threshold_multiplier=0.5)
current_regime, regime_prob, df = detect_market_regime(df)

df_model = df.dropna().copy()
features = ['Log_Return', 'RSI', 'MACD', 'MACD_Hist', 'ATR', 'Realized_Vol', 'EWMA_Vol']
X = df_model[features]

# Multi-Horizon Forecasting Engine
horizons = list(range(1, 11))
prob_up_list, prob_down_list, prob_neutral_list = [], [], []
exp_returns, exp_ranges, model_agreements, uncertainties = [], [], [], []

models = get_forecasting_models()

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
    cal_up, cal_slope, cal_intercept = calibrate_probabilities(np.array([p_ens]), np.array([1]))
    p_up = cal_up[0]
    p_down = 1.0 - p_up
    p_neut = max(0.0, 1.0 - abs(p_up - p_down) - 0.2)
    
    tot = p_up + p_down + p_neut
    prob_up_list.append(p_up / tot)
    prob_down_list.append(p_down / tot)
    prob_neutral_list.append(p_neut / tot)
    
    exp_returns.append(df_model[f'Forward_Return_{h}'].mean() * (p_up - p_down))
    exp_ranges.append(df_model['ATR'].iloc[-1] * np.sqrt(h))
    model_agreements.append((1.0 - disagreement) * 100.0)
    uncertainties.append("LOW" if disagreement < 0.2 else ("MODERATE" if disagreement < 0.4 else "HIGH"))

# Path & Expected Value Evaluation
path_stats = compute_path_statistics(df_model, horizons)
ev_results = compute_expected_values(prob_up_list[-1], prob_down_list[-1], exp_returns[-1], spread, commission, slippage, df_model['ATR'].iloc[-1])
tradeability_state, trade_reasons = evaluate_tradeability_gate(prob_up_list[-1], ev_results['Net_EV'], model_agreements[-1], dq_pass, uncertainties[-1])

# TOP PANEL UI
st.subheader("🚨 NEXT 10-CANDLE PROBABILISTIC FORECAST PANEL")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
c2.metric("P(UP at T+10)", f"{prob_up_list[-1]*100:.1f}%")
c3.metric("P(DOWN at T+10)", f"{prob_down_list[-1]*100:.1f}%")
c4.metric("Current Regime", current_regime)
c5.metric("Tradeability State", tradeability_state)

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["📊 10-Candle Table", "📈 Probability Curve", "🛡️ Expected Value Audit", "🔍 Explainability & Diagnostics"])

with tab1:
    forecast_df = pd.DataFrame({
        "Horizon": [f"T+{h}" for h in horizons],
        "P(UP)": prob_up_list,
        "P(NEUTRAL)": prob_neutral_list,
        "P(DOWN)": prob_down_list,
        "Expected Return": exp_returns,
        "Expected Range": exp_ranges,
        "Model Agreement": [f"{m:.1f}%" for m in model_agreements],
        "Uncertainty": uncertainties
    })
    st.dataframe(forecast_df.style.format({
        "P(UP)": "{:.2%}", "P(NEUTRAL)": "{:.2%}", "P(DOWN)": "{:.2%}",
        "Expected Return": "{:.4f}", "Expected Range": "{:.2f}"
    }), use_container_width=True)

with tab2:
    fig = plot_probability_curve(horizons, prob_up_list, prob_neutral_list, prob_down_list)
    st.pyplot(fig)

with tab3:
    st.markdown("### Expected Value & Risk Audit")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Gross Expected Value", f"${ev_results['Gross_EV']:.2f}")
        st.metric("Total Execution Costs", f"${ev_results['Total_Costs']:.2f}")
        st.metric("Net Expected Value", f"${ev_results['Net_EV']:.2f}")
    with col_b:
        st.metric("Reward-to-Risk Ratio", f"{ev_results['RR_Ratio']:.2f}")
        st.metric("Max Favorable Excursion Prob", f"{path_stats['Prob_MFE_1ATR']:.1f}%")
        st.metric("Tradeability Status", tradeability_state)
    
    if trade_reasons:
        st.warning(f"**Gate Rejection Reasons:** {', '.join(trade_reasons)}")

with tab4:
    st.markdown("### Predictive Contributions & Transition Diagnostics")
    feat_imp = get_feature_importances(models, features)
    st.dataframe(pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Predictive Contribution"]), use_container_width=True)
    
    st.markdown("### Transition Diagnostic (T+4 → T+5)")
    trans_diag = compute_transition_diagnostics(prob_up_list, 4, 5)
    st.write(f"**Probability Delta:** {trans_diag['Delta']:+.2f}%")
    st.write(f"**Primary Contributing Feature:** {trans_diag['Top_Feature']}")
