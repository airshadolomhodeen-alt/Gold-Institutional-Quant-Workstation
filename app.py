# -*- coding: utf-8 -*-
"""
Institutional Quant Workstation - Root Application Hub
Imports modularized engines from src/ and renders the Streamlit Terminal.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Import modular backend files from src/
from src.validation import run_data_quality_gate
from src.feature_engineering import compute_features
from src.targets import create_multi_horizon_targets
from src.regime_models import detect_market_regime
from src.forecasting_models import get_forecasting_model
from src.calibration import calibrate_probabilities
from src.risk_engine import evaluate_trading_decision
from src.visualization import plot_probability_curve

# ==============================================================================
# STREAMLIT PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Institutional Quant Terminal | 10-Candle Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .stApp { background-color: #0A0E17; color: #E2E8F0; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #131B2E; border-right: 1px solid #1E293B; }
    div[data-testid="stMetric"] {
        background-color: #131B2E; border: 1px solid #1E293B;
        padding: 15px 20px; border-radius: 6px;
    }
    div[data-testid="stMetric"] label { color: #94A3B8 !important; font-size: 0.75rem; text-transform: uppercase; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #F8FAFC !important; font-family: 'Courier New', Courier, monospace; }
    h1, h2, h3 { color: #F8FAFC; letter-spacing: -0.025em; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #0A0E17; }
    .stTabs [data-baseweb="tab"] {
        background-color: #131B2E; border: 1px solid #1E293B; border-radius: 4px; color: #94A3B8; font-weight: 600;
    }
    .stTabs [aria-selected="true"] { background-color: #00FF66 !important; color: #0A0E17 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ INSTITUTIONAL QUANT WORKSTATION: 10-CANDLE PROBABILISTIC ENGINE")
st.markdown("**Terminal Status:** Secure Connection Active | **Architecture:** Modular `src/` Pipeline")

# ==============================================================================
# 1. DATA INGESTION & DATA QUALITY GATE
# ==============================================================================
st.sidebar.header("1. Data & Execution Parameters")
uploaded_file = st.sidebar.file_uploader("Upload Market Feed CSV Data", type=["csv", "txt"])

@st.cache_data
def load_data(file_bytes=None):
    if file_bytes is not None:
        content = file_bytes.getvalue().decode("utf-8", errors="ignore")
        lines = content.splitlines()
        rows = []
        for line in lines[1:]:
            parts = line.split("\t") if "\t" in line else line.split(",")
            if len(parts) >= 6:
                rows.append([parts[0].strip().replace('"', ""), parts[1].strip(), parts[2].strip(), parts[3].strip(), parts[4].strip(), parts[-1].strip()])
        return pd.DataFrame(rows, columns=["Time", "Open", "High", "Low", "Close", "Volume"])
    else:
        # Fallback dummy frame for immediate UI testing if no file uploaded
        dates = pd.date_range(end=pd.Timestamp.now(), periods=500, freq='15min')
        prices = 2000 + np.cumsum(np.random.normal(0, 2, 500))
        return pd.DataFrame({"Time": dates, "Open": prices, "High": prices+1, "Low": prices-1, "Close": prices, "Volume": 500})

df_raw = load_data(uploaded_file)

# Run Data Quality Gate (from src/validation.py)
dq_pass, dq_checks, df = run_data_quality_gate(df_raw)

if not dq_pass:
    st.error("🚨 DATA QUALITY GATE FAILED: Review anomalies below.")
    st.write(dq_checks)
    st.stop()
else:
    st.sidebar.success("Data Quality Gate: PASS ✅")

# ==============================================================================
# 2. HYPERPARAMETERS
# ==============================================================================
rsi_per = st.sidebar.slider("RSI Lookback Period", 3, 25, 14)
col_f, col_s = st.sidebar.columns(2)
macd_f = col_f.slider("MACD Fast", 3, 15, 12)
macd_s = col_s.slider("MACD Slow", 10, 30, 26)
atr_per = st.sidebar.slider("ATR Period", 5, 30, 14)
neutral_band = st.sidebar.slider("Neutral Band Threshold", 0.0, 0.5, 0.33, 0.05)
model_type = st.sidebar.selectbox("Quant Model Engine", ["Gradient Boosting", "Random Forest", "L2 Regularized Logistic"])

spread_ticks = st.sidebar.number_input("Spread (USD)", value=0.20, step=0.05)
commission_pct = st.sidebar.number_input("Commission (%)", value=0.02, step=0.01) / 100.0

# ==============================================================================
# 3. PIPELINE EXECUTION (Features, Targets, Regimes)
# ==============================================================================
df = compute_features(df, rsi_per, macd_f, macd_s, atr_per)
df = create_multi_horizon_targets(df)
current_regime, regime_prob, df = detect_market_regime(df)

df_model = df.dropna().copy()
features = ['Log_Return', 'RSI', 'MACD', 'MACD_Hist', 'ATR', 'Realized_Vol']
X = df_model[features]

# ==============================================================================
# 4. 10-CANDLE FORECAST GENERATION
# ==============================================================================
horizons = list(range(1, 11))
prob_up, prob_down, prob_neutral, exp_returns, exp_ranges = [], [], [], [], []

clf = get_forecasting_model(model_type)

for h in horizons:
    y_h = df_model[f'Target_Dir_{h}']
    valid_idx = y_h != 0.5
    X_h, y_bin = X.values[valid_idx], (y_h[valid_idx] == 1).astype(int)
    
    if len(np.unique(y_bin)) > 1:
        clf.fit(X_h, y_bin)
        probs = clf.predict_proba(X.iloc[[-1]])[0]
        p_up = probs[1] if len(probs) > 1 else 0.5
        p_down = probs[0] if len(probs) > 1 else 0.5
    else:
        p_up, p_down = 0.5, 0.5

    # Calibrate probabilities
    cal_up, _ = calibrate_probabilities(np.array([p_up]), np.array([1]))
    p_up = cal_up[0]
    p_down = 1.0 - p_up
    p_neut = max(0.0, 1.0 - abs(p_up - p_down) - 0.2)
    
    tot = p_up + p_down + p_neut
    prob_up.append(p_up / tot)
    prob_down.append(p_down / tot)
    prob_neutral.append(p_neut / tot)
    
    exp_returns.append(df_model[f'Forward_Return_{h}'].mean() * (p_up - p_down))
    exp_ranges.append(df_model['ATR'].iloc[-1] * np.sqrt(h))

forecast_df = pd.DataFrame({
    "Horizon": [f"t+{h}" for h in horizons],
    "P(UP)": prob_up,
    "P(NEUTRAL)": prob_neutral,
    "P(DOWN)": prob_down,
    "Expected Return": exp_returns,
    "Expected Range": exp_ranges
})

# Risk Evaluation
max_prob = max(prob_up[-1], prob_down[-1])
avg_win = df['ATR'].iloc[-1]
avg_loss = df['ATR'].iloc[-1] * 0.9
trading_state, ev = evaluate_trading_decision(max_prob, neutral_band, avg_win, avg_loss, spread_ticks, commission_pct, dq_pass)

# ==============================================================================
# 5. DASHBOARD UI RENDERING
# ==============================================================================
st.subheader("🚨 NEXT 10-CANDLE FORECAST PANEL")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
c2.metric("P(UP at t+10)", f"{prob_up[-1]*100:.1f}%")
c3.metric("P(DOWN at t+10)", f"{prob_down[-1]*100:.1f}%")
c4.metric("Current Regime", current_regime)
c5.metric("Trading State", trading_state)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Horizon Table", "📈 Probability Curve", "🛡️ Expected Value Audit"])

with tab1:
    st.dataframe(forecast_df.style.format({
        "P(UP)": "{:.2%}", "P(NEUTRAL)": "{:.2%}", "P(DOWN)": "{:.2%}",
        "Expected Return": "{:.4f}", "Expected Range": "{:.2f}"
    }), use_container_width=True)

with tab2:
    fig = plot_probability_curve(horizons, prob_up, prob_neutral, prob_down)
    st.pyplot(fig)

with tab3:
    st.write(f"**Net Expected Value after Costs:** ${ev:.2f}")
    st.write(f"**Decision Rule Triggered:** {trading_state}")
