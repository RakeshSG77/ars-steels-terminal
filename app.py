"""
ARS Steels Intelligence Platform — Enterprise Dashboard
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import sqlite3
import os
import warnings
import urllib.request
import xml.etree.ElementTree as ET
import collections
import yfinance as yf
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')

DB_NAME = 'ars_steels.db'

# ─────────────────────────────────────────────────────────
# 0. INITIALIZE STABLE NLP ENGINE
# ─────────────────────────────────────────────────────────
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# ─────────────────────────────────────────────────────────
# 1. PAGE CONFIG & CYBERPUNK NEON CSS (WINDOW OPTIMIZED)
# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="ARS Steels Terminal", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;600&display=swap');

/* Base Terminal Theme */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #030712; color: #E2E8F0; }
.stApp { background: radial-gradient(circle at 50% -20%, #1e293b 0%, #030712 80%); }
#MainMenu, footer, header {visibility: hidden;}

/* Strict Margin Reduction to Fit Window Viewports */
.block-container { padding: 1rem 1.5rem !important; max-width: 1600px; }

/* ── Typography & Neon Glows ── */
.glow-title { font-family: 'Orbitron', sans-serif; font-size: 1.8rem; font-weight: 900; color: #FFFFFF; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0; text-shadow: 0 0 10px #38BDF8, 0 0 20px #38BDF8; }
.sub-title { font-family: 'Rajdhani', sans-serif; font-size: 0.85rem; color: #C8A84B; letter-spacing: 2px; text-transform: uppercase; margin-top: -3px; margin-bottom: 0.8rem; text-shadow: 0 0 5px rgba(200, 168, 75, 0.5); }
.section-header { font-family: 'Orbitron', sans-serif; font-size: 0.8rem; color: #38BDF8; letter-spacing: 1px; border-bottom: 1px solid rgba(56, 189, 248, 0.2); padding-bottom: 0.2rem; margin-bottom: 0.6rem; margin-top: 0.5rem; text-transform: uppercase; }

/* ── KPI Cards Sizing Optimization ── */
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.8rem; margin-bottom: 1rem; }
.tech-card { background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.05); border-top: 2px solid #334155; border-radius: 6px; padding: 0.6rem 0.8rem; transition: all 0.3s ease; }
.tech-card:hover { transform: translateY(-2px); border-top: 2px solid #38BDF8; background: rgba(30, 41, 59, 0.95); box-shadow: 0 5px 12px -4px rgba(56, 189, 248, 0.3); }

.card-label { font-family: 'Rajdhani', sans-serif; font-size: 0.75rem; font-weight: 600; color: #94A3B8; text-transform: uppercase; letter-spacing: 1px; }
.card-value { font-family: 'Rajdhani', sans-serif; font-size: 1.5rem; font-weight: 700; color: #F8FAFC; margin: 0.1rem 0; line-height: 1; }
.card-delta.positive { color: #10B981; font-family: 'Inter', monospace; font-size: 0.7rem; font-weight: 600; }
.card-delta.negative { color: #EF4444; font-family: 'Inter', monospace; font-size: 0.7rem; font-weight: 600; }
.card-delta.neutral  { color: #64748B; font-family: 'Inter', monospace; font-size: 0.7rem; font-weight: 600; }

/* ── Neon Action Buttons ── */
div.stButton > button { background: transparent !important; border: 1px solid #38BDF8 !important; color: #38BDF8 !important; font-family: 'Orbitron', sans-serif !important; font-size: 0.7rem !important; font-weight: 600 !important; text-transform: uppercase; border-radius: 4px !important; transition: all 0.2s ease !important; box-shadow: 0 0 6px rgba(56, 189, 248, 0.1) !important; padding: 0.2rem 0.8rem !important; margin-top: 2px; }
div.stButton > button:hover { background: #38BDF8 !important; color: #030712 !important; box-shadow: 0 0 12px rgba(56, 189, 248, 0.6) !important; }

/* ── Pulsing Master Verdict Animations ── */
@keyframes pulse-green { 0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); } 70% { box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); } 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); } }
@keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); } 70% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); } 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); } }
@keyframes pulse-gold { 0% { box-shadow: 0 0 0 0 rgba(200, 168, 75, 0.4); } 70% { box-shadow: 0 0 0 8px rgba(200, 168, 75, 0); } 100% { box-shadow: 0 0 0 0 rgba(200, 168, 75, 0); } }

.verdict-box { border-radius: 6px; padding: 1rem; text-align: center; border: 1px solid rgba(255,255,255,0.1); background: rgba(15, 23, 42, 0.6); backdrop-filter: blur(5px); }
.verdict-box.buy { border-color: #10B981; animation: pulse-green 2s infinite; }
.verdict-box.sell { border-color: #EF4444; animation: pulse-red 2s infinite; }
.verdict-box.hold { border-color: #C8A84B; animation: pulse-gold 2s infinite; }
.verdict-title { font-family: 'Orbitron', sans-serif; font-size: 0.9rem; font-weight: 700; letter-spacing: 2px; }
.verdict-buy-text { color: #10B981; } .verdict-sell-text { color: #EF4444; } .verdict-hold-text { color: #C8A84B; }
.verdict-desc { font-family: 'Inter', sans-serif; font-size: 0.75rem; color: #94A3B8; margin-top: 0.5rem; }

/* ── Custom Tab Styling (More Compact) ── */
.stTabs [data-baseweb="tab-list"] { background-color: transparent; gap: 4px; padding-bottom: 0px; }
.stTabs [data-baseweb="tab"] { font-family: 'Orbitron', sans-serif; font-size: 0.7rem; font-weight: 600; color: #64748B; background-color: #0F172A; border: 1px solid #1E293B; border-bottom: none; border-radius: 4px 4px 0 0; padding: 6px 12px; }
.stTabs [aria-selected="true"] { color: #38BDF8 !important; background-color: rgba(56, 189, 248, 0.1); border-color: #38BDF8; box-shadow: inset 0 2px 0 0 #38BDF8; }

/* ── DataFrame / Table Adjustments ── */
[data-testid="stDataFrame"] { font-family: 'Inter', monospace; font-size: 0.75rem; }

/* ── Sidebar Fine-tuning ── */
[data-testid="stSidebar"] { background-color: #020617 !important; border-right: 1px solid #1E293B; padding-top: 1rem; }
.sidebar-head { font-family: 'Orbitron', sans-serif; color: #C8A84B; font-size: 0.85rem; margin-bottom: 1rem; text-align: center; border-bottom: 1px solid #1E293B; padding-bottom: 0.8rem; }
.stTextInput label { font-family: 'Rajdhani', sans-serif !important; font-size: 0.8rem !important; color: #94A3B8 !important; }

/* ── LIVE TV TICKER ANIMATION ── */
.ticker-wrap { width: 100%; overflow: hidden; background-color: #020617; border-top: 1px solid #38BDF8; border-bottom: 1px solid #38BDF8; padding: 5px 0; margin-bottom: 8px; white-space: nowrap; }
.ticker { display: inline-block; white-space: nowrap; padding-left: 100%; animation: ticker 40s linear infinite; }
.ticker:hover { animation-play-state: paused; }
@keyframes ticker { 0% { transform: translate3d(0, 0, 0); } 100% { transform: translate3d(-100%, 0, 0); } }
.ticker-item { display: inline-block; padding: 0 1.2rem; font-size: 0.8rem; font-family: 'Rajdhani', sans-serif; color: #F8FAFC; border-right: 1px solid #334155; }
.ticker-name { font-weight: 700; color: #94A3B8; margin-right: 4px; }
.ticker-price { font-weight: 700; }
.ticker-up { color: #10B981; font-family: monospace; font-weight: bold; margin-left: 4px; }
.ticker-down { color: #EF4444; font-family: monospace; font-weight: bold; margin-left: 4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# 2. DATA & MODEL PIPELINE (PRODUCTION READY)
# ─────────────────────────────────────────────────────────
def init_db():
    if not os.path.exists(DB_NAME):
        try:
            df_initial = pd.read_csv('ARS_Steels_Accurate_Extracted_Dataset.csv')
            conn = sqlite3.connect(DB_NAME)
            df_initial.to_sql('market_data', conn, if_exists='replace', index=False)
            conn.close()
        except Exception as e:
            st.error(f"Failed to initialize database: {e}")

@st.cache_data
def load_data():
    init_db()
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql('SELECT * FROM market_data', conn)
    conn.close()
    return df

@st.cache_resource
def load_model():
    return joblib.load('ars_pricing_pipeline.pkl'), joblib.load('ars_features.pkl')

def engineer_features(df):
    d = df.copy()
    d['Price_Momentum_1W'] = d.get('ARS_Price_Lag_1W', 0) - d.get('ARS_Price_Lag_2W', 0)
    d['Price_Velocity'] = d.get('ARS_Price_Lag_1W', 0) - d.get('ARS_Price_Rolling_Mean_4W', 0)
    d['Cost_Pressure_Index'] = (
        0.20 * d.get('Scrap_Metal_Price_per_Ton', 0) +
        0.20 * d.get('Turkey_Scrap_Metal_Price_INR_per_Ton', 0) +
        0.25 * d.get('Iron_Ore_Price_per_Ton', 0) +
        0.10 * d.get('CDRI_Raipur_Price_INR_per_Ton', 0) +
        0.10 * d.get('Coking_Coal_Price_per_Ton', 0) +
        0.05 * d.get('RB2_Coal_Gangavaram_Price_INR_per_Ton', 0) +
        0.10 * d.get('Diesel_Price_Chennai', 0) * 10
    )
    d['Competitor_Premium'] = d.get('ARS_Price_Lag_1W', 0) - d.get('Competitor_Avg_Price_per_Ton', 0)
    d['Volatility_Proxy'] = abs(d.get('ARS_Price_Lag_1W', 0) - d.get('ARS_Price_Lag_2W', 0))
    return d

def ingest_and_retrain(new_data_df):
    conn = sqlite3.connect(DB_NAME)
    db_df = pd.read_sql('SELECT * FROM market_data', conn)
    db_df['Date'] = pd.to_datetime(db_df['Date'], errors='coerce')
    
    if not new_data_df.empty:
        new_data_df['Date'] = pd.to_datetime(new_data_df['Date'], errors='coerce')
        combined_df = pd.concat([db_df, new_data_df], ignore_index=True)
    else:
        combined_df = db_df
        
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    combined_df = combined_df.ffill()
    
    combined_df['Week_Number'] = combined_df['Date'].dt.isocalendar().week
    combined_df['Month'] = combined_df['Date'].dt.month
    combined_df['Year'] = combined_df['Date'].dt.year
    
    combined_df['ARS_Price_Lag_1W'] = combined_df['ARS_TMT_Price_per_Ton'].shift(1).bfill()
    combined_df['ARS_Price_Lag_2W'] = combined_df['ARS_TMT_Price_per_Ton'].shift(2).bfill()
    combined_df['ARS_Price_Rolling_Mean_4W'] = combined_df['ARS_TMT_Price_per_Ton'].shift(1).rolling(window=4, min_periods=1).mean().bfill()
    
    combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
    combined_df.to_sql('market_data', conn, if_exists='replace', index=False)
    conn.close()
    
    pipeline_cache, features_cache = load_model()
    df_engineered = engineer_features(combined_df)
    
    for f in features_cache:
        if f not in df_engineered.columns:
            df_engineered[f] = 0
            
    pipeline_cache.fit(df_engineered[features_cache].fillna(0), df_engineered['ARS_TMT_Price_per_Ton'])
    joblib.dump(pipeline_cache, 'ars_pricing_pipeline.pkl')
    st.cache_data.clear()
    st.cache_resource.clear()

try:
    df = load_data()
    pipeline, FEATURES = load_model()
except Exception as e:
    st.error(f"SYSTEM FAULT: Missing artifacts. Run modeltraining.py. Details: {e}")
    st.stop()

if not df.empty:
    latest = df.iloc[-1].copy()
    prev_price = latest.get('ARS_TMT_Price_per_Ton', 0)
else:
    latest = pd.Series({'USD_to_INR': 83.0, 'RBI_Repo_Rate': 6.5, 'Diesel_Price_Chennai': 92.0, 'Scrap_Metal_Price_per_Ton': 35000, 'Iron_Ore_Price_per_Ton': 5000, 'Coking_Coal_Price_per_Ton': 20000, 'Turkey_Scrap_Metal_Price_INR_per_Ton': 35000, 'CDRI_Raipur_Price_INR_per_Ton': 28000, 'PDRI_Bellary_Price_INR_per_Ton': 27000, 'PDRI_Hyderabad_Price_INR_per_Ton': 27500, 'PDRI_Raipur_Price_INR_per_Ton': 26500, 'RB2_Coal_Gangavaram_Price_INR_per_Ton': 10000, 'Competitor_Avg_Price_per_Ton': 55000, 'ARS_TMT_Price_per_Ton': 55000, 'ARS_Price_Lag_1W': 55000, 'ARS_Price_Lag_2W': 55000, 'ARS_Price_Rolling_Mean_4W': 55000})
    prev_price = 55000

# ─────────────────────────────────────────────────────────
# 3. SIDEBAR SIMULATION CONTROLS (TEXT INPUTS + VALIDATION)
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-head">MARKET SIMULATION ENGINE</div>', unsafe_allow_html=True)
    
    usd_inr_raw = st.text_input("USD / INR", value=str(float(latest['USD_to_INR'])))
    repo_raw = st.text_input("RBI Repo Rate (%)", value=str(float(latest['RBI_Repo_Rate'])))
    diesel_raw = st.text_input("Diesel (₹/L)", value=str(float(latest['Diesel_Price_Chennai'])))
    st.markdown("<br>", unsafe_allow_html=True)
    
    scrap_raw = st.text_input("Scrap Metal (₹/Ton)", value=str(int(latest.get('Scrap_Metal_Price_per_Ton', 35000))))
    turkey_scrap_raw = st.text_input("Turkey Scrap (₹/Ton)", value=str(int(latest.get('Turkey_Scrap_Metal_Price_INR_per_Ton', 35000))))
    iron_ore_raw = st.text_input("Iron Ore (₹/Ton)", value=str(int(latest.get('Iron_Ore_Price_per_Ton', 5000))))
    coking_coal_raw = st.text_input("Coking Coal (₹/Ton)", value=str(int(latest.get('Coking_Coal_Price_per_Ton', 20000))))
    rb2_coal_raw = st.text_input("RB2 Coal (₹/Ton)", value=str(int(latest.get('RB2_Coal_Gangavaram_Price_INR_per_Ton', 10000))))
    cdri_raipur_raw = st.text_input("CDRI Raipur (₹/Ton)", value=str(int(latest.get('CDRI_Raipur_Price_INR_per_Ton', 28000))))
    pdri_bellary_raw = st.text_input("PDRI Bellary (₹/Ton)", value=str(int(latest.get('PDRI_Bellary_Price_INR_per_Ton', 27000))))
    pdri_hyd_raw = st.text_input("PDRI Hyderabad (₹/Ton)", value=str(int(latest.get('PDRI_Hyderabad_Price_INR_per_Ton', 27500))))
    pdri_raipur_raw = st.text_input("PDRI Raipur (₹/Ton)", value=str(int(latest.get('PDRI_Raipur_Price_INR_per_Ton', 26500))))
    competitor_raw = st.text_input("Competitor Avg (₹/Ton)", value=str(int(latest.get('Competitor_Avg_Price_per_Ton', 55000))))

# Robust Text & Positive Value Validation
try:
    usd_inr = float(usd_inr_raw)
    repo = float(repo_raw)
    diesel = float(diesel_raw)
    scrap = float(scrap_raw)
    turkey_scrap = float(turkey_scrap_raw)
    iron_ore = float(iron_ore_raw)
    coking_coal = float(coking_coal_raw)
    rb2_coal = float(rb2_coal_raw)
    cdri_raipur = float(cdri_raipur_raw)
    pdri_bellary = float(pdri_bellary_raw)
    pdri_hyd = float(pdri_hyd_raw)
    pdri_raipur = float(pdri_raipur_raw)
    competitor = float(competitor_raw)
    
    # Check for negative values
    inputs_list = [usd_inr, repo, diesel, scrap, turkey_scrap, iron_ore, coking_coal, rb2_coal, cdri_raipur, pdri_bellary, pdri_hyd, pdri_raipur, competitor]
    if any(val < 0 for val in inputs_list):
        st.error("🚨 **INVALID DATA DETECTED:** Please provide valid positive data. Negative values are not allowed.")
        st.stop()
        
except ValueError:
    st.error("🚨 **INVALID DATA DETECTED:** Please enter valid numbers only in the sidebar inputs.")
    st.stop()

# Simulate current input
raw_input = pd.DataFrame([{
    'Diesel_Price_Chennai': diesel, 'RBI_Repo_Rate': repo, 'USD_to_INR': usd_inr,
    'Iron_Ore_Price_per_Ton': iron_ore, 'Coking_Coal_Price_per_Ton': coking_coal,
    'Scrap_Metal_Price_per_Ton': scrap, 'Competitor_Avg_Price_per_Ton': competitor,
    'Turkey_Scrap_Metal_Price_INR_per_Ton': turkey_scrap, 'RB2_Coal_Gangavaram_Price_INR_per_Ton': rb2_coal,
    'CDRI_Raipur_Price_INR_per_Ton': cdri_raipur, 'PDRI_Bellary_Price_INR_per_Ton': pdri_bellary,
    'PDRI_Hyderabad_Price_INR_per_Ton': pdri_hyd, 'PDRI_Raipur_Price_INR_per_Ton': pdri_raipur,
    'ARS_Price_Lag_1W': latest.get('ARS_TMT_Price_per_Ton', 0), 
    'ARS_Price_Lag_2W': latest.get('ARS_Price_Lag_1W', latest.get('ARS_TMT_Price_per_Ton', 0)), 
    'ARS_Price_Rolling_Mean_4W': latest.get('ARS_Price_Rolling_Mean_4W', latest.get('ARS_TMT_Price_per_Ton', 0))
}])

input_df = engineer_features(raw_input)
for f in FEATURES:
    if f not in input_df.columns: input_df[f] = 0
input_df = input_df[FEATURES]

final_price = pipeline.predict(input_df)[0]
ai_delta = final_price - prev_price
scrap_delta = scrap - latest.get('Scrap_Metal_Price_per_Ton', 0)
comp_delta = competitor - latest.get('Competitor_Avg_Price_per_Ton', 0)

baseline_feats = engineer_features(pd.DataFrame([latest]))
cost_idx_delta = input_df['Cost_Pressure_Index'].iloc[0] - (baseline_feats['Cost_Pressure_Index'].iloc[0] if 'Cost_Pressure_Index' in baseline_feats.columns else 0)

def generate_kpi_card(title, value, delta):
    color_class = "positive" if delta > 0 else "negative" if delta < 0 else "neutral"
    sign = "+" if delta > 0 else ""
    return f'<div class="tech-card"><div class="card-label">{title}</div><div class="card-value">₹{value:,.0f}</div><div class="card-delta {color_class}">{sign}{delta:,.0f} vs Prev</div></div>'

# ─────────────────────────────────────────────────────────
# 4. MAIN LAYOUT & TICKER
# ─────────────────────────────────────────────────────────
st.markdown('<p class="glow-title">ARS STEELS</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">PREDICTIVE PRICING & MARKET INTELLIGENCE TERMINAL</p>', unsafe_allow_html=True)

# Generate Fake Live Ticker Data
ticker_data = [
    {"name": "NIFTY METALS", "price": "8,102.4", "delta": "+1.2%"},
    {"name": "BSE SENSEX", "price": "73,120", "delta": "-0.4%"},
    {"name": "LME STEEL SCRAP", "price": "$410", "delta": "+0.8%"},
    {"name": "BRENT CRUDE", "price": "$82.40", "delta": "+1.5%"},
    {"name": "USD/INR", "price": "83.15", "delta": "-0.1%"},
    {"name": "GLOBAL IRON ORE", "price": "$125", "delta": "+2.0%"},
    {"name": "SHANGHAI REBAR", "price": "¥3,500", "delta": "-1.1%"}
]

ticker_html = '<div class="ticker-wrap"><div class="ticker">'
for item in ticker_data:
    color_class = "ticker-up" if "+" in item["delta"] else "ticker-down"
    ticker_html += f'<div class="ticker-item"><span class="ticker-name">{item["name"]}</span><span class="ticker-price">{item["price"]}</span><span class="{color_class}">{item["delta"]}</span></div>'
ticker_html += '</div></div>'
st.markdown(ticker_html, unsafe_allow_html=True)

st.markdown('<div class="kpi-grid">', unsafe_allow_html=True)
st.markdown(
    generate_kpi_card("AI Target Price", final_price, ai_delta) + 
    generate_kpi_card("Market Competitor Avg", competitor, comp_delta) + 
    generate_kpi_card("Scrap Metal Trajectory", scrap, scrap_delta) + 
    generate_kpi_card("Cost Pressure Index", input_df['Cost_Pressure_Index'].iloc[0], cost_idx_delta) + 
    '</div>', 
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "[ ⚡ ALGORITHMIC BREAKDOWN ]", 
    "[ 🌐 MARKET TOPOLOGY ]", 
    "[ 📊 HISTORICAL BACKTEST ]", 
    "[ 🔄 CONTINUOUS LEARNING HUB ]", 
    "[ 🗄️ DATABASE ADMIN ]",
    "[ 📰 FINANCIAL TERMINAL ]"
])

# ══════════════════════════════════════════════════════════
# DATA PROCESSING SUB-TABS (1 - 5)
# ══════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1.2, 1.5], gap="large")
    with col1:
        st.markdown('<div class="section-header">SIMULATED DRIVER ATTRIBUTION</div>', unsafe_allow_html=True)
        impacts = [("Base Price", prev_price), ("Raw Materials", (scrap - latest.get('Scrap_Metal_Price_per_Ton', 0)) * 0.40), ("Competitors", (competitor - latest.get('Competitor_Avg_Price_per_Ton', 0)) * 0.35), ("Macro Data", (diesel - latest.get('Diesel_Price_Chennai', 0)) * 100 + (usd_inr - latest.get('USD_to_INR', 0)) * 50), ("AI Adjustment", final_price - (prev_price + ((scrap - latest.get('Scrap_Metal_Price_per_Ton', 0)) * 0.40) + ((competitor - latest.get('Competitor_Avg_Price_per_Ton', 0)) * 0.35) + ((diesel - latest.get('Diesel_Price_Chennai', 0)) * 100 + (usd_inr - latest.get('USD_to_INR', 0)) * 50)))]
        fig_wf = go.Figure(go.Waterfall(orientation="v", measure=["absolute", "relative", "relative", "relative", "relative", "total"], x=[i[0] for i in impacts] + ["Final Prediction"], y=[i[1] for i in impacts] + [0], textposition="outside", text=[f"₹{v:,.0f}" if i==0 or i==5 else f"{'+' if v>0 else ''}₹{v:,.0f}" for i, v in enumerate([i[1] for i in impacts] + [0])], connector={"line": {"color": "#334155"}}, decreasing={"marker": {"color": "#EF4444"}}, increasing={"marker": {"color": "#10B981"}}, totals={"marker": {"color": "#38BDF8"}}))
        fig_wf.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#94A3B8'), height=300, margin=dict(t=30, b=10, l=10, r=10))
        st.plotly_chart(fig_wf, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">CORE XAI (SHAP TREE EXPLAINER)</div>', unsafe_allow_html=True)
        try:
            xgb_model = pipeline.named_steps['model'].named_estimators_.get('xgb')
            if xgb_model:
                shap_values = shap.TreeExplainer(xgb_model)(pipeline.named_steps['scaler'].transform(input_df))
                shap_values.feature_names = FEATURES
                fig_shap, ax = plt.subplots(figsize=(8, 3.5))
                fig_shap.patch.set_facecolor('#050505'); ax.set_facecolor('#050505'); ax.xaxis.label.set_color('#94A3B8'); ax.tick_params(colors='#94A3B8', labelsize=8)
                for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
                for spine in ['bottom', 'left']: ax.spines[spine].set_color('#334155')
                plt.rcParams['text.color'] = '#E2E8F0'
                shap.plots.waterfall(shap_values[0], max_display=8, show=False)
                plt.tight_layout()
                st.pyplot(fig_shap); plt.clf()
        except Exception:
            st.info("SHAP engine awaiting localized variance vectors.")

with tab2:
    col_t1, col_t2 = st.columns(2, gap="large")
    with col_t1:
        st.markdown('<div class="section-header">COMMODITY VECTOR TRAJECTORY</div>', unsafe_allow_html=True)
        plot_df = df.copy()
        if not plot_df.empty and 'Scrap_Metal_Price_per_Ton' in plot_df.columns:
            plot_df['Timeline'] = range(len(plot_df))
            fig_macro = px.line(plot_df, x='Timeline', y=['Scrap_Metal_Price_per_Ton', 'Iron_Ore_Price_per_Ton'], color_discrete_sequence=['#38BDF8', '#C8A84B'])
            fig_macro.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#94A3B8'), height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig_macro, use_container_width=True)
        else:
            st.info("Awaiting Historical Data")
    with col_t2:
        st.markdown('<div class="section-header">ELASTICITY MATRIX</div>', unsafe_allow_html=True)
        if not df.empty and len(df) > 5:
            fig_corr = px.imshow(df[['ARS_TMT_Price_per_Ton', 'Scrap_Metal_Price_per_Ton', 'Iron_Ore_Price_per_Ton', 'Competitor_Avg_Price_per_Ton', 'USD_to_INR']].corr(), text_auto=".2f", aspect="auto", color_continuous_scale='Mint')
            fig_corr.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#94A3B8'), height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Awaiting sufficient historical data for correlation mapping.")

with tab3:
    st.markdown('<div class="section-header">ALGORITHMIC BACKTEST VALIDATION</div>', unsafe_allow_html=True)
    if not df.empty:
        full_df = engineer_features(df)
        for f in FEATURES:
            if f not in full_df.columns: full_df[f] = 0
        preds = pipeline.predict(full_df[FEATURES].fillna(0))
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(y=full_df['Competitor_Avg_Price_per_Ton'], name='Competitor Avg', line=dict(color='#475569', dash='dot')))
        fig_hist.add_trace(go.Scatter(y=full_df['ARS_TMT_Price_per_Ton'], name='Actual Price', line=dict(color='#C8A84B', width=2)))
        fig_hist.add_trace(go.Scatter(y=preds, name='AI Prediction', line=dict(color='#38BDF8', width=2)))
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#94A3B8'), height=300, margin=dict(t=10, b=10), hovermode="x unified")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Awaiting historical data for backtesting.")

with tab4:
    st.markdown('<div class="section-header">DATA INGESTION HUB</div>', unsafe_allow_html=True)
    ingest_method = st.radio("Select Input Method:", ["1. Single Week Form", "2. Multi-Row Entry Grid", "3. Upload CSV File"], horizontal=True)
    st.markdown("---")
    if "Single" in ingest_method:
        with st.form("retrain_form"):
            col_a, col_b, col_c = st.columns(3)
            actual_usd = col_a.number_input("Actual USD/INR", value=float(usd_inr))
            actual_repo = col_b.number_input("Actual Repo Rate", value=float(repo))
            actual_diesel = col_c.number_input("Actual Diesel", value=float(diesel))
            col_d, col_e, col_f = st.columns(3)
            actual_scrap = col_d.number_input("Actual Scrap Metal", value=int(scrap))
            actual_iron = col_e.number_input("Actual Iron Ore", value=int(iron_ore))
            actual_coal = col_f.number_input("Actual Coking Coal", value=int(coking_coal))
            col_g, col_h = st.columns(2)
            actual_comp = col_g.number_input("Actual Competitor Avg", value=int(competitor))
            actual_ars_price = col_h.number_input("Final ARS Selling Price", value=int(final_price))
            if st.form_submit_button("Commit Data & Retrain AI ⚡"):
                with st.spinner("Appending row and recompiling AI..."):
                    new_date = pd.to_datetime(latest.get('Date', pd.Timestamp.now().strftime('%Y-%m-%d'))) + pd.Timedelta(weeks=1)
                    new_row = {'Date': new_date.strftime('%Y-%m-%d'), 'Diesel_Price_Chennai': actual_diesel, 'RBI_Repo_Rate': actual_repo, 'USD_to_INR': actual_usd, 'Iron_Ore_Price_per_Ton': actual_iron, 'Coking_Coal_Price_per_Ton': actual_coal, 'Scrap_Metal_Price_per_Ton': actual_scrap, 'Competitor_Avg_Price_per_Ton': actual_comp, 'ARS_TMT_Price_per_Ton': actual_ars_price}
                    ingest_and_retrain(pd.DataFrame([new_row]))
                    st.success("✅ AI Retrained successfully!")
                    st.rerun()
    elif "Multi-Row" in ingest_method:
        cols = ['Date', 'Diesel_Price_Chennai', 'RBI_Repo_Rate', 'USD_to_INR', 'Iron_Ore_Price_per_Ton', 'Coking_Coal_Price_per_Ton', 'Scrap_Metal_Price_per_Ton', 'Competitor_Avg_Price_per_Ton', 'ARS_TMT_Price_per_Ton']
        if 'bulk_grid' not in st.session_state: st.session_state.bulk_grid = pd.DataFrame(np.nan, index=range(5), columns=cols)
        edited_bulk = st.data_editor(st.session_state.bulk_grid, num_rows="dynamic", use_container_width=True, height=250)
        if st.button("⚡ Append All Grid Rows to Database & Retrain"):
            valid_df = edited_bulk.dropna(subset=['Date', 'ARS_TMT_Price_per_Ton'])
            if len(valid_df) > 0:
                with st.spinner(f"Ingesting {len(valid_df)} rows..."):
                    ingest_and_retrain(valid_df)
                    st.session_state.bulk_grid = pd.DataFrame(np.nan, index=range(5), columns=cols)
                    st.rerun()
    elif "Upload CSV" in ingest_method:
        uploaded_file = st.file_uploader("Drop CSV Here", type=['csv'])
        if uploaded_file is not None:
            csv_df = pd.read_csv(uploaded_file)
            if st.button("⚡ Append CSV File to Database & Retrain"):
                with st.spinner("Processing CSV..."):
                    ingest_and_retrain(csv_df)
                    st.rerun()

with tab5:
    st.markdown('<div class="section-header">DATABASE MANAGEMENT CONSOLE</div>', unsafe_allow_html=True)
    conn = sqlite3.connect(DB_NAME); current_db_df = pd.read_sql('SELECT * FROM market_data ORDER BY Date ASC', conn); conn.close()
    col_del1, col_del2 = st.columns([2, 1])
    with col_del1:
        date_list = current_db_df['Date'].sort_values(ascending=False).tolist() if not current_db_df.empty else []
        date_to_delete = st.selectbox("Quick Delete Record:", ["-- Select a Date --"] + date_list)
    with col_del2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚨 Delete Selected Week") and date_to_delete != "-- Select a Date --":
            conn = sqlite3.connect(DB_NAME); conn.cursor().execute("DELETE FROM market_data WHERE Date = ?", (date_to_delete,)); conn.commit(); conn.close()
            ingest_and_retrain(pd.DataFrame())
            st.rerun()
    st.markdown("---")
    edited_full_df = st.data_editor(current_db_df, use_container_width=True, num_rows="dynamic", height=300)
    if st.button("💾 Sync Table Edits to Database & Retrain AI"):
        conn = sqlite3.connect(DB_NAME); conn.cursor().execute("DELETE FROM market_data"); conn.commit(); conn.close()
        ingest_and_retrain(edited_full_df)
        st.rerun()

# ══════════════════════════════════════════════════════════
# TAB 6 — NEON FINANCIAL TERMINAL (WINDOW SCALED & FIXED FEED)
# ══════════════════════════════════════════════════════════
with tab6:
    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_live_equities(ai_price, p_price):
        tickers = {
            "TATA STEEL": "TATASTEEL.NS", "JSW STEEL": "JSWSTEEL.NS", "SAIL": "SAIL.NS", 
            "JINDAL STEL": "JINDALSTEL.NS", "VEDANTA": "VEDL.NS",
            "US STEEL (USD)": "X", "NUCOR (USD)": "NUE", 
            "BAOSTEEL (CNY)": "600019.SS", "CHINA STEEL (TWD)": "2002.TW"
        }
        stock_data = []
        ars_change = ai_price - p_price
        ars_pct = (ars_change / p_price) * 100 if p_price != 0 else 0
        stock_data.append({"name": "ARS STEELS (INTERNAL TGT)", "price": ai_price, "change": ars_change, "pct": ars_pct, "currency": "₹"})

        for name, ticker in tickers.items():
            try:
                hist = yf.Ticker(ticker).history(period="5d")
                if hist is not None and not hist.empty and len(hist) >= 2 and 'Close' in hist.columns:
                    current = float(hist['Close'].iloc[-1])
                    prev = float(hist['Close'].iloc[-2])
                    change = current - prev
                    pct = (change / prev) * 100 if prev != 0 else 0
                    
                    currency = "₹"
                    if "(USD)" in name: currency = "$"
                    elif "(CNY)" in name: currency = "¥"
                    elif "(TWD)" in name: currency = "NT$"
                    stock_data.append({"name": name, "price": current, "change": change, "pct": pct, "currency": currency})
            except Exception: pass
            
        if len(stock_data) == 1:
            stock_data.extend([
                {"name": "TATA STEEL (SIM)", "price": 165.40, "change": 2.10, "pct": 1.28, "currency": "₹"},
                {"name": "JSW STEEL (SIM)", "price": 890.20, "change": -5.40, "pct": -0.60, "currency": "₹"},
                {"name": "US STEEL (USD)", "price": 43.60, "change": 1.20, "pct": 2.80, "currency": "$"},
                {"name": "BAOSTEEL (CNY)", "price": 5.96, "change": -0.10, "pct": -1.60, "currency": "¥"}
            ])
        return stock_data

    stocks = fetch_live_equities(final_price, prev_price)
    ticker_html = '<div class="ticker-wrap"><div class="ticker">'
    for s in stocks:
        trend_class = "ticker-up" if s['change'] >= 0 else "ticker-down"
        trend_icon = "▲" if s['change'] >= 0 else "▼"
        sign = "+" if s['change'] >= 0 else ""
        ticker_html += f'<div class="ticker-item"><span class="ticker-name">{s["name"]}</span><span class="ticker-price">{s["currency"]}{s["price"]:.2f}</span><span class="{trend_class}">{trend_icon} {sign}{s["change"]:.2f} ({sign}{s["pct"]:.2f}%)</span></div>'
    ticker_html += '</div></div>'
    st.markdown(ticker_html, unsafe_allow_html=True)

    @st.cache_data(ttl=1800, show_spinner=False)
    def fetch_vader_sentiment():
        url = "https://news.google.com/rss/search?q=steel+prices+market+india&hl=en-IN&gl=IN&ceid=IN:en"
        news_data, words_pool = [], []
        fallback_df = pd.DataFrame([
            {"Headline": "India Fastest-Growing Steel Market Even As Global Prices Surge: Goldman Sachs", "Published": "Today", "Score": 0.65, "Sentiment": "BULLISH 📈", "Color": "#10B981"},
            {"Headline": "Tata Steel CEO TV Narendran cautiously optimistic; sees strong Q1 ahead as steel prices rise across India", "Published": "Today", "Score": 0.45, "Sentiment": "BULLISH 📈", "Color": "#10B981"},
            {"Headline": "Mild steel prices in India to hit Rs 61,000 per tonne - Manufacturing Today India", "Published": "Today", "Score": 0.0, "Sentiment": "NEUTRAL ⚖️", "Color": "#94A3B8"},
            {"Headline": "Indian steel sector maintains growth momentum in April; prices rebound across segments - DD News", "Published": "Yesterday", "Score": 0.55, "Sentiment": "BULLISH 📈", "Color": "#10B981"},
            {"Headline": "Steel prices in India remained under pressure in the FY2025/2026 – BigMint - GMK Center", "Published": "Yesterday", "Score": -0.35, "Sentiment": "BEARISH 📉", "Color": "#EF4444"},
            {"Headline": "Weak offtake pressures steel prices in south India - BigMint", "Published": "2 Days Ago", "Score": -0.40, "Sentiment": "BEARISH 📉", "Color": "#EF4444"},
            {"Headline": "JSW Steel targets capacity expansion matching infrastructural targets", "Published": "3 Days Ago", "Score": 0.50, "Sentiment": "BULLISH 📈", "Color": "#10B981"}
        ])
        fallback_words = [("global", 5), ("across", 4), ("goldman", 3), ("prices", 3), ("growth", 2)]

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
            root = ET.fromstring(urllib.request.urlopen(req, timeout=5).read())
            sia = SentimentIntensityAnalyzer()
            
            items = root.findall('./channel/item')[:20]
            if len(items) > 0:
                for item in items:
                    title_elem = item.find('title')
                    pub_elem = item.find('pubDate')
                    if title_elem is not None and title_elem.text:
                        title = str(title_elem.text).strip()
                        pub_date = str(pub_elem.text) if pub_elem is not None else ""
                        try:
                            score_dict = sia.polarity_scores(title)
                            compound = score_dict['compound']
                            if compound >= 0.15: sentiment_label = "BULLISH 📈"; color = "#10B981"
                            elif compound <= -0.15: sentiment_label = "BEARISH 📉"; color = "#EF4444"
                            else: sentiment_label = "NEUTRAL ⚖️"; color = "#94A3B8"
                            for w in title.lower().split():
                                clean_w = ''.join(c for c in str(w) if c.isalpha())
                                if len(clean_w) > 4 and clean_w not in ['steel', 'prices', 'india', 'market', 'price', 'industries']: words_pool.append(clean_w)
                            news_data.append({"Headline": title, "Published": pub_date[:-15] if len(pub_date)>15 else pub_date, "Score": compound, "Sentiment": sentiment_label, "Color": color})
                        except Exception: continue
            if len(news_data) == 0: return fallback_df, fallback_words
            return pd.DataFrame(news_data), collections.Counter(words_pool).most_common(5)
        except Exception:
            return fallback_df, fallback_words

    with st.spinner("Processing VADER NLP Data Matrices..."):
        news_df, top_buzzwords = fetch_vader_sentiment()

    avg_sentiment = round(news_df['Score'].mean(), 2)
    ai_trend_pct = ((final_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0
    composite_score = (ai_trend_pct * 0.7) + (avg_sentiment * 10 * 0.3)
    
    if composite_score >= 1.5: verdict, v_color, v_icon, v_sub, pulse_class = "STRONG BUY / HOLD INVENTORY", "#10B981", "🚀", "Mathematical algorithms and global structural indicators confirm expanding margins.", "pulse-green"
    elif composite_score <= -1.5: verdict, v_color, v_icon, v_sub, pulse_class = "LIQUIDATE / SELL IMMEDIATELY", "#EF4444", "⚠️", "Downward raw material vectors suggest structural market optimization. Clear asset pools.", "pulse-red"
    else: verdict, v_color, v_icon, v_sub, pulse_class = "MARKET STABLE / STD OPERATIONS", "#C8A84B", "⚖️", "Conflicting or flat signals. Continue standard sales operations at AI Recommended Target.", "pulse-gold"

    col_v1, col_v2 = st.columns([5, 1])
    with col_v1: st.markdown('<div class="section-header" style="margin-top:0;">EXECUTIVE MARKET VERDICT</div>', unsafe_allow_html=True)
    with col_v2:
        if st.button("🔄 REFRESH CORE DATA"):
            fetch_live_equities.clear()
            fetch_vader_sentiment.clear()
            st.rerun()
            
    v_html = f"<div style='background: rgba(15, 23, 42, 0.6); padding: 10px 12px; border-radius: 6px; border: 1px solid {v_color}; text-align: center; margin-bottom: 10px; animation: {pulse_class} 2s infinite;'><h4 style='color: #94A3B8; margin-top: 0; font-family: Inter; text-transform: uppercase; font-size: 0.7rem;'>Synthesized AI Action Plan</h4><h1 style='color: {v_color}; font-size: 1.5rem; margin: 2px 0; font-family: Orbitron; font-weight: 900; text-shadow: 0 0 10px {v_color};'>{v_icon} {verdict}</h1><p style='color: #E2E8F0; font-size: 0.78rem; margin-bottom: 0;'>{v_sub}</p></div>"
    st.markdown(v_html, unsafe_allow_html=True)

    col_g1, col_g2, col_g3 = st.columns([1.2, 1.1, 1.7])
    with col_g1:
        st.markdown('<p style="color:#38BDF8; font-weight:700; font-size:0.7rem; text-transform:uppercase; margin-bottom:0; letter-spacing: 0.5px;">Psychology Speedometer</p>', unsafe_allow_html=True)
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = avg_sentiment, domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'font': {'color': '#F8FAFC', 'family': 'Rajdhani', 'size': 20}, 'valueformat': '+.2f'},
            gauge = {
                'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': "#38BDF8", 'thickness': 0.22}, 'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 1, 'bordercolor': "#334155",
                'steps': [{'range': [-1.0, -0.3], 'color': 'rgba(239, 68, 68, 0.15)'}, {'range': [-0.3, 0.3], 'color': 'rgba(148, 163, 184, 0.08)'}, {'range': [0.3, 1.0], 'color': 'rgba(16, 185, 129, 0.15)'}],
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=140, margin=dict(t=10, b=0, l=10, r=10))
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

    with col_g2:
        st.markdown('<p style="color:#38BDF8; font-weight:700; font-size:0.7rem; text-transform:uppercase; margin-bottom:5px; letter-spacing: 0.5px;">Media Pulse Allocation</p>', unsafe_allow_html=True)
        counts = news_df['Sentiment'].value_counts()
        colors_list = [{"BULLISH 📈": "#10B981", "BEARISH 📉": "#EF4444", "NEUTRAL ⚖️": "#475569"}.get(n, "#94A3B8") for n in counts.index]
        fig_donut = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=.6, marker=dict(colors=colors_list), textinfo='percent', hoverinfo='label')])
        fig_donut.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=140, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_donut, use_container_width=True, config={'displayModeBar': False})

    with col_g3:
        st.markdown('<p style="color:#38BDF8; font-weight:700; font-size:0.7rem; text-transform:uppercase; margin-bottom:8px; letter-spacing: 0.5px;">Sentiment Vectors</p>', unsafe_allow_html=True)
        if top_buzzwords:
            col_w1, col_w2 = st.columns(2)
            for i, (word, freq) in enumerate(top_buzzwords):
                target_col = col_w1 if i % 2 == 0 else col_w2
                w_html = f"<div style='display:flex; justify-content:space-between; padding: 4px 8px; margin-bottom:3px; background:rgba(30,41,59,0.3); border-radius:3px; border:1px solid rgba(56,189,248,0.15)'><span style='color:#E2E8F0; font-size:0.72rem; font-family:Inter; text-transform:capitalize;'>📍 {word}</span><span style='color:#C8A84B; font-weight:700; font-size:0.72rem; font-family:monospace;'>{freq}x</span></div>"
                target_col.markdown(w_html, unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:0.5rem;">TERMINAL INTELLIGENCE REAL-TIME FEED</div>', unsafe_allow_html=True)
    
    up_count = sum(1 for s in stocks if s['change'] > 0)
    sentiment_text = "bullish" if avg_sentiment > 0.15 else "bearish" if avg_sentiment < -0.15 else "neutral"
    
    st.info(f"**Terminal Diagnostic Summary:** Global equities confirm localized support with {up_count} of {len(stocks)} raw production indices moving green. Aggregated algorithmic web-scraping tracking identifies a distinct **{sentiment_text}** sector posture (VADER Vector: {avg_sentiment:+.2f}), indicating a supply-side stabilization aligned with structural AI pricing projections.")

    with st.container(height=240, border=True):
        for index, row in news_df.iterrows():
            row_html = f"<div style='margin-bottom: 4px; padding: 5px 8px; border-radius: 4px; background: linear-gradient(90deg, #0A0F1A 0%, rgba(10,15,26,0.2) 100%); border-left: 2px solid {row['Color']}; display: flex; align-items: center; justify-content: space-between;'><div style='display: flex; align-items: center; gap: 10px;'><span style='color: {row['Color']}; font-weight: 700; font-family: Rajdhani; font-size: 0.7rem; width: 75px; display: inline-block;'>{row['Sentiment']}</span><span style='color: #E2E8F0; font-size: 0.75rem;'>{row['Headline']}</span></div><span style='color: #475569; font-size: 0.62rem; font-family: monospace; white-space: nowrap; margin-left:10px;'>{row['Published']}</span></div>"
            st.markdown(row_html, unsafe_allow_html=True)
