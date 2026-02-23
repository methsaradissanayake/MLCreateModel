import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
import base64
import plotly.graph_objects as go
import plotly.express as px

# --- Helper to load local image as base64 for CSS background ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Use the background image from the project assets
bg_img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets', 'background.png')
BACKGROUND_IMAGE_URL = ""
if os.path.exists(bg_img_path):
    bg_encoded = get_base64_of_bin_file(bg_img_path)
    BACKGROUND_IMAGE_URL = f"url('data:image/png;base64,{bg_encoded}')"
else:
    # Fallback dark gradient
    BACKGROUND_IMAGE_URL = "linear-gradient(to bottom right, #1a1a2e, #16213e, #0f3460)"

# --- 1. Configuration and Loading ---
st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed" # Hide sidebar
)

# Custom CSS for Dark Minimalist style (Using regular string to avoid f-string brace issues)
css_style = """
<style>
    /* Final CSS Fix: Using regular string. Braces are now single. */
    header[data-testid="stHeader"] {
        display: none;
    }
    .block-container {
        padding-top: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }
    
    /* Global Background and Text Color */
    .stApp {
        background-image: __BG_IMAGE_URL__;
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #ffffff;
    }
    
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #f0f0f0 !important;
    }
    
    /* Custom Header */
    .custom-header {
        background-color: rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(10px);
        color: #FFFFFF;
        padding: 1.5rem 0rem;
        display: flex;
        justify-content: center;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 999;
        font-family: 'Inter', sans-serif;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    .header-logo {
        color: #d4af37 !important; /* Gold accent */
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(212, 175, 55, 0.3);
    }
    
    /* Hero Section */
    .hero-section {
        background-color: transparent;
        padding: 4rem 2rem;
        text-align: center;
        margin-bottom: 3rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #cccccc !important;
        font-weight: 400;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Product Card Style for Prediction */
    .product-card {
        background: rgba(20, 20, 20, 0.85);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 15px 35px rgba(0,0,0,0.4);
        margin: 2rem auto;
        max-width: 800px;
    }
    .product-brand {
        text-transform: uppercase;
        font-size: 0.8rem;
        font-weight: 600;
        color: #aaa !important;
        letter-spacing: 2px;
    }
    .product-name {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff !important;
        margin: 0.5rem 0;
    }
    .product-price {
        font-size: 2rem;
        font-weight: 600;
        color: #d4af37 !important; /* Gold price */
        margin-top: 1rem;
        text-shadow: 0 0 15px rgba(212, 175, 55, 0.4);
    }
    
    /* Section Titles */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid rgba(255,255,255,0.3);
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    /* Inputs Override for Dark Mode - FORCING BLACK TEXT */
    [data-baseweb="popover"], [data-baseweb="select"] > div, [data-baseweb="select"] span {
        color: #000000 !important;
    }
    div[data-testid="stSelectbox"] div, div[data-testid="stSelectbox"] span, div[data-testid="stSelectbox"] p {
        color: #000000 !important;
    }
    [data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border-radius: 8px;
    }
    [data-baseweb="input"] > div {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    input[type="number"], div[data-testid="stNumberInput"] input {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    /* Listbox items force black - targeting all children */
    div[role="listbox"] *, [data-baseweb="popover"] * {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: #000000 !important;
    }
    
    /* Button Style */
    button[kind="primary"] {
        background-color: #d4af37 !important;
        color: #000000 !important;
        font-weight: bold !important;
        border: none !important;
    }
    button[kind="primary"]:hover {
        background-color: #c4a132 !important;
        box-shadow: 0 0 15px rgba(212, 175, 55, 0.6) !important;
    }
    
</style>
""".replace("__BG_IMAGE_URL__", BACKGROUND_IMAGE_URL)

st.markdown(css_style, unsafe_allow_html=True)

# Custom Header HTML
st.markdown("""
<div class="custom-header">
    <div class="header-logo">MOBILE PREDICTOR</div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_data():
    try:
        model = joblib.load('models/xgb_model.joblib')
        encoders = joblib.load('models/encoders.joblib')
        with open('outputs/metrics.json', 'r') as f:
            metrics = json.load(f)
        ref_df = pd.read_csv('data/processed/reference.csv')
        return model, encoders, metrics, ref_df
    except Exception as e:
        st.error(f"Error loading required files: {e}")
        return None, None, None, None

model, encoders, metrics, ref_df = load_models_and_data()

if model is None:
    st.stop()

# Build lists
brands = sorted([b for b in encoders['Brand'].classes_ if b != 'Unknown'])
conditions = sorted([c for c in encoders['Condition'].classes_ if c != 'Unknown'])
locations = sorted([l for l in encoders['location'].classes_ if l != 'Unknown'])

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">Find the Fair Market Value</div>
    <div class="hero-subtitle">Enter device specifications below to get an instant, AI-driven price estimate in Sri Lanka.</div>
</div>
""", unsafe_allow_html=True)

# --- Centered Main Container Wrapper ---
pad_l, center_col, pad_r = st.columns([1, 6, 1])

with center_col:
    # --- Input Grid ---
    st.markdown('<div class="section-title">Device Specifications</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_brand = st.selectbox("Brand", brands)
    with col2:
        available_models = ref_df[ref_df['Brand'] == selected_brand]['Model'].unique()
        available_models = [m for m in available_models if m != 'Unknown']
        if len(available_models) == 0:
            available_models = [m for m in encoders['Model'].classes_ if m != 'Unknown']
        selected_model = st.selectbox("Model", sorted(available_models))
    with col3:
        selected_condition = st.selectbox("Condition", conditions)
    with col4:
        selected_storage = st.number_input("Storage (GB)", min_value=16, max_value=2048, value=128, step=16)
    
    col_loc, col_btn, _, _ = st.columns(4)
    with col_loc:
        selected_location = st.selectbox("Location", locations)
    with col_btn:
        st.write("") # spacing
        st.write("")
        predict_clicked = st.button("Calculate Price Estimation", use_container_width=True, type="primary")
    
    # --- Prediction & Results ---
    if predict_clicked:
        # Prepare data
        input_data = {
            'Brand': [selected_brand],
            'Model': [selected_model],
            'Condition': [selected_condition],
            'location': [selected_location],
            'membershipLevel': ['free'],
            'Storage_GB': [selected_storage]
        }
        input_df = pd.DataFrame(input_data)
        
        encoded_input = pd.DataFrame()
        for col in input_df.columns:
            if col in encoders:
                try:
                    encoded_input[f"{col}_Encoded"] = encoders[col].transform(input_df[col].astype(str))
                except ValueError:
                    encoded_input[f"{col}_Encoded"] = 0
            else:
                encoded_input[col] = input_df[col]
                
        encoded_input = encoded_input.astype(float)
        predicted_price = model.predict(encoded_input)[0]
        
        # Store in session state
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(encoded_input)
        st.session_state['prediction'] = predicted_price
        st.session_state['local_shap'] = (shap_values, encoded_input.iloc[0])
    
    # Render Output if available
    if 'prediction' in st.session_state:
        st.markdown('<div class="section-title" style="margin-top: 3rem;">Estimated Value</div>', unsafe_allow_html=True)
        
        pred = st.session_state['prediction']
        
        # Product Card
        st.markdown(f"""
        <div class="product-card">
            <div class="product-brand">{selected_brand}</div>
            <div class="product-name">{selected_model} {selected_storage}GB ({selected_condition})</div>
            <div style="margin: 1rem 0; color: #aaa !important;">Location: {selected_location}</div>
            <div class="product-price">Rs {pred:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1);'><br>", unsafe_allow_html=True)
        
        # Explainable AI Section
        st.markdown('<div class="section-title">Explainable AI & Model Insights</div>', unsafe_allow_html=True)
        st.markdown("""
        Our predictor doesn't just give you a number; it explains *why*. Using **SHAP (SHapley Additive exPlanations)**, 
        we break down the internal logic of the XGBoost model to show you exactly how each feature influenced your specific price.
        """)
        
        # Performance Metrics
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1:
            st.metric("Accuracy (R²)", f"{metrics['R2_Score']:.1%}")
        with perf_col2:
            st.metric("Avg. Error (MAE)", f"Rs {metrics['MAE']:,.0f}")
        with perf_col3:
            st.metric("Model RMSE", f"Rs {metrics['RMSE']:,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Explanations Grid
        exp_col1, exp_col2 = st.columns(2, gap="large")
        
        with exp_col1:
            st.markdown('### How did we calculate this?')
            
            shap_vals = st.session_state['local_shap'][0]
            base_value = shap_vals.base_values[0]
            contributions = shap_vals.values[0]
            feature_names = st.session_state['local_shap'][1].index.tolist()
            
            sorted_indices = np.argsort(np.abs(contributions))
            sorted_contributions = contributions[sorted_indices]
            sorted_features = np.array(feature_names)[sorted_indices].tolist()
            
            fig_waterfall = go.Figure(go.Waterfall(
                orientation="h",
                measure=["absolute"] + ["relative"] * len(sorted_features) + ["total"],
                y=["Average Price"] + [f.split("_")[0] for f in sorted_features] + ["Estimated Price"],
                x=[base_value] + list(sorted_contributions) + [0],
                connector={"line": {"color": "rgb(120, 120, 120)"}},
                decreasing={"marker": {"color": "#4da6ff"}},
                increasing={"marker": {"color": "#ff4d4d"}},
                totals={"marker": {"color": "#d4af37"}}
            ))
            fig_waterfall.update_layout(
                showlegend=False, 
                margin=dict(l=0, r=0, t=20, b=0), 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff')
            )
            fig_waterfall.update_yaxes(showgrid=False)
            fig_waterfall.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
        with exp_col2:
            st.markdown('### Global Market Variables')
            
            importance_df = pd.DataFrame({
                'Feature': [f.split("_")[0] for f in model.feature_names_in_], # Clean names
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=True)
    
            fig_global = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                color='Importance', color_continuous_scale='Mint') # Lighter mint scale for dark bg
            fig_global.update_layout(
                margin=dict(l=0, r=0, t=20, b=0), 
                coloraxis_showscale=False, 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#ffffff')
            )
            fig_global.update_yaxes(showgrid=False)
            fig_global.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            st.plotly_chart(fig_global, use_container_width=True)
