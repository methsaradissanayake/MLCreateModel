import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
import plotly.graph_objects as go
import plotly.express as px

# --- 1. Configuration and Loading ---
st.set_page_config(
    page_title="Mobile Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed" # Hide sidebar to match francium style
)

# Custom CSS for Francium.lk style (Black/White/Gold minimalist)
st.markdown("""
<style>
    /* Hide the default Streamlit header and main padding */
    header[data-testid="stHeader"] {
        display: none;
    }
    .block-container {
        padding-top: 0rem !important;
        padding-left: 0rem !important;
        padding-right: 0rem !important;
        max-width: 100% !important;
    }
    
    /* Custom Header */
    .custom-header {
        background-color: #000000;
        color: #FFFFFF;
        padding: 1.5rem 0rem;
        display: flex;
        justify-content: center;
        align-items: center;
        position: sticky;
        top: 0;
        z-index: 999;
        font-family: 'Inter', sans-serif;
    }
    .header-logo {
        color: #d4af37; /* Gold accent */
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: 2px;
    }
    
    /* Hero Section */
    .hero-section {
        background-color: #f8f9fa;
        padding: 4rem 2rem;
        text-align: center;
        border-bottom: 1px solid #eaeaea;
        margin-bottom: 3rem;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        color: #111;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #666;
        font-weight: 400;
    }
    
    /* Product Card Style for Prediction */
    .product-card {
        background: #ffffff;
        border: 1px solid #f0f0f0;
        border-radius: 12px;
        padding: 3rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        margin: 2rem auto;
        max-width: 800px;
    }
    .product-brand {
        text-transform: uppercase;
        font-size: 0.8rem;
        font-weight: 600;
        color: #888;
        letter-spacing: 2px;
    }
    .product-name {
        font-size: 2.5rem;
        font-weight: 700;
        color: #111;
        margin: 0.5rem 0;
    }
    .product-price {
        font-size: 2rem;
        font-weight: 600;
        color: #d4af37; /* Gold price */
        margin-top: 1rem;
    }
    
    /* Section Titles */
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #111;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #111;
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    /* Input Styling overrides matching minimalist theme */
    div[data-baseweb="select"] > div {
        border-radius: 8px;
    }
    
</style>
""", unsafe_allow_html=True)

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
brands = list(encoders['Brand'].classes_)
conditions = list(encoders['Condition'].classes_)
locations = list(encoders['location'].classes_)

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
        if len(available_models) == 0:
            available_models = list(encoders['Model'].classes_)
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
        
        # Francium.lk Product Card Style
        st.markdown(f"""
        <div class="product-card">
            <div class="product-brand">{selected_brand}</div>
            <div class="product-name">{selected_model} {selected_storage}GB ({selected_condition})</div>
            <div style="margin: 1rem 0; color: #888;">Location: {selected_location}</div>
            <div class="product-price">Rs {pred:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br><hr><br>", unsafe_allow_html=True)
        
        # Explanations Grid
        exp_col1, exp_col2 = st.columns(2, gap="large")
        
        with exp_col1:
            st.markdown('### How did we calculate this?')
            st.markdown("The waterfall chart below breaks down exactly which features increased (red) or decreased (blue) the price from the market average.")
            
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
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#3366cc"}},
                increasing={"marker": {"color": "#dc3912"}},
                totals={"marker": {"color": "#d4af37"}}
            ))
            fig_waterfall.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_waterfall, use_container_width=True)
            
        with exp_col2:
            st.markdown('### Global Market Variables')
            st.markdown("Across all models tracked in Sri Lanka, the chart below dictates which overarching features dictate phone prices.")
            
            importance_df = pd.DataFrame({
                'Feature': [f.split("_")[0] for f in model.feature_names_in_], # Clean names
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=True)
    
            fig_global = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                color='Importance', color_continuous_scale='darkmint') # Clean green/dark scale
            fig_global.update_layout(margin=dict(l=0, r=0, t=20, b=0), coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_global, use_container_width=True)

