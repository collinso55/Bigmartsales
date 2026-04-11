import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Big Mart Elite Sales Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Redesign - Deep Burgundy (#80011f) and Ivory Cream (#fefaee)
st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;600&display=swap');
    
    .stApp { 
        background-color: #fefaee !important; /* Ivory Cream Background */
    }
    
    /* Typography - Excluding icons from font override */
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        color: #2d2d2a !important;
    }
    
    /* Ensure Streamlit's internal icons (like the sidebar toggle) 
       don't get turned into text by our global font settings */
    span[data-testid="stIconMaterial"], 
    .st-emotion-cache-1vt4y69, 
    [data-testid="stSidebarNav"] * {
        font-family: inherit !important;
    }

    /* Professional Redesign - Deep Burgundy (#80011f) and Ivory Cream (#fefaee) */
    section[data-testid="stSidebar"] {
        background-color: #80011f !important;
        border-right: 2px solid #fefaee;
    }
    
    section[data-testid="stSidebar"] * {
        color: #fefaee !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Professional Module Headers */
    .sidebar-mod-header {
        font-size: 0.75rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.15em !important;
        text-transform: uppercase !important;
        color: #fefaee !important;
        margin: 2rem 0 1rem 0 !important;
        opacity: 0.8;
        border-bottom: 1px solid rgba(254, 250, 238, 0.2);
        padding-bottom: 5px;
    }

    /* Clean Input Styling */
    section[data-testid="stSidebar"] div[data-baseweb="select"],
    section[data-testid="stSidebar"] input {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(254, 250, 238, 0.3) !important;
        border-radius: 4px !important;
    }

    /* Expander Reconstruction - Professional Info Buttons */
    section[data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid rgba(254, 250, 238, 0.4) !important;
        border-radius: 4px !important;
        background-color: rgba(254, 250, 238, 0.05) !important;
        margin-bottom: 5px !important;
    }

    section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    /* Enhanced Toggle Button - High Priority Overrides */
    header[data-testid="stHeader"] button {
        background-color: #80011f !important;
        color: #fefaee !important;
        border-radius: 2px !important;
        width: 120px !important;
        height: 38px !important;
        z-index: 999999 !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15) !important;
        border: 1px solid #fefaee !important;
    }

    /* Hide ALL internal text/icons */
    header[data-testid="stHeader"] button span,
    header[data-testid="stHeader"] button div {
        display: none !important;
    }

    header[data-testid="stHeader"] button[aria-label="Open sidebar"]::after {
        content: "CONFIGURATION";
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        display: block !important;
        letter-spacing: 0.1em;
    }

    header[data-testid="stHeader"] button[aria-label="Close sidebar"]::after {
        content: "SAVE & CLOSE";
        font-size: 0.7rem !important;
        font-weight: 700 !important;
        display: block !important;
        letter-spacing: 0.1em;
    }
    
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Clean Header */
    h1 { 
        font-family: 'Playfair Display', serif;
        color: #80011f !important; 
        font-size: 3rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Result Section - Deep Burgundy Highlight over Ivory */
    .prediction-container {
        background-color: #80011f !important;
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        border: 2px solid #80011f;
        box-shadow: 0 15px 35px rgba(128, 1, 31, 0.1);
    }
    
    .prediction-title { 
        font-size: 1.2rem;
        color: #fefaee;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .prediction-value { 
        font-size: 4.5rem; 
        font-weight: 900; 
        color: #fefaee !important; 
        line-height: 1;
        margin-top: 0;
    }

    /* Professional Insight Cards */
    .insight-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #80011f;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    
    .insight-header {
        color: #80011f;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 5px;
    }

    /* Mobile Adjustments */
    @media (max-width: 640px) {
        .prediction-value { font-size: 3rem !important; }
        h1 { font-size: 2rem !important; }
    }
    </style>
""", unsafe_allow_html=True)

# Assets loader
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('best_sales_model.pkl')
        preproc = joblib.load('preprocessing_info.pkl')
        return model, preproc
    except Exception as e:
        return None, None

model, preproc = load_assets()

if model is None:
    st.error("🚨 System Assets Missing. Please run training pipeline.")
    st.stop()

# Header Area
st.title("Big Mart Elite")

# Button-styled call to action for the sidebar
st.markdown("""
    <div style="
        display: inline-block;
        background-color: #80011f;
        color: #fefaee;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.1em;
        margin-bottom: 2rem;
        box-shadow: 0 4px 10px rgba(128, 1, 31, 0.2);
    ">
        GENERATE INSIGHTS ↓ ADJUST PARAMETERS IN SIDEBAR
    </div>
""", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;"><h2 style="color: #fefaee; font-family: Playfair Display; font-size: 1.5rem;">CONTROL PANEL</h2></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-mod-header">Operational Parameters</div>', unsafe_allow_html=True)
    
    with st.expander("INFO: Outlet Classification"):
        st.write("Classification based on inventory and regional capacity. **Impact:** Direct scale factor for high-volume supermarkets.")
    outlet_type = st.selectbox("Outlet Classification", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])
    
    with st.expander("INFO: Territory Location"):
        st.write("Regional market tier. **Impact:** High-tier urban centers correlate with higher purchasing power in current logic.")
    outlet_location = st.selectbox("Territory Location", ["Tier 1", "Tier 2", "Tier 3"])
    
    with st.expander("INFO: Floor Space"):
        st.write("Physical square footage. **Impact:** Medium and High sizes provide more stable forecasts.")
    outlet_size = st.selectbox("Floor Space", ["Small", "Medium", "High"])
    
    outlet_years = st.slider("Market Presence (Years)", 0, 30, 15)
    
    st.markdown('<div class="sidebar-mod-header">Product Specifications</div>', unsafe_allow_html=True)
    
    with st.expander("INFO: Product Price (FCFA)"):
        st.write("Unit price point. **Impact:** Primary driver of total sales revenue calculation.")
    item_mrp = st.slider("Product Price (FCFA)", 50, 5000, 1500)
    
    with st.expander("INFO: Shelf Prominence"):
        st.write("Display visibility. **Impact:** Percentage-based uplift for impulse purchase likelihood.")
    item_visibility_pct = st.slider("Shelf Prominence (%)", 0, 100, 15)
    
    item_weight = st.number_input("Unit Weight (g)", 4.0, 22.0, 12.0)
    item_type_cat = st.selectbox("Inventory Category", ["Food", "Drinks", "Non-Consumable"])
    item_fat = st.selectbox("Health Rating (Fat)", ["Low Fat", "Regular", "Non-Edible"])

# Pure Prediction Logic
def get_prediction():
    size_map = {"High": 0, "Medium": 1, "Small": 2}
    loc_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
    
    data = {
        'Item_Weight': [item_weight],
        'Item_Visibility': [item_visibility_pct / 100.0],
        'Item_MRP': [item_mrp],
        'Outlet_Size_Label': [size_map[outlet_size]],
        'Outlet_Location_Type_Label': [loc_map[outlet_location]],
        'Outlet_Years': [outlet_years],
        'Item_Fat_Content_Low Fat': [1 if item_fat == "Low Fat" else 0],
        'Item_Fat_Content_Non-Edible': [1 if item_fat == "Non-Edible" else 0],
        'Item_Fat_Content_Regular': [1 if item_fat == "Regular" else 0],
        'Outlet_Type_Grocery Store': [1 if outlet_type == "Grocery Store" else 0],
        'Outlet_Type_Supermarket Type1': [1 if outlet_type == "Supermarket Type1" else 0],
        'Outlet_Type_Supermarket Type2': [1 if outlet_type == "Supermarket Type2" else 0],
        'Outlet_Type_Supermarket Type3': [1 if outlet_type == "Supermarket Type3" else 0],
        'Item_Type_Combined_Drinks': [1 if item_type_cat == "Drinks" else 0],
        'Item_Type_Combined_Food': [1 if item_type_cat == "Food" else 0],
        'Item_Type_Combined_Non-Consumable': [1 if item_type_cat == "Non-Consumable" else 0],
    }
    
    df = pd.DataFrame(data)
    return max(0, model.predict(df[preproc['columns']])[0])

# Direct Result Presentation
prediction = get_prediction()

st.markdown(f"""
    <div class="prediction-container">
        <div class="prediction-title">Estimated Sales Performance</div>
        <div class="prediction-value">{prediction:,.0f} FCFA</div>
    </div>
""", unsafe_allow_html=True)

# Straightforward Intelligence (Text-based, no graphs)
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Strategic Diagnosis")
    
    if outlet_type == "Grocery Store":
        st.markdown(f"""
            <div class="insight-card">
                <div class="insight-header">Format Restriction</div>
                Current 'Grocery Store' format limits turnover potential by approx 60% compared to Supermarket models.
            </div>
        """, unsafe_allow_html=True)
        
    if item_visibility_pct < 10:
        st.markdown(f"""
            <div class="insight-card">
                <div class="insight-header">Visibility Deficit</div>
                Shelf prominence is below optimal baseline. Increasing visibility to 15%+ is recommended for this category.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="insight-card">
                <div class="insight-header">Optimized Positioning</div>
                Product visibility is correctly aligned with high-performance retail standards.
            </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("### Operational Summary")
    
    summary_text = f"""
    The analysis considers a **{item_type_cat}** product priced at **{item_mrp:,} FCFA** within a **{outlet_size}** sized establishment. 
    Based on the **{outlet_years} years** of market presence in a **{outlet_location}** territory, the predictive engine expects 
    the turnover volume to remain stable at the projected level.
    """
    st.info(summary_text)

# Footer
st.markdown("---")
st.caption("Elite Predictive System | Powered by Advanced Regression Analytics | localized for Cameroon")
