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

    /* Sidebar Styling - Deep Burgundy (#80011f) and Fixed-feel */
    section[data-testid="stSidebar"] {
        background-color: #80011f !important;
        border-right: 1px solid #fefaee;
        overflow-x: hidden !important; /* Strictly no horizontal scrolling */
    }
    
    /* Hide scrollbar for sidebar while keeping functionality */
    section[data-testid="stSidebar"] > div {
        overflow-y: auto;
        overflow-x: hidden !important;
        -ms-overflow-style: none; /* IE and Edge */
        scrollbar-width: none; /* Firefox */
    }
    section[data-testid="stSidebar"] > div::-webkit-scrollbar {
        display: none; /* Chrome, Safari and Opera */
    }
    
    /* Sidebar Text & Inputs - Condensed for 'Fixed' feel */
    section[data-testid="stSidebar"] * {
        color: #fefaee !important;
    }
    
    section[data-testid="stSidebar"] .stSelectbox, 
    section[data-testid="stSidebar"] .stSlider {
        margin-bottom: -15px !important;
    }

    /* Enhanced Native Toggle Button (The open/close button) */
    button[data-testid="stBaseButton-header"],
    button[aria-label="Open sidebar"],
    button[aria-label="Close sidebar"] {
        background-color: #80011f !important;
        color: #fefaee !important;
        border-radius: 8px !important;
        width: auto !important;
        min-width: 80px !important;
        height: 40px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2) !important;
        font-size: 0 !important; /* Hide the weird internal text */
    }
    
    button[aria-label="Open sidebar"]::after {
        content: "OPTIONS";
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    button[aria-label="Close sidebar"]::after {
        content: "CLOSE";
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    [data-testid="stHeader"] {
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

# Sidebar - Elegant Input Controls
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/efc9e3/shop.png", width=80)
    st.markdown("### OPERATIONAL PARAMETERS")
    
    outlet_type = st.selectbox("Outlet Classification", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])
    outlet_size = st.selectbox("Floor Space (Size)", ["Small", "Medium", "High"])
    outlet_location = st.selectbox("Territory Location", ["Tier 1", "Tier 2", "Tier 3"])
    outlet_years = st.slider("Market Presence (Years)", 0, 30, 15)
    
    st.markdown("---")
    st.markdown("### PRODUCT SPECIFICATIONS")
    item_mrp = st.slider("Product Price (FCFA)", 50, 5000, 1500)
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
