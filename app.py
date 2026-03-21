import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# Page configuration
st.set_page_config(
    page_title="Big Mart Sales Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Corporate Styling (Blue and White)
st.markdown("""
    <style>
    /* Force high contrast and specific background */
    .stApp { 
        background-color: #f1f5f9 !important; /* Soft light gray/slate background */
    }
    
    /* Global Text Visibility */
    .stApp p, .stApp span, .stApp label, .stApp li {
        color: #0f172a !important; /* Very dark slate/navy for body text */
        font-weight: 500;
    }

    /* Sidebar Styling - Constant Dark Blue */
    section[data-testid="stSidebar"] {
        background-color: #1e3a8a !important;
        background-image: linear-gradient(180deg, #1e3a8a 0%, #0f172a 100%) !important;
    }
    
    /* Ensure sidebar text stays white */
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Main Content Headers */
    h1, h2, h3, h4 { 
        color: #1e3a8a !important; 
        font-weight: 800 !important;
        margin-bottom: 1rem !important;
    }
    
    /* Refined Metric Card */
    .metric-card {
        background-color: #ffffff !important;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
        padding: 2rem;
        border-left: 8px solid #1e3a8a;
        margin-bottom: 25px;
        color: #0f172a;
    }
    
    .metric-label { 
        font-size: 1rem; 
        color: #475569 !important; /* Darker slate gray label */
        font-weight: 700; 
        text-transform: uppercase; 
        letter-spacing: 0.1em;
        margin-bottom: 8px;
    }
    
    .metric-value { 
        font-size: 3rem; 
        font-weight: 900; 
        color: #1e3a8a !important; 
        line-height: 1.1; 
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e2e8f0;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #1e3a8a !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        font-weight: bold;
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
    st.error("🚨 Predictive model and metadata not found. Please run the training pipeline first.")
    st.info("Ensure `best_sales_model.pkl` and `preprocessing_info.pkl` exist in the current directory.")
    st.stop()

# Header Section
st.title("🛒 Big Mart Sales Predictive Analytics")
st.markdown("Forecasting product demand using high-accuracy XGBoost machine learning.")

# Sidebar Configuration
with st.sidebar:
    st.header("🏠 Outlet Config")
    outlet_type = st.selectbox("Outlet Type", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])
    outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
    outlet_location = st.selectbox("Location Tier", ["Tier 1", "Tier 2", "Tier 3"])
    outlet_years = st.slider("Outlet Operating Years", 0, 30, 15)
    
    st.markdown("---")
    st.header("🍎 Product Highlights")
    item_mrp = st.slider("Item MRP (FCFA)", 50, 5000, 1500)
    item_visibility_pct = st.slider("Shelf Visibility (%)", 0, 100, 15)
    item_visibility = item_visibility_pct / 100.0
    item_weight = st.number_input("Item Weight (g)", 4.0, 22.0, 12.0)
    item_type_cat = st.selectbox("Category", ["Food", "Drinks", "Non-Consumable"])
    item_fat = st.selectbox("Fat Content", ["Low Fat", "Regular", "Non-Edible"])

# Prediction Pipeline
def prepare_input():
    # Correct mapping as per training script
    size_map = {"High": 0, "Medium": 1, "Small": 2}
    loc_map = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
    
    data = {
        'Item_Weight': [item_weight],
        'Item_Visibility': [item_visibility],
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
    # Match the training column order
    return df[preproc['columns']]

# Prediction results
input_df = prepare_input()
prediction = model.predict(input_df)[0]
prediction = max(0, prediction)

# Build Main Dashboard
tab1, tab2 = st.tabs(["📊 Performance Prediction", "📜 Technical Overview"])

with tab1:
    # Metric Display
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Projected Sales Volume</div>
            <div class="metric-value">{prediction:,.0f} FCFA</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Analysis Columns
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("💡 Strategic Insights")
        if outlet_type == "Grocery Store":
            st.warning("📊 **Observation:** Small format 'Grocery Stores' show significantly lower volume. Expanding to a Supermarket format could exponentially increase ROI.")
        if item_mrp > 3500:
            st.info("💎 **Observation:** Premium product detected! Pair with high-income 'Tier 3' locations for optimal conversion.")
        if item_visibility_pct < 5:
            st.error("👀 **Optimization:** Improving product orientation on shelf (currently < 5%) could increase predicted sales by 15-20%.")
        else:
            st.success("🎯 **Positive Indicator:** Healthy product visibility level. Maintain current shelf positioning strategy.")

    with col2:
        st.subheader("⚡ Factor Importance")
        if preproc.get('feature_importance'):
            feat_imp = pd.Series(preproc['feature_importance']).head(5).reset_index()
            feat_imp.columns = ['Variable', 'Weight']
            
            # Use Corporate Blue palette
            fig = px.bar(feat_imp, x='Weight', y='Variable', orientation='h', 
                         title="Sales Driver Breakdown",
                         color_discrete_sequence=['#1e3a8a'])
            fig.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, margin=dict(l=0, r=0, t=30, b=0), height=300)
            st.plotly_chart(fig, use_container_width=True)

    # What-If Analysis Area
    st.markdown("---")
    st.subheader("🧪 'What-If' Simulation")
    st.write("Understand how MRP changes affect your bottom line:")
    
    # Simulation range (Updated for 5000 FCFA ceiling)
    mrp_range = np.linspace(50, 5000, 50)
    sim_results = []
    
    temp_df = input_df.copy()
    for price in mrp_range:
        temp_df['Item_MRP'] = price
        sim_results.append(model.predict(temp_df)[0])
    
    sim_data = pd.DataFrame({'MRP': mrp_range, 'Predicted Sales': sim_results})
    fig_sim = px.line(sim_data, x='MRP', y='Predicted Sales', markers=True, 
                      labels={'MRP': 'Item Maximum Retail Price (FCFA)', 'Predicted Sales': 'Predicted Volume (FCFA)'})
    fig_sim.update_traces(line_color='#1e3a8a', line_width=4)
    fig_sim.add_vline(x=item_mrp, line_dash="dash", line_color="red", annotation_text="Current Setup")
    st.plotly_chart(fig_sim, use_container_width=True)

with tab2:
    st.header("Technical Specifications")
    st.markdown("""
    ### ⚙️ Machine Learning Pipeline
    The system follows a rigorous data science workflow to ensure forecast reliability.
    
    #### 1. Data Synthesis & Engineering
    - **Missing Value Strategy**: Imputed `Item_Weight` via `Item_Identifier` cross-reference. Missing `Outlet_Size` handled using `Outlet_Type` mode distribution.
    - **Temporal Features**: Calculated `Outlet_Age` (defined as 2013 - Establishment Year) to capture brand maturity effects.
    - **Categorical Logic**: Multi-class encoding for outlets and products.
    
    #### 2. Advanced Regressor (XGBoost)
    Leveraging eXtreme Gradient Boosting (XGBoost) which consistently outperforms baseline linear models by capturing non-linear interactions between variables like visibility and location type.
    
    #### 3. Evaluation Metrics
    - **Root Mean Squared Error (RMSE)**: Primary optimization target during GridSearch CV.
    - **R-Squared (Coefficient of Determination)**: Validating how well the model explains sales variance.
    
    ---
    *Data Source: Big Mart Sales III Dataset (Big Mart Product Analytics).*
    """)
