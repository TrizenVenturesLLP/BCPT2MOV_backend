
import streamlit as st
import pandas as pd
import joblib
import os
import datetime

# --- Configuration ---
st.set_page_config(page_title="Agri Market Insights", layout="wide")

MODEL_DIR = 'models'
PRICE_MODEL_PATH = os.path.join(MODEL_DIR, 'price_model.pkl')
PRICE_ENCODERS_PATH = os.path.join(MODEL_DIR, 'price_encoders.pkl')
DEMAND_MODEL_PATH = os.path.join(MODEL_DIR, 'demand_model.pkl')
DEMAND_ENCODERS_PATH = os.path.join(MODEL_DIR, 'demand_encoders.pkl')

# --- Helper Functions ---
@st.cache_resource
def load_artifacts(model_path, encoder_path):
    try:
        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path)
        return model, encoders
    except:
        return None, None

def safe_transform(encoder, value):
    try:
        return encoder.transform([value])[0]
    except:
        return 0

# --- Shared Configuration ---
PRICE_FEATURES = [
    'commodity_group', 'crop_type', 'state_name', 'market_location', 
    'quantity_kg', 'quality_grade', 'season', 'transport_cost', 
    'demand_index', 'month', 'day_of_week'
]

DEMAND_FEATURES = [
    'commodity_group', 'crop_type', 'state_name', 'market_location', 'season', 
    'total_quantity_sold', 'avg_price_per_kg', 'historical_demand_7d', 'price_trend_7d', 
    'estimated_production_kg', 'policy_support_score', 'festival_flag', 'weather_index', 
    'month', 'day_of_week'
]

def map_demand_level(value):
    if value <= 100:
        return "Low"
    elif value <= 3960:
        return "Medium"
    else:
        return "High"

# --- Main App ---
st.title("🌾 Agri Market Insights Platform")
st.markdown("Predict **Crop Prices** and **Market Demand** using Machine Learning.")

# Sidebar for Accuracy
if os.path.exists('models/accuracy.txt'):
    with st.sidebar:
        st.header("📊 Model Accuracy")
        with open('models/accuracy.txt', 'r') as f:
            for line in f:
                if 'Accuracy' in line:
                    st.write(line.strip())

tab1, tab2 = st.tabs(["💰 Price Prediction", "📈 Demand Forecasting"])

# --- Tab 1: Price Prediction ---
with tab1:
    st.header("Price Prediction")
    p_model, p_encoders = load_artifacts(PRICE_MODEL_PATH, PRICE_ENCODERS_PATH)
    
    if p_model and p_encoders:
        col1, col2, col3 = st.columns(3)
        with col1:
            date_input = st.date_input("Date", datetime.date.today(), key='p_date')
            commodity = st.selectbox("Commodity Group", p_encoders['commodity_group'].classes_, key='p_comm')
            crop = st.selectbox("Crop Type", p_encoders['crop_type'].classes_, key='p_crop')
            state = st.selectbox("State", p_encoders['state_name'].classes_, key='p_state')
        with col2:
            market = st.selectbox("Market Location", p_encoders['market_location'].classes_, key='p_market')
            season = st.selectbox("Season", p_encoders['season'].classes_, key='p_season')
            quantity = st.number_input("Quantity (kg)", min_value=1.0, value=100.0, key='p_qty')
        with col3:
            quality = st.number_input("Quality Grade (1-5)", min_value=1, max_value=5, value=1, key='p_qual')
            transport = st.number_input("Transport Cost", min_value=0.0, value=20.0, key='p_trans')
            demand_idx = st.slider("Demand Index", 0.0, 2.0, 1.0, 0.1, key='p_dem')

        if st.button("Predict Price"):
            input_data = {
                'commodity_group': safe_transform(p_encoders['commodity_group'], commodity),
                'crop_type': safe_transform(p_encoders['crop_type'], crop),
                'state_name': safe_transform(p_encoders['state_name'], state),
                'market_location': safe_transform(p_encoders['market_location'], market),
                'quantity_kg': quantity,
                'quality_grade': quality,
                'season': safe_transform(p_encoders['season'], season),
                'transport_cost': transport,
                'demand_index': demand_idx,
                'month': date_input.month,
                'day_of_week': date_input.weekday()
            }
            # Explicit feature ordering
            features = pd.DataFrame([input_data])[PRICE_FEATURES]
            prediction = p_model.predict(features)[0]
            st.success(f"### Predicted Price: ₹{prediction:.2f} per kg")
    else:
        st.error("Price model not found. Run `python train_models.py`.")

# --- Tab 2: Demand Forecasting ---
with tab2:
    st.header("Demand Forecasting")
    d_model, d_encoders = load_artifacts(DEMAND_MODEL_PATH, DEMAND_ENCODERS_PATH)
    
    if d_model and d_encoders:
        col1, col2, col3 = st.columns(3)
        with col1:
            d_date = st.date_input("Date", datetime.date.today(), key='d_date')
            d_commodity = st.selectbox("Commodity Group", d_encoders['commodity_group'].classes_, key='d_comm')
            d_crop = st.selectbox("Crop Type", d_encoders['crop_type'].classes_, key='d_crop')
            d_state = st.selectbox("State", d_encoders['state_name'].classes_, key='d_state')
            d_market = st.selectbox("Market Location", d_encoders['market_location'].classes_, key='d_market')
        with col2:
            d_season = st.selectbox("Season", d_encoders['season'].classes_, key='d_season')
            tot_qty = st.number_input("Total Qty Sold", value=1000.0, key='d_tot_qty')
            avg_price = st.number_input("Avg Price", value=50.0, key='d_avg_price')
            hist_dem = st.number_input("Hist. Demand (7d)", value=500.0, key='d_hist')
            price_trend = st.number_input("Price Trend (7d)", value=50.0, key='d_trend')
        with col3:
            est_prod = st.number_input("Est. Production", value=10000.0, key='d_prod')
            policy = st.slider("Policy Score", 0.0, 1.0, 0.5, key='d_pol')
            weather = st.slider("Weather Index", 0.0, 1.0, 0.5, key='d_weath')
            festival = st.selectbox("Festival Flag (0/1)", [0, 1], key='d_fest')

        if st.button("Forecast Demand"):
            input_data = {
                'commodity_group': safe_transform(d_encoders['commodity_group'], d_commodity),
                'crop_type': safe_transform(d_encoders['crop_type'], d_crop),
                'state_name': safe_transform(d_encoders['state_name'], d_state),
                'market_location': safe_transform(d_encoders['market_location'], d_market),
                'season': safe_transform(d_encoders['season'], d_season),
                'total_quantity_sold': tot_qty,
                'avg_price_per_kg': avg_price,
                'historical_demand_7d': hist_dem,
                'price_trend_7d': price_trend,
                'estimated_production_kg': est_prod,
                'policy_support_score': policy,
                'festival_flag': festival,
                'weather_index': weather,
                'month': d_date.month,
                'day_of_week': d_date.weekday()
            }
            # Explicit feature ordering
            features = pd.DataFrame([input_data])[DEMAND_FEATURES]
            prediction = d_model.predict(features)[0]
            category = map_demand_level(prediction)
            
            st.success(f"### Forecasted Demand Level: {category}")
            st.info(f"Numeric Demand Score: {prediction:.2f}")
    else:
        st.error("Demand model not found. Run `python train_models.py`.")
