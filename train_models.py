
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Ensure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

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

def train_price_model():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
    
    print("Training Price Model...")
    df = pd.read_csv('price_suggestion_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    target = 'selling_price_per_kg'
    
    encoders = {}
    for col in ['commodity_group', 'crop_type', 'state_name', 'market_location', 'season']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    X = df[PRICE_FEATURES]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    accuracy = max(0, 100 * (1 - mape))
    
    joblib.dump(model, 'models/price_model.pkl')
    joblib.dump(encoders, 'models/price_encoders.pkl')
    
    return f"Price Model Accuracy: {accuracy:.2f}%"

def train_demand_model():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
    
    print("Training Demand Model...")
    df = pd.read_csv('demand_forecasting_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    target = 'demand_level'
    
    encoders = {}
    for col in ['commodity_group', 'crop_type', 'state_name', 'market_location', 'season']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    X = df[DEMAND_FEATURES]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    accuracy = max(0, 100 * (1 - mape))
    
    joblib.dump(model, 'models/demand_model.pkl')
    joblib.dump(encoders, 'models/demand_encoders.pkl')
    
    return f"Demand Model Accuracy: {accuracy:.2f}%"

if __name__ == "__main__":
    p_acc = train_price_model()
    d_acc = train_demand_model()
    
    with open('models/accuracy.txt', 'w') as f:
        f.write(p_acc + "\n")
        f.write(d_acc + "\n")
    
    print("Training Complete. Metrics saved to models/accuracy.txt")
    print(p_acc)
    print(d_acc)
