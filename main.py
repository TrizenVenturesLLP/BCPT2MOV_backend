from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

# ... (imports remain)


import pandas as pd
import joblib
import os
import datetime
from typing import List, Optional

from database import engine, get_db
import models
import sqlite3
import os

# Create tables
models.Base.metadata.create_all(bind=engine)

# Migration: Add transport_cost column if it doesn't exist
DB_PATH = 'agri_insights.db'
if os.path.exists(DB_PATH):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(products)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'transport_cost' not in columns:
            print("Running migration: Adding transport_cost column...")
            cursor.execute("ALTER TABLE products ADD COLUMN transport_cost REAL")
            conn.commit()
            print("✓ Migration successful!")
        conn.close()
    except Exception as e:
        print(f"Migration warning: {e}")

app = FastAPI()

# --- CORS Configuration ---
# Get frontend URL from env or default to deployed frontend
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://bcpt2mov-frontend.llp.trizenventures.com")

# More permissive CORS for debugging - allow all origins temporarily
# In production, restrict to specific domains
CORS_ORIGINS = [
    FRONTEND_URL,
    "https://bcpt2mov-frontend.llp.trizenventures.com",  # Explicit
    "http://bcpt2mov-frontend.llp.trizenventures.com",  # HTTP fallback
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "*",  # Temporary: allow all for debugging
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily allow all to test if origin matching is the issue
    allow_credentials=False,  # Set to False when using wildcard
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

MODEL_DIR = 'models'
PRICE_MODEL_PATH = os.path.join(MODEL_DIR, 'price_model.pkl')
PRICE_ENCODERS_PATH = os.path.join(MODEL_DIR, 'price_encoders.pkl')
DEMAND_MODEL_PATH = os.path.join(MODEL_DIR, 'demand_model.pkl')
DEMAND_ENCODERS_PATH = os.path.join(MODEL_DIR, 'demand_encoders.pkl')

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

# --- Models and Encoders loading ---
def load_artifacts(model_path, encoder_path):
    try:
        model = joblib.load(model_path)
        encoders = joblib.load(encoder_path)
        return model, encoders
    except Exception as e:
        print(f"Error loading {model_path}: {e}")
        return None, None

p_model, p_encoders = load_artifacts(PRICE_MODEL_PATH, PRICE_ENCODERS_PATH)
d_model, d_encoders = load_artifacts(DEMAND_MODEL_PATH, DEMAND_ENCODERS_PATH)

def safe_transform(encoder, value):
    try:
        return encoder.transform([value])[0]
    except:
        # Fallback for unknown categories
        return 0

def map_demand_level(value):
    if value <= 100:
        return "Low"
    elif value <= 3960:
        return "Medium"
    else:
        return "High"

# --- Request Models ---
class PricePredictRequest(BaseModel):
    date: str  # YYYY/MM/DD
    commodity_group: str
    crop_type: str
    state_name: str
    market_location: str
    quantity_kg: float
    quality_grade: int
    season: str
    transport_cost: float
    demand_index: float

class DemandForecastRequest(BaseModel):
    date: str # YYYY/MM/DD
    commodity_group: str
    crop_type: str
    state_name: str
    market_location: str
    season: str
    total_quantity_sold: float
    avg_price_per_kg: float
    historical_demand_7d: float
    price_trend_7d: float
    estimated_production_kg: float
    policy_support_score: float
    festival_flag: int
    weather_index: float

# --- DB Request Models ---
class UserLoginRequest(BaseModel):
    wallet_address: str
    role: str # 'farmer', 'buyer', 'logistics'
    signature: Optional[str] = None # For future verification

class ProductCreateRequest(BaseModel):
    id: str # Generating on frontend for now to keep it simple
    crop_type: str
    commodity_group: str
    quantity: float
    price: float
    suggested_price: float
    quality_grade: int
    market_location: str
    state: str
    season: str
    farmer_address: str
    transport_cost: float

class OrderCreateRequest(BaseModel):
    id: str
    product_id: str
    buyer_address: str
    quantity: float
    total_price: float
    farmer_location: str
    buyer_location: str
    tx_hash: str
    estimated_delivery: Optional[str] = None 

class OrderStatusUpdate(BaseModel):
    status: str

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"message": "Agri Predict API is running"}

@app.get("/health")
def health_check():
    """Health check endpoint with CORS headers"""
    return {
        "status": "ok",
        "cors_enabled": True,
        "frontend_url": FRONTEND_URL
    }

@app.post("/predict_price")
def predict_price(req: PricePredictRequest):
    if not p_model or not p_encoders:
        raise HTTPException(status_code=500, detail="Price model not loaded")
    
    try:
        dt = datetime.datetime.strptime(req.date, "%Y/%m/%d")
    except:
        dt = datetime.date.today()

    input_data = {
        'commodity_group': safe_transform(p_encoders['commodity_group'], req.commodity_group),
        'crop_type': safe_transform(p_encoders['crop_type'], req.crop_type),
        'state_name': safe_transform(p_encoders['state_name'], req.state_name),
        'market_location': safe_transform(p_encoders['market_location'], req.market_location),
        'quantity_kg': req.quantity_kg,
        'quality_grade': req.quality_grade,
        'season': safe_transform(p_encoders['season'], req.season),
        'transport_cost': req.transport_cost,
        'demand_index': req.demand_index,
        'month': dt.month,
        'day_of_week': dt.weekday()
    }
    
    features = pd.DataFrame([input_data])[PRICE_FEATURES]
    prediction = p_model.predict(features)[0]
    return {"predicted_price": round(float(prediction), 2)}

@app.post("/forecast_demand")
def forecast_demand(req: DemandForecastRequest):
    if not d_model or not d_encoders:
        raise HTTPException(status_code=500, detail="Demand model not loaded")
    
    try:
        dt = datetime.datetime.strptime(req.date, "%Y/%m/%d")
    except:
        dt = datetime.date.today()

    input_data = {
        'commodity_group': safe_transform(d_encoders['commodity_group'], req.commodity_group),
        'crop_type': safe_transform(d_encoders['crop_type'], req.crop_type),
        'state_name': safe_transform(d_encoders['state_name'], req.state_name),
        'market_location': safe_transform(d_encoders['market_location'], req.market_location),
        'season': safe_transform(d_encoders['season'], req.season),
        'total_quantity_sold': req.total_quantity_sold,
        'avg_price_per_kg': req.avg_price_per_kg,
        'historical_demand_7d': req.historical_demand_7d,
        'price_trend_7d': req.price_trend_7d,
        'estimated_production_kg': req.estimated_production_kg,
        'policy_support_score': req.policy_support_score,
        'festival_flag': req.festival_flag,
        'weather_index': req.weather_index,
        'month': dt.month,
        'day_of_week': dt.weekday()
    }
    
    features = pd.DataFrame([input_data])[DEMAND_FEATURES]
    prediction = d_model.predict(features)[0]
    category = map_demand_level(prediction)
    
    return {
        "demand_score": round(float(prediction), 2),
        "demand_level": category
    }

# --- DB Endpoints ---

@app.post("/auth/login")
def login_or_register(req: UserLoginRequest, db: Session = Depends(get_db)):
    # In Phase 1, we trust the frontend wallet address. IN Phase 2, verify signature.
    db_user = db.query(models.User).filter(models.User.wallet_address == req.wallet_address).first()
    if not db_user:
        db_user = models.User(wallet_address=req.wallet_address, role=req.role)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
    return {"status": "ok", "role": db_user.role, "wallet_address": db_user.wallet_address}

@app.get("/users/{wallet_address}")
def get_user_role(wallet_address: str, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.wallet_address == wallet_address).first()
    if not db_user:
        return {"role": None}
    return {"role": db_user.role}

@app.post("/products")
def create_product(req: ProductCreateRequest, db: Session = Depends(get_db)):
    db_product = models.Product(**req.dict())
    db.add(db_product)
    db.commit()
    return {"status": "ok", "id": req.id}

@app.get("/products")
def get_products(db: Session = Depends(get_db)):
    # Return all unsold products for the marketplace
    return db.query(models.Product).filter(models.Product.is_sold == False).all()

@app.post("/orders")
def create_order(req: OrderCreateRequest, db: Session = Depends(get_db)):
    # 1. Create Order
    est_delivery = datetime.datetime.now() + datetime.timedelta(days=3) # Mock delivery time
    
    db_order = models.Order(
        id=req.id,
        product_id=req.product_id,
        buyer_address=req.buyer_address,
        quantity=req.quantity,
        total_price=req.total_price,
        farmer_location=req.farmer_location,
        buyer_location=req.buyer_location,
        tx_hash=req.tx_hash,
        estimated_delivery=est_delivery
    )
    db.add(db_order)
    
    # 2. Mark Product as Sold
    db_product = db.query(models.Product).filter(models.Product.id == req.product_id).first()
    if db_product:
        db_product.is_sold = True
    
    db.commit()
    return {"status": "ok", "order_id": req.id}

@app.get("/orders")
def get_orders(db: Session = Depends(get_db)):
    # Use joinedload to efficiently fetch the related product
    orders = db.query(models.Order).options(joinedload(models.Order.product)).all()
    
    result = []
    for o in orders:
        # Construct response dict with product info
        result.append({
            "id": o.id,
            "product_id": o.product_id,
            "buyer_address": o.buyer_address,
            "quantity": o.quantity,
            "total_price": o.total_price,
            "farmer_location": o.farmer_location,
            "buyer_location": o.buyer_location,
            "status": o.status,
            "tx_hash": o.tx_hash,
            "created_at": o.created_at,
            "estimated_delivery": o.estimated_delivery,
            # Add the product name (crop_type) from the relationship
            "product_name": o.product.crop_type if o.product else "Unknown Crop",
            "farmer_address": o.product.farmer_address if o.product else "Unknown"
        })
    return result

@app.get("/orders/{wallet_address}")
def get_my_orders(wallet_address: str, db: Session = Depends(get_db)):
    # Get orders where user is buyer OR orders where user is farmer (via product)
    # For simplicity in this prototype, just fetching where buyer matches
    return db.query(models.Order).filter(models.Order.buyer_address == wallet_address).all()

@app.patch("/orders/{order_id}/status")
def update_order_status(order_id: str, req: OrderStatusUpdate, db: Session = Depends(get_db)):
    db_order = db.query(models.Order).filter(models.Order.id == order_id).first()
    if not db_order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    db_order.status = req.status
    db.commit()
    return {"status": "updated"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

