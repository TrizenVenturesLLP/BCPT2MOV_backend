from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from database import Base
import datetime

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    wallet_address = Column(String, unique=True, index=True)
    role = Column(String)  # 'farmer', 'buyer', 'logistics'
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Product(Base):
    __tablename__ = "products"

    id = Column(String, primary_key=True, index=True) # Using String ID to match frontend 'PRD-...' format if needed, or we can use Integer and let DB handle it. Let's stick to String to match current frontend generation or switch to DB auto-increment.
    # Frontend generates IDs like `PRD-${Date.now()}`. Let's persist that for now or allow server to generate.
    # To keep it simple and consistent with frontend logic so far, let's allow String IDs.
    
    crop_type = Column(String, index=True)
    commodity_group = Column(String)
    quantity = Column(Float)
    price = Column(Float)
    suggested_price = Column(Float)
    quality_grade = Column(Integer)
    market_location = Column(String)
    state = Column(String)
    season = Column(String)
    farmer_address = Column(String, ForeignKey("users.wallet_address"))
    transport_cost = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_sold = Column(Boolean, default=False)

    # Relationships
    farmer = relationship("User", foreign_keys=[farmer_address])

class Order(Base):
    __tablename__ = "orders"

    id = Column(String, primary_key=True, index=True) # 'ORD-...'
    product_id = Column(String, ForeignKey("products.id"))
    buyer_address = Column(String, ForeignKey("users.wallet_address"))
    quantity = Column(Float)
    total_price = Column(Float)
    farmer_location = Column(String)
    buyer_location = Column(String)
    status = Column(String, default="pending") # 'pending', 'picked_up', 'in_transit', 'delivered'
    tx_hash = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    estimated_delivery = Column(DateTime)

    # Relationships
    product = relationship("Product")
    buyer = relationship("User", foreign_keys=[buyer_address])
