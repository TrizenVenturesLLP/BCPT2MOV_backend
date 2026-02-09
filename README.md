# 🌾 AgriPredict Backend - ML API

This is the machine learning backend for the AgriPredict platform, serving price prediction and demand forecasting models via a **FastAPI** server.

## 🚀 Features

### 1. Price Prediction API (`/predict_price`)
- Predicts selling price per kg for Millets and Oilseeds.
- Uses a **Random Forest Regressor** with ~92% accuracy.

### 2. Demand Forecasting API (`/forecast_demand`)
- Forecasts market demand levels (Low, Medium, High).
- Accounts for weather, festivals, and market trends.
- ~92% accuracy.

## 📂 Backend Structure

- `main.py`: The FastAPI application that serves the models.
- `train_models.py`: Pipeline to train the machine learning models.
- `models/`: Directory containing trained `.pkl` models and encoders.
- `price_suggestion_dataset.csv`: Dataset for price training.
- `demand_forecasting_dataset.csv`: Dataset for demand training.
- `requirements.txt`: Python dependencies.

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.8+ (Created with Python 3.8.10)
- Virtual Environment recommended.

### Installation & Run
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Models**:
   (Required if you update the datasets or switch environments)
   ```bash
   python train_models.py
   ```

3. **Run the API Server**:
   ```bash
   python main.py
   ```
   *The API will be available at [http://localhost:8000](http://localhost:8000).*
   *Documentation at [http://localhost:8000/docs](http://localhost:8000/docs).*

## ⚠️ Note
This backend is designed to be called by the React frontend. Ensure the frontend is configured to call `http://localhost:8000`.

## 🔗 Frontend Repository

**Frontend Repository**: [Link to frontend repo]

The frontend application connects to this API for ML predictions and data management.

## 🌐 API Endpoints

### Authentication
- `POST /auth/login` - Sign up or sign in a user

### ML Predictions
- `POST /predict_price` - Get price prediction for crops
- `POST /forecast_demand` - Get demand forecast

### Products
- `GET /products` - Get all available products
- `POST /products` - Create a new product listing

### Orders
- `GET /orders` - Get all orders
- `POST /orders` - Create a new order
- `PATCH /orders/{order_id}/status` - Update order status

### Users
- `GET /users/{wallet_address}` - Get user role by wallet address

Full API documentation available at `http://localhost:8000/docs` when server is running.

## 🔧 Configuration

### CORS Settings

The backend is configured to allow all origins in development. For production, update `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],  # Update this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```