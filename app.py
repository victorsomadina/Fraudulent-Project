from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import http
from datetime import datetime

# 1: Load model and preprocessor: utilize in data transformation and making prediction
# 2: Instantiate our fastapi app
# 3 validate our incoming/input data
# 4: write an api function that handles prediction
   # 4.1: convert to dataframe 
   # 4.2: write the structure to engineer new features
   # 4.3 Prepocess
   # 4.4: Make Predictions and return results
   # 4.5: decorating a function (@) with fastapi


model = joblib.load('EDA/rf_model.joblib')
preprocessor = joblib.load('EDA/preprocessor.joblib')

app = FastAPI(title = "Fraud Detection API", version="1.0.0")

class Input(BaseModel):
   home_country: str = Field(example='us')
   source_currency: str = Field(example="usd")
   dest_currency: str = Field(example="cad")
   channel: str = Field(example="web")
   amount_src: float = Field(example=278.10)
   amount_usd: float = Field(example=278.10)
   fee: float = Field(example=4.20)
   exchange_rate_src_to_dest: float = Field(example=1.0)
   new_device: bool = Field(example=0)
   ip_country: str = Field(example="us")
   location_mismatch: bool = Field(example=1)
   ip_risk_score: float = Field(example=0.13)
   kyc_tier: str = Field(example="standard")
   account_age_days: int = Field(example=260)
   device_trust_score: float = Field(example=0.5)
   chargeback_history_count: int = Field(example=0)
   risk_score_internal: float = Field(example=0.22)
   txn_velocity_1h: int = Field(example=0)
   txn_velocity_24h: int = Field(example=0)
   corridor_risk: float = Field(example=0.0)
   timestamp: datetime = Field(example="2026-10-16T9:30:00")

@app.get("/")
def welcome():
   return "Welcome to Fraud Prediction API"

@app.post("/predict")
def predict_fraud(input: Input) -> dict:
   data = pd.DataFrame([input.model_dump()])

   data['amount_bucket'] = pd.cut(data['amount_usd'],
                             bins=[0, 100, 500, 1000, 2000, 5000, 20000],
                             labels=['<$100', '$100-500', '$500-1k', '$1k-2k', '$2k-5k', '>$5k'])
   data['age_bucket'] = pd.cut(data['account_age_days'],
                          bins=[0, 30, 90, 180, 365, 2000],
                          labels=['<30d', '30-90d', '90-180d', '180-365d', '>1yr'])
   data['hour'] = data['timestamp'].dt.hour
   data['day_of_week'] = data['timestamp'].dt.dayofweek
   data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

   data = preprocessor.transform(data)

   prediction = int(model.predict(data)[0])
   print(f"This is our prediction {prediction}")

   return {
      "prediction": prediction,
      "label": "fraud" if prediction == 1 else "legit" 
   }