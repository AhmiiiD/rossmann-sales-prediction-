
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

with open("rossmann_best_model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI(title="Rossmann Sales Prediction API")


class SalesInput(BaseModel):
    year: int
    month: int
    day: int
    weekofyear: int
    quarter: int
    dayofweek: int
    promo: int
    schoolholiday: int
    is_state_holiday: int
    competitiondistance: float
    competition_open_months: float
    has_competition_info: int
    promo2: int
    promo2_active: int
    lag_7: float
    lag_14: float
    rolling_mean_7: float
    rolling_mean_30: float
    days_to_next_holiday: float
    days_since_last_holiday: float
    day_before_holiday: int
    day_after_holiday: int
    stateholiday: str
    storetype: str
    assortment: str
    promointerval: str


@app.get("/")
def home():
    return {"message": "Rossmann API is running"}


@app.post("/predict")
def predict(data: SalesInput):
    df = pd.DataFrame([data.model_dump()])
    prediction = model.predict(df)[0]
    return {"predicted_sales": round(float(prediction), 2)}