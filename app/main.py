import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Model path (Windows)
# --------------------------------------------------
MODEL_PATH = Path(
    r"C:\Users\Rasulbek907\Desktop\LifeSpan_Peoples\Models\GradientBoostingRegressor.joblib"
)

# --------------------------------------------------
# FastAPI app
# --------------------------------------------------
app = FastAPI(
    title="🧬 Life Span Prediction API",
    description="Regression model orqali inson umr davomiyligini bashorat qilish",
    version="1.0"
)

model = None

# --------------------------------------------------
# Load model on startup
# --------------------------------------------------
@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("Model muvaffaqiyatli yuklandi: %s", MODEL_PATH)
    except Exception as e:
        logger.error("Model yuklashda xatolik: %s", e)
        model = None

# --------------------------------------------------
# Input Schema (FEATURES)
# (life_span YO‘Q, chunki u TARGET)
# --------------------------------------------------
class LifeSpanInput(BaseModel):
    name: float
    birth_date: float
    birth_place: float
    death_date: float
    death_place: float
    occupation: float
    awards: float
    alma_mater: float
    education: float
    spouse: float
    children: float
    occupation_cluster: float
    birth_year: float
    death_year: float
    life_span_cluster: float
    edu_award_cluster: float
    bio_cluster: float

# --------------------------------------------------
# Output Schema
# --------------------------------------------------
class LifeSpanPrediction(BaseModel):
    predicted_life_span: float

# --------------------------------------------------
# Health endpoints
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "running",
        "model": "GradientBoostingRegressor",
        "task": "Life Span Prediction"
    }

@app.get("/health")
def health():
    return {"status": "health"}

# --------------------------------------------------
# Predict endpoint
# --------------------------------------------------
@app.post("/predict", response_model=LifeSpanPrediction)
def predict(data: LifeSpanInput):

    if model is None:
        raise HTTPException(status_code=500, detail="Model yuklanmagan")

    # Input -> DataFrame
    df = pd.DataFrame([data.model_dump()])

    try:
        prediction = model.predict(df)[0]
    except Exception as e:
        logger.error("Prediction xatoligi: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    logger.info("Predicted life span: %.2f years", prediction)

    return LifeSpanPrediction(
        predicted_life_span=round(float(prediction), 2)
    )
