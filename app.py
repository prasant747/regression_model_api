import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load model and encoder
model = joblib.load('insurance_model.joblib')
encoder = joblib.load('encodernew.joblib')

print('Model imported')

# Input schema
class InputFile(BaseModel):
    age: int
    sex: str
    bmi: float
    ped: int
    smoker: str
    region: str
    length_of_stay: float
    premium: int

app = FastAPI()

@app.post("/predict")
def inputdata(data: InputFile):  # ✅ Proper way
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])   # ✅ wrap inside list for single row

    # Encode categorical features
    df['sex'] = df['sex'].apply(lambda x: 1 if x.lower() == 'male' else 0)
    df['smoker'] = df['smoker'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    df['region'] = encoder.transform(df['region'])

    # Predict
    output = model.predict(df)
    return {"prediction": float(output[0])}
